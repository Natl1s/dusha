import argparse
import builtins
import json
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


def _exec_config(config_path: Path) -> dict:
    config_ns = {"__file__": str(config_path)}
    exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns


PROJECT_MY_EXP_DIR = Path(__file__).resolve().parents[2]

_train_data_config_ns = _exec_config(PROJECT_MY_EXP_DIR / "train_data.config")
DEFAULT_TRAIN_LMDB = Path(_train_data_config_ns["train_data_path"])

_test_data_config_ns = _exec_config(PROJECT_MY_EXP_DIR / "test_data.config")
DEFAULT_TEST_LMDB = Path(_test_data_config_ns["test_data_path"])

DEFAULT_AUDIO_SCALER_PATH = (
    PROJECT_MY_EXP_DIR
    / "audio_models"
    / "baseline"
    / "models_params"
    / "svm_combine_balanced_train_scaler.pkl"
)
DEFAULT_AUDIO_MODEL_PATH = DEFAULT_AUDIO_SCALER_PATH.with_name(
    DEFAULT_AUDIO_SCALER_PATH.name.replace("_scaler.pkl", "_model.pkl")
)

DEFAULT_TEXT_MODEL_PATH = (
    PROJECT_MY_EXP_DIR
    / "text_models"
    / "baseline"
    / "models_params"
    / "TF-IDF_LogReg_combine_balanced_train_model.pkl"
)
DEFAULT_TEXT_VECTORIZER_PATH = DEFAULT_TEXT_MODEL_PATH.with_name(
    DEFAULT_TEXT_MODEL_PATH.name.replace("_model.pkl", "_vectorizer.pkl")
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

TARGET_NAMES = ["angry", "sad", "neutral", "positive"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _extract_text(payload: dict) -> str:
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip().lower()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


def _to_fixed_audio_vector(feat: np.ndarray) -> np.ndarray:
    arr = np.asarray(feat)
    if arr.ndim == 0:
        raise ValueError("Пустой/скалярный feature-тензор")
    if arr.ndim == 1:
        return arr.astype(np.float32)
    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def _label_to_name(label) -> str:
    if isinstance(label, (np.integer, int)):
        idx = int(label)
        if 0 <= idx < len(TARGET_NAMES):
            return TARGET_NAMES[idx]
    return str(label).strip().lower()


def _align_proba_to_targets(model, probs: np.ndarray) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"Ожидается матрица вероятностей [N, C], получено {probs.shape}")
    if not hasattr(model, "classes_"):
        raise AttributeError("Модель не содержит classes_ и не может быть выровнена по TARGET_NAMES")

    class_to_col = {_label_to_name(cls): i for i, cls in enumerate(model.classes_)}
    aligned = np.zeros((probs.shape[0], len(TARGET_NAMES)), dtype=np.float32)
    for j, target in enumerate(TARGET_NAMES):
        if target not in class_to_col:
            raise ValueError(f"В модели отсутствует класс '{target}', classes_={model.classes_}")
        aligned[:, j] = probs[:, class_to_col[target]]
    return aligned


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    if np.any(denom == 0):
        raise ValueError("Не удалось нормализовать decision_function в вероятности (нулевой знаменатель)")
    return (exp_scores / denom).astype(np.float32)


def _predict_proba_or_decision(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float32)

    if not hasattr(model, "decision_function"):
        raise AttributeError(
            f"Модель {type(model).__name__} не поддерживает ни predict_proba, ни decision_function"
        )
    if not hasattr(model, "classes_"):
        raise AttributeError(f"Модель {type(model).__name__} не содержит classes_")

    scores = np.asarray(model.decision_function(X), dtype=np.float64)
    if scores.ndim == 1:
        clipped = np.clip(scores, -50.0, 50.0)
        p_pos = 1.0 / (1.0 + np.exp(-clipped))
        probs = np.column_stack([1.0 - p_pos, p_pos]).astype(np.float32)
    elif scores.ndim == 2:
        probs = _softmax(scores)
    else:
        raise ValueError(f"Некорректная форма decision_function: {scores.shape}")

    if probs.shape[1] != len(model.classes_):
        raise ValueError(
            "Число столбцов вероятностей не совпадает с числом классов модели: "
            f"{probs.shape[1]} vs {len(model.classes_)}"
        )
    return probs


class FusionLmdbDataset:
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)

        total = get_lmdb_length(self.env)
        valid_indices = []
        valid_labels = []
        with self.env.begin() as txn:
            for idx in tqdm(
                range(total),
                desc=f"Сканирование {self.lmdb_path.name}",
                unit="sample",
            ):
                raw = txn.get(str(idx).encode("utf-8"))
                if raw is None:
                    continue
                payload = pickle.loads(raw)
                if not isinstance(payload, dict):
                    continue

                text = _extract_text(payload)
                if not text:
                    continue

                if "x" not in payload:
                    continue
                try:
                    _to_fixed_audio_vector(np.asarray(payload["x"], dtype=np.float32))
                except (TypeError, ValueError):
                    continue

                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                try:
                    label = parse_label_to_index(label_raw)
                except ValueError:
                    continue

                valid_indices.append(idx)
                valid_labels.append(label)

        if not valid_indices:
            raise ValueError(f"В LMDB нет валидных мультимодальных примеров: {self.lmdb_path}")
        self.indices = np.asarray(valid_indices, dtype=np.int64)
        self.labels = np.asarray(valid_labels, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, item_idx: int):
        lmdb_idx = int(self.indices[item_idx])
        with self.env.begin() as txn:
            raw = txn.get(str(lmdb_idx).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {lmdb_idx}")

        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Некорректный payload у ключа {lmdb_idx}: ожидается dict")

        text = _extract_text(payload)
        if not text:
            raise ValueError(f"Пустой текст в валидном примере (idx={lmdb_idx})")

        if "x" not in payload:
            raise KeyError(f"В payload нет ключа 'x' (idx={lmdb_idx})")
        audio_vec = _to_fixed_audio_vector(np.asarray(payload["x"], dtype=np.float32))
        label = parse_label_to_index(payload.get("y", payload.get("label", payload.get("emotion"))))
        return audio_vec, text, int(label)


def _load_pickle(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def collect_model_probs(
    audio_model,
    audio_scaler,
    text_model,
    text_vectorizer,
    dataset: FusionLmdbDataset,
    item_indices: np.ndarray,
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    probs_audio_all = []
    probs_text_all = []

    for start in tqdm(range(0, len(item_indices), batch_size), desc=desc, unit="batch", leave=False):
        batch_ids = item_indices[start : start + batch_size]
        audio_batch = []
        text_batch = []
        y_batch = []
        for ds_idx in batch_ids:
            audio_vec, text, label = dataset[int(ds_idx)]
            audio_batch.append(audio_vec)
            text_batch.append(text)
            y_batch.append(label)

        X_audio = np.stack(audio_batch, axis=0)
        X_audio_scaled = audio_scaler.transform(X_audio)
        probs_audio = _predict_proba_or_decision(audio_model, X_audio_scaled)

        X_text = text_vectorizer.transform(text_batch)
        probs_text = _predict_proba_or_decision(text_model, X_text)

        probs_audio_all.append(_align_proba_to_targets(audio_model, np.asarray(probs_audio)))
        probs_text_all.append(_align_proba_to_targets(text_model, np.asarray(probs_text)))
        y_true_all.append(np.asarray(y_batch, dtype=np.int64))

    y_true = np.concatenate(y_true_all, axis=0)
    probs_audio = np.concatenate(probs_audio_all, axis=0)
    probs_text = np.concatenate(probs_text_all, axis=0)
    return y_true, probs_audio, probs_text


def evaluate_fusion(y_true: np.ndarray, probs_audio: np.ndarray, probs_text: np.ndarray, alpha: float) -> dict:
    fused_probs = alpha * probs_audio + (1.0 - alpha) * probs_text
    y_pred = np.argmax(fused_probs, axis=1)
    metrics = {
        "alpha": float(alpha),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return {"metrics": metrics, "y_pred": y_pred, "fused_probs": fused_probs}


def print_eval(title: str, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict) -> dict:
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        digits=4,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES))))

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(conf)

    return {
        "metrics": metrics,
        "classification_report_text": report_text,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(TARGET_NAMES))),
            target_names=TARGET_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": conf.tolist(),
    }


def _save_results(
    results_dir: Path,
    args,
    best_alpha: float,
    best_val_f1: float,
    alpha_search: list[dict],
    val_result: dict,
    test_result: dict,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"late_fusion_baseline_results_{stamp}.json"

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "best_alpha": float(best_alpha),
        "best_val_f1_macro": float(best_val_f1),
        "alpha_search": alpha_search,
        "args": {
            "train_lmdb": str(args.train_lmdb),
            "test_lmdb": str(args.test_lmdb),
            "audio_model_path": str(args.audio_model_path),
            "audio_scaler_path": str(args.audio_scaler_path),
            "text_model_path": str(args.text_model_path),
            "text_vectorizer_path": str(args.text_vectorizer_path),
            "batch_size": int(args.batch_size),
            "val_size": float(args.val_size),
            "alpha_step": float(args.alpha_step),
            "seed": int(args.seed),
        },
        "val": val_result,
        "test": test_result,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Late Fusion (soft voting) для baseline-моделей audio SVM + text TF-IDF LogReg."
    )
    parser.add_argument("--train-lmdb", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--test-lmdb", type=Path, default=DEFAULT_TEST_LMDB)
    parser.add_argument("--audio-scaler-path", type=Path, default=DEFAULT_AUDIO_SCALER_PATH)
    parser.add_argument("--audio-model-path", type=Path, default=DEFAULT_AUDIO_MODEL_PATH)
    parser.add_argument("--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument("--text-vectorizer-path", type=Path, default=DEFAULT_TEXT_VECTORIZER_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {args.train_lmdb}")
    if not args.test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {args.test_lmdb}")
    if not args.audio_model_path.exists():
        raise FileNotFoundError(f"Аудио модель не найдена: {args.audio_model_path}")
    if not args.audio_scaler_path.exists():
        raise FileNotFoundError(f"Аудио scaler не найден: {args.audio_scaler_path}")
    if not args.text_model_path.exists():
        raise FileNotFoundError(f"Текстовая модель не найдена: {args.text_model_path}")
    if not args.text_vectorizer_path.exists():
        raise FileNotFoundError(f"Текстовый векторизатор не найден: {args.text_vectorizer_path}")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError(f"val-size должен быть в (0,1), получено: {args.val_size}")
    if not (0.0 < args.alpha_step <= 1.0):
        raise ValueError(f"alpha-step должен быть в (0,1], получено: {args.alpha_step}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size должен быть > 0, получено: {args.batch_size}")

    set_seed(args.seed)
    print(f"Train LMDB: {args.train_lmdb}")
    print(f"Test LMDB:  {args.test_lmdb}")
    print(f"Audio model:  {args.audio_model_path}")
    print(f"Audio scaler: {args.audio_scaler_path}")
    print(f"Text model:   {args.text_model_path}")
    print(f"Text vectorizer: {args.text_vectorizer_path}")

    audio_model = _load_pickle(args.audio_model_path)
    audio_scaler = _load_pickle(args.audio_scaler_path)
    text_model = _load_pickle(args.text_model_path)
    text_vectorizer = _load_pickle(args.text_vectorizer_path)

    train_ds = FusionLmdbDataset(args.train_lmdb)
    test_ds = FusionLmdbDataset(args.test_lmdb)

    train_indices = np.arange(len(train_ds), dtype=np.int64)
    _, val_indices = train_test_split(
        train_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_ds.labels,
    )
    test_indices = np.arange(len(test_ds), dtype=np.int64)

    print(f"Размер train (валидные мультимодальные): {len(train_ds)}")
    print(f"Размер val: {len(val_indices)}")
    print(f"Размер test (валидные мультимодальные): {len(test_ds)}")

    y_val, p_audio_val, p_text_val = collect_model_probs(
        audio_model=audio_model,
        audio_scaler=audio_scaler,
        text_model=text_model,
        text_vectorizer=text_vectorizer,
        dataset=train_ds,
        item_indices=np.asarray(val_indices, dtype=np.int64),
        batch_size=args.batch_size,
        desc="Инференс val",
    )
    y_test, p_audio_test, p_text_test = collect_model_probs(
        audio_model=audio_model,
        audio_scaler=audio_scaler,
        text_model=text_model,
        text_vectorizer=text_vectorizer,
        dataset=test_ds,
        item_indices=test_indices,
        batch_size=args.batch_size,
        desc="Инференс test",
    )

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 4)
    best_alpha = None
    best_val_f1 = -1.0
    alpha_search = []
    print("\nПодбор alpha по val (macro-F1):")
    for alpha in tqdm(alphas, desc="Поиск alpha", unit="alpha"):
        val_eval = evaluate_fusion(y_val, p_audio_val, p_text_val, float(alpha))
        val_f1 = val_eval["metrics"]["f1_macro"]
        alpha_item = {"alpha": float(alpha), "val_f1_macro": float(val_f1)}
        alpha_search.append(alpha_item)
        tqdm.write(f"alpha={alpha:.2f} -> val_f1_macro={val_f1:.6f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_alpha = float(alpha)

    assert best_alpha is not None
    print(f"\nЛучший alpha: {best_alpha:.2f} (val_f1_macro={best_val_f1:.6f})")

    val_best = evaluate_fusion(y_val, p_audio_val, p_text_val, best_alpha)
    test_best = evaluate_fusion(y_test, p_audio_test, p_text_test, best_alpha)
    val_result = print_eval("LATE FUSION @ VAL", y_val, val_best["y_pred"], val_best["metrics"])
    test_result = print_eval("LATE FUSION @ TEST", y_test, test_best["y_pred"], test_best["metrics"])

    results_path = _save_results(
        results_dir=args.results_dir,
        args=args,
        best_alpha=best_alpha,
        best_val_f1=best_val_f1,
        alpha_search=alpha_search,
        val_result=val_result,
        test_result=test_result,
    )
    print(f"\nРезультаты сохранены: {results_path}")


if __name__ == "__main__":
    main()
