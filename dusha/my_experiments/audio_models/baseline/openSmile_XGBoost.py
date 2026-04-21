import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from my_experiments.lmdb_utils import load_feature_vectors_from_lmdb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _exec_config(config_path: Path) -> dict:
    config_ns = {"__file__": str(config_path)}
    exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = _exec_config(_data_config_path)
DATASET_PATH = _data_config_ns["base_path"]

_train_data_config_path = Path(__file__).parent.parent.parent / "train_data.config"
_train_data_config_ns = _exec_config(_train_data_config_path)
TRAIN_DATA_PATH = Path(_train_data_config_ns["train_data_path"])

_test_data_config_path = Path(__file__).parent.parent.parent / "test_data.config"
_test_data_config_ns = _exec_config(_test_data_config_path)
TEST_DATA_PATH = Path(_test_data_config_ns["test_data_path"])

# Путь для сохранения моделей
MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem  # openSmile_XGBoost

EMOTIONS = ["angry", "sad", "neutral", "positive"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

WAV_SUBDIRS = [
    Path("crowd_train") / "wavs",
    Path("crowd_test") / "wavs",
    Path("podcast_train") / "wavs",
    Path("podcast_test") / "wavs",
]


def weighted_accuracy(y_true, y_pred, labels=EMOTIONS):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for lbl in labels:
        mask = y_true == lbl
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == lbl).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def _to_emotion_labels(y):
    """Приводит массив меток к строковым эмоциям из EMOTIONS."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in {"i", "u"}:
        idx_arr = y_arr.astype(np.int64, copy=False)
        invalid = (idx_arr < 0) | (idx_arr >= len(EMOTIONS))
        if np.any(invalid):
            raise ValueError(
                f"Обнаружены индексы классов вне диапазона [0, {len(EMOTIONS) - 1}]: "
                f"{np.unique(idx_arr[invalid]).tolist()}"
            )
        return np.array([EMOTIONS[i] for i in idx_arr], dtype=object)

    labels = y_arr.astype(str, copy=False)
    unknown = sorted({label for label in np.unique(labels) if label not in EMOTION_TO_IDX})
    if unknown:
        raise ValueError(f"Обнаружены неизвестные эмоции: {unknown}")
    return labels


def _to_emotion_indices(y):
    labels = _to_emotion_labels(y)
    return np.array([EMOTION_TO_IDX[label] for label in labels], dtype=np.int64)


def _align_proba_to_emotions(model, proba):
    """Переставляет столбцы predict_proba в фиксированном порядке EMOTIONS."""
    if proba.ndim != 2:
        raise ValueError(f"Ожидается 2D-массив вероятностей, получено shape={proba.shape}")

    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError("У модели отсутствует classes_ — невозможно выровнять вероятности")

    class_labels = _to_emotion_labels(model_classes)
    aligned = np.zeros((proba.shape[0], len(EMOTIONS)), dtype=proba.dtype)
    for src_idx, class_label in enumerate(class_labels):
        aligned[:, EMOTION_TO_IDX[class_label]] = proba[:, src_idx]
    return aligned


def save_model(
    model,
    dataset_name,
    model_name=MODEL_NAME,
    training_params=None,
    test_metrics=None,
):
    """Сохраняет модель в файл"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    model_path_timestamped = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pkl"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    # Сохранение основных файлов
    joblib.dump(model, model_path)

    # Сохранение с временной меткой (для истории)
    joblib.dump({"model": model}, model_path_timestamped)

    report_lines = [
        f"model_name: {model_name}",
        f"dataset_name: {dataset_name}",
        f"saved_at: {timestamp}",
        "",
        "training_params:",
        json.dumps(training_params or {}, ensure_ascii=False, indent=2),
        "",
        "test_metrics:",
        json.dumps(test_metrics or {}, ensure_ascii=False, indent=2),
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'=' * 60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Бэкап:  {model_path_timestamped.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")


def load_model(dataset_name, model_name=MODEL_NAME):
    """Загружает модель из файла"""
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена! Проверьте наличие файла:\n"
            f"  {model_path}"
        )

    model = joblib.load(model_path)

    print(f"✓ Модель загружена из {model_path}")
    return model


def model_exists(dataset_name, model_name=MODEL_NAME):
    """Проверяет существование сохраненной модели"""
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    return model_path.exists()


def get_dataset_name(train_manifest_path):
    """Извлекает имя датасета из пути к манифесту"""
    return Path(train_manifest_path).stem


def _init_opensmile_extractor():
    try:
        import opensmile
    except ImportError as exc:
        raise ImportError(
            "Для этого baseline требуется opensmile.\n"
            "Установите: poetry add opensmile\n"
            "или: pip install opensmile"
        ) from exc

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _resolve_wav_path(row, dataset_root):
    if "audio_path" in row and row["audio_path"]:
        path = Path(row["audio_path"])
        if not path.is_absolute():
            path = dataset_root / path
        if path.exists():
            return path

    key = row.get("hash_id", row.get("id"))
    if key:
        for subdir in WAV_SUBDIRS:
            candidate = dataset_root / subdir / f"{key}.wav"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Не удалось найти wav для sample. "
        f"Доступные поля: {sorted(row.keys())}; hash/id={row.get('hash_id', row.get('id'))}"
    )


def _extract_focus_stats(feature_names, vector):
    """Агрегирует ключевые группы eGeMAPS: pitch/energy/jitter/shimmer."""
    idx = {name: i for i, name in enumerate(feature_names)}

    def _group_mean(substrings):
        vals = [float(vector[i]) for name, i in idx.items() if any(s in name for s in substrings)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "pitch_group_mean": _group_mean(["F0semitone", "F0final"]),
        "energy_group_mean": _group_mean(["loudness", "HNR", "alphaRatio", "slope", "spectralFlux"]),
        "jitter_group_mean": _group_mean(["jitter"]),
        "shimmer_group_mean": _group_mean(["shimmer"]),
    }


def _to_fixed_vector(feat: np.ndarray) -> np.ndarray:
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def load_opensmile_features_from_manifest(
    manifest_path,
    dataset_root,
    cache_path=None,
    max_samples=None,
    progress_desc=None,
):
    """Загрузка eGeMAPS признаков из аудио по JSONL манифесту."""
    manifest_path = Path(manifest_path)
    if manifest_path.suffix == ".lmdb":
        X, y = load_feature_vectors_from_lmdb(
            lmdb_path=manifest_path,
            vectorize_fn=_to_fixed_vector,
            label_kind="emotion",
        )
        feature_names = [f"lmdb_feature_{idx}" for idx in range(X.shape[1])]
        focus_stats_rows = []
        return X, y, feature_names, focus_stats_rows, 0

    if cache_path and Path(cache_path).exists():
        cached = joblib.load(cache_path)
        return (
            cached["X"],
            cached["y"],
            cached["feature_names"],
            cached["focus_stats"],
            cached["n_skipped"],
        )

    smile = _init_opensmile_extractor()
    features = []
    labels = []
    feature_names = None
    focus_stats_rows = []
    n_skipped = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        line_iterator = tqdm(
            f,
            total=total_lines,
            desc=progress_desc,
            unit="sample",
        )
        for line_idx, line in enumerate(line_iterator, 1):
            if max_samples is not None and len(labels) >= max_samples:
                break

            row = json.loads(line.strip())
            emotion = row.get("emotion")
            if emotion not in EMOTIONS:
                n_skipped += 1
                continue

            try:
                wav_path = _resolve_wav_path(row, dataset_root=dataset_root)
                df_feat = smile.process_file(str(wav_path))
                if df_feat.empty:
                    raise ValueError(f"OpenSMILE вернул пустые признаки: {wav_path}")
                if feature_names is None:
                    feature_names = list(df_feat.columns)
                vec = df_feat.iloc[0].to_numpy(dtype=np.float32)
                features.append(vec)
                labels.append(emotion)
                focus_stats_rows.append(_extract_focus_stats(feature_names, vec))
            except Exception as exc:
                n_skipped += 1
                if line_idx <= 5:
                    print(f"⚠ Пропуск sample (строка {line_idx}): {exc}")
                continue

    if not features:
        raise ValueError(
            f"Не удалось извлечь признаки из {manifest_path}. "
            "Проверьте доступность аудио и зависимость opensmile."
        )

    X = np.vstack(features)
    y = np.array(labels)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "focus_stats": focus_stats_rows,
                "n_skipped": n_skipped,
            },
            cache_path,
        )

    return X, y, feature_names, focus_stats_rows, n_skipped


def _build_classifier(
    model_type="auto",
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
):
    model_type = model_type.lower()
    errors = []

    def _try_xgboost():
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(EMOTIONS),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
        ), "xgboost"

    def _try_lightgbm():
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(EMOTIONS),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=random_state,
            n_jobs=-1,
        ), "lightgbm"

    if model_type in {"xgboost", "auto"}:
        try:
            return _try_xgboost()
        except Exception as exc:
            errors.append(f"xgboost: {exc}")
            if model_type == "xgboost":
                raise
    if model_type in {"lightgbm", "auto"}:
        try:
            return _try_lightgbm()
        except Exception as exc:
            errors.append(f"lightgbm: {exc}")
            if model_type == "lightgbm":
                raise

    raise ImportError(
        "Не удалось инициализировать ни XGBoost, ни LightGBM.\n"
        "Установите хотя бы одну библиотеку:\n"
        "  poetry add xgboost\n"
        "  poetry add lightgbm\n"
        f"Ошибки: {errors}"
    )


def print_model_parameters(model, resolved_model_name, feature_names):
    """Вывод критичных параметров бустинг-модели"""
    print(f"\n{'=' * 60}")
    print("ПАРАМЕТРЫ БУСТИНГ-МОДЕЛИ")
    print(f"{'=' * 60}")
    print(f"Реализация: {resolved_model_name}")
    print(f"Классов: {len(model.classes_)}")
    try:
        print(f"Классы: {_to_emotion_labels(model.classes_)}")
    except ValueError:
        print(f"Классы: {model.classes_}")
    print(f"n_estimators: {getattr(model, 'n_estimators', 'N/A')}")
    print(f"learning_rate: {getattr(model, 'learning_rate', 'N/A')}")
    print(f"max_depth: {getattr(model, 'max_depth', 'N/A')}")
    print(f"Количество eGeMAPS признаков: {len(feature_names)}")


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, focus_stats_train):
    """Оценка модели на обучающей и тестовой выборках"""
    resolved_model_name = model.__class__.__name__
    print_model_parameters(model, resolved_model_name, feature_names)

    y_train_labels = _to_emotion_labels(y_train)
    y_test_labels = _to_emotion_labels(y_test)

    # Оценка на обучающей выборке
    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    train_pred = _to_emotion_labels(model.predict(X_train))
    print(
        classification_report(
            y_train_labels,
            train_pred,
            labels=EMOTIONS,
            target_names=EMOTIONS,
            zero_division=0,
        )
    )

    # Оценка на тестовой выборке
    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    test_pred = _to_emotion_labels(model.predict(X_test))
    test_proba = _align_proba_to_emotions(model, model.predict_proba(X_test))
    test_report_text = classification_report(
        y_test_labels,
        test_pred,
        labels=EMOTIONS,
        target_names=EMOTIONS,
        zero_division=0,
    )
    print(test_report_text)

    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test_labels, test_pred, labels=EMOTIONS)
    print(test_cm)

    test_report_dict = classification_report(
        y_test_labels,
        test_pred,
        labels=EMOTIONS,
        target_names=EMOTIONS,
        zero_division=0,
        output_dict=True,
    )

    y_test_idx = _to_emotion_indices(y_test_labels)
    y_pred_idx = _to_emotion_indices(test_pred)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test_labels, test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test_labels, test_pred)),
        "test_f1_macro": float(
            f1_score(y_test_labels, test_pred, labels=EMOTIONS, average="macro", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(y_test_labels, test_pred, labels=EMOTIONS, average="weighted", zero_division=0)
        ),
        "test_precision_macro": float(
            precision_score(y_test_labels, test_pred, labels=EMOTIONS, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test_labels, test_pred, labels=EMOTIONS, average="macro", zero_division=0)
        ),
        "test_UAR": float(recall_score(y_test_labels, test_pred, labels=EMOTIONS, average="macro", zero_division=0)),
        "test_WA": float(weighted_accuracy(y_test_labels, test_pred, labels=EMOTIONS)),
        "test_mcc": float(matthews_corrcoef(y_test_labels, test_pred)),
        "test_cohen_kappa_qwk": float(
            cohen_kappa_score(y_test_labels, test_pred, labels=EMOTIONS, weights="quadratic")
        ),
        "test_top2_accuracy": float(
            top_k_accuracy_score(y_test_idx, test_proba, k=2, labels=list(range(len(EMOTIONS))))
        ),
        "test_log_loss": float(log_loss(y_test_idx, test_proba, labels=list(range(len(EMOTIONS))))),
        "test_classification_report_text": test_report_text,
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm.tolist(),
        "train_focus_feature_means": {
            "pitch_group_mean": float(np.nanmean([r["pitch_group_mean"] for r in focus_stats_train])),
            "energy_group_mean": float(np.nanmean([r["energy_group_mean"] for r in focus_stats_train])),
            "jitter_group_mean": float(np.nanmean([r["jitter_group_mean"] for r in focus_stats_train])),
            "shimmer_group_mean": float(np.nanmean([r["shimmer_group_mean"] for r in focus_stats_train])),
        },
    }
    try:
        metrics["test_roc_auc_ovr_macro"] = float(
            roc_auc_score(y_test_idx, test_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["test_roc_auc_ovr_macro"] = float("nan")

    if hasattr(model, "feature_importances_"):
        top_idx = np.argsort(model.feature_importances_)[-20:][::-1]
        metrics["top_20_feature_importances"] = [
            {"feature": feature_names[i], "importance": float(model.feature_importances_[i])}
            for i in top_idx
        ]

    return metrics


def train_opensmile_boosting(
    save=True,
    model_type="auto",
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    max_train_samples=None,
    max_test_samples=None,
    use_cache=True,
):
    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    train_cache = None
    test_cache = None
    if use_cache:
        cache_dir = MODELS_DIR / "feature_cache"
        train_cache = cache_dir / f"{MODEL_NAME}_{dataset_name}_train.joblib"
        test_cache = cache_dir / f"{MODEL_NAME}_{dataset_name}_test.joblib"

    print("Извлечение OpenSMILE eGeMAPS признаков для train...")
    X_train, y_train, feature_names, focus_stats_train, n_train_skipped = load_opensmile_features_from_manifest(
        train_manifest,
        dataset_root=DATASET_PATH,
        cache_path=train_cache,
        max_samples=max_train_samples,
        progress_desc="OpenSMILE train",
    )
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Пропущено в train: {n_train_skipped}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")

    print("\nИзвлечение OpenSMILE eGeMAPS признаков для test...")
    X_test, y_test, _, _, n_test_skipped = load_opensmile_features_from_manifest(
        test_manifest,
        dataset_root=DATASET_PATH,
        cache_path=test_cache,
        max_samples=max_test_samples,
        progress_desc="OpenSMILE test",
    )
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Пропущено в test: {n_test_skipped}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    print(f"\n{'=' * 60}")
    print("ОБУЧЕНИЕ БУСТИНГ-МОДЕЛИ (OpenSMILE eGeMAPS)")
    print(f"{'=' * 60}")
    print(
        f"Параметры: model_type={model_type}, n_estimators={n_estimators}, "
        f"learning_rate={learning_rate}, max_depth={max_depth}, random_state={random_state}"
    )

    model, resolved_model_name = _build_classifier(
        model_type=model_type,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    model.fit(X_train, _to_emotion_indices(y_train))
    print(f"✓ Обучение завершено! ({resolved_model_name})")

    # Оценка модели
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, focus_stats_train)

    # Сохранение модели
    if save:
        training_params = {
            "resolved_model_name": resolved_model_name,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
            "feature_set": "OpenSMILE eGeMAPSv02 Functionals",
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "dataset_root": str(DATASET_PATH),
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "use_cache": use_cache,
        }
        save_model(
            model,
            dataset_name,
            training_params=training_params,
            test_metrics=metrics,
        )

    return model, dataset_name


def load_and_evaluate(
    model_type="auto",
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    max_train_samples=None,
    max_test_samples=None,
    use_cache=True,
):
    """Загрузить существующую модель и оценить её"""
    print(f"{'=' * 60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'=' * 60}")

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    model = load_model(dataset_name)

    train_cache = None
    test_cache = None
    if use_cache:
        cache_dir = MODELS_DIR / "feature_cache"
        train_cache = cache_dir / f"{MODEL_NAME}_{dataset_name}_train.joblib"
        test_cache = cache_dir / f"{MODEL_NAME}_{dataset_name}_test.joblib"

    print("\nИзвлечение train-признаков...")
    X_train, y_train, feature_names, focus_stats_train, _ = load_opensmile_features_from_manifest(
        train_manifest,
        dataset_root=DATASET_PATH,
        cache_path=train_cache,
        max_samples=max_train_samples,
        progress_desc="OpenSMILE train",
    )
    print(f"Количество обучающих примеров: {len(y_train)}")

    print("\nИзвлечение test-признаков...")
    X_test, y_test, _, _, _ = load_opensmile_features_from_manifest(
        test_manifest,
        dataset_root=DATASET_PATH,
        cache_path=test_cache,
        max_samples=max_test_samples,
        progress_desc="OpenSMILE test",
    )
    print(f"Количество тестовых примеров: {len(y_test)}")

    # Оценка модели
    evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, focus_stats_train)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка OpenSMILE(eGeMAPS)+XGBoost/LightGBM для классификации эмоций"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto"],
        default="auto",
        help="Режим работы: train - обучить новую модель, load - загрузить существующую, auto - загрузить если есть, иначе обучить",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять модель после обучения",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "xgboost", "lightgbm"],
        default="auto",
        help="Тип модели бустинга: auto/xgboost/lightgbm",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Количество деревьев",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Максимальная глубина деревьев",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Ограничение количества train-сэмплов (для быстрых прогонов)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Ограничение количества test-сэмплов (для быстрых прогонов)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Не использовать кэш OpenSMILE признаков",
    )

    args = parser.parse_args()

    # Получение имени датасета для проверки существования модели
    dataset_name = get_dataset_name(TRAIN_DATA_PATH)

    common_kwargs = {
        "model_type": args.model_type,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "use_cache": not args.no_cache,
    }

    # Определение режима работы
    if args.mode == "train":
        print("🎯 Режим: Обучение новой модели\n")
        model, _ = train_opensmile_boosting(save=not args.no_save, **common_kwargs)
    elif args.mode == "load":
        print("📂 Режим: Загрузка существующей модели\n")
        model = load_and_evaluate(**common_kwargs)
    else:  # auto
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            model = load_and_evaluate(**common_kwargs)
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            model, _ = train_opensmile_boosting(save=not args.no_save, **common_kwargs)
