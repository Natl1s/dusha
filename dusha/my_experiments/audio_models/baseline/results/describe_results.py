import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.svm import SVC


# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns["base_path"]

EMOTIONS = ["angry", "sad", "neutral", "positive"]
MODEL_PREFIXES = ["logictic_regressoin", "svm", "random_forest"]


@dataclass
class EvaluationResult:
    model_key: str
    model_type: str
    params_text: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_report_text: str
    test_report_text: str
    test_report_dict: Dict[str, Dict[str, float]]
    train_cm: np.ndarray
    test_cm: np.ndarray
    extra_text: str


def _to_fixed_vector(feat: np.ndarray) -> np.ndarray:
    """Преобразует feature-тензор в вектор фиксированной длины."""
    arr = np.asarray(feat)
    if arr.ndim == 0:
        raise ValueError("Пустой/скалярный feature-тензор")
    if arr.ndim == 1:
        return arr.astype(np.float32)
    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def load_features_from_manifest(manifest_path: Path, base_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка features из JSONL манифеста."""
    features: List[np.ndarray] = []
    labels: List[str] = []
    expected_dim: Optional[int] = None

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            npy_path = data["hash_id"] + ".npy"
            feature_path = Path("features") / npy_path
            if base_path is not None:
                feature_path = base_path / feature_path

            feat = np.load(feature_path)
            feat_flat = _to_fixed_vector(feat)

            if expected_dim is None:
                expected_dim = feat_flat.shape[0]
            elif feat_flat.shape[0] != expected_dim:
                sample_id = data.get("id", "<unknown>")
                raise ValueError(
                    f"Несовпадение размерности признаков для id={sample_id}: "
                    f"получено {feat_flat.shape[0]}, ожидалось {expected_dim}. Файл: {feature_path}"
                )

            features.append(feat_flat)
            labels.append(data["emotion"])

    return np.stack(features), np.array(labels)


def resolve_models_dir() -> Path:
    """Находит директорию с параметрами baseline-моделей."""
    baseline_dir = Path(__file__).parent.parent
    candidates = [
        baseline_dir / "model_params",   # путь из запроса
        baseline_dir / "models_params",  # фактический путь в проекте
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Не найдена директория с моделями. Ожидались: "
        + ", ".join(str(p) for p in candidates)
    )


def _latest_by_mtime(paths: List[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def _is_timestamped_backup(path: Path) -> bool:
    return re.search(r"_model_\d{8}_\d{6}\.pkl$", path.name) is not None


def _prefer_canonical_model_path(paths: List[Path]) -> Path:
    """Выбирает основной файл модели (*_model.pkl) раньше бэкапов с timestamp."""
    return min(
        paths,
        key=lambda p: (
            1 if _is_timestamped_backup(p) else 0,
            -p.stat().st_mtime,
        ),
    )


def _unwrap_model_payload(model_obj, scaler_obj):
    """
    Поддерживает два формата model.pkl:
    1) sklearn-оценщик;
    2) dict с ключами model/scaler (timestamp-бэкапы).
    """
    if isinstance(model_obj, dict):
        payload = model_obj
        if "model" in payload:
            model_obj = payload["model"]
        elif "estimator" in payload:
            model_obj = payload["estimator"]

        if "scaler" in payload:
            # Используем scaler из bundle, если он был сохранен вместе с моделью.
            scaler_obj = payload["scaler"]

    if not hasattr(model_obj, "predict"):
        raise TypeError(
            f"Загруженный объект модели имеет тип {type(model_obj).__name__} и не поддерживает predict()."
        )
    if not hasattr(scaler_obj, "transform"):
        raise TypeError(
            f"Загруженный объект scaler имеет тип {type(scaler_obj).__name__} и не поддерживает transform()."
        )
    return model_obj, scaler_obj


def _collect_variants(models_dir: Path, model_prefix: str) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Собирает доступные варианты dataset для model/scaler по выбранному префиксу."""
    model_variants: Dict[str, List[Path]] = {}
    scaler_variants: Dict[str, List[Path]] = {}

    for path in models_dir.glob(f"{model_prefix}_*_model*.pkl"):
        name = path.name
        if not name.startswith(f"{model_prefix}_"):
            continue
        tail = name[len(model_prefix) + 1 :]
        split_idx = tail.find("_model")
        if split_idx == -1:
            continue
        variant = tail[:split_idx]
        model_variants.setdefault(variant, []).append(path)

    for path in models_dir.glob(f"{model_prefix}_*_scaler*.pkl"):
        name = path.name
        if not name.startswith(f"{model_prefix}_"):
            continue
        tail = name[len(model_prefix) + 1 :]
        split_idx = tail.find("_scaler")
        if split_idx == -1:
            continue
        variant = tail[:split_idx]
        scaler_variants.setdefault(variant, []).append(path)

    latest_models = {k: _prefer_canonical_model_path(v) for k, v in model_variants.items()}
    latest_scalers = {k: _latest_by_mtime(v) for k, v in scaler_variants.items()}
    return latest_models, latest_scalers


def load_model_and_scaler(models_dir: Path, model_prefix: str, dataset_name: str):
    model_path = models_dir / f"{model_prefix}_{dataset_name}_model.pkl"
    scaler_path = models_dir / f"{model_prefix}_{dataset_name}_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        model_obj, scaler_obj = _unwrap_model_payload(joblib.load(model_path), joblib.load(scaler_path))
        return model_obj, scaler_obj, model_path, scaler_path

    latest_models, latest_scalers = _collect_variants(models_dir, model_prefix)
    common_variants = set(latest_models) & set(latest_scalers)
    if common_variants:
        if dataset_name in common_variants:
            selected_variant = dataset_name
        else:
            selected_variant = None
            startswith_matches = sorted(v for v in common_variants if v.startswith(dataset_name))
            if startswith_matches:
                # Предпочитаем минимальное отклонение от запрошенного dataset (например, *_small).
                selected_variant = min(startswith_matches, key=len)
            else:
                selected_variant = sorted(common_variants)[0]

        model_path = latest_models[selected_variant]
        scaler_path = latest_scalers[selected_variant]
        print(
            f"Предупреждение: точная пара для dataset='{dataset_name}' не найдена. "
            f"Используется ближайший вариант: '{selected_variant}'."
        )
        model_obj, scaler_obj = _unwrap_model_payload(joblib.load(model_path), joblib.load(scaler_path))
        return model_obj, scaler_obj, model_path, scaler_path

    available_models = sorted(str(p.name) for p in models_dir.glob(f"{model_prefix}_*_model*.pkl"))
    available_scalers = sorted(str(p.name) for p in models_dir.glob(f"{model_prefix}_*_scaler*.pkl"))
    raise FileNotFoundError(
        f"Не найдены согласованные model/scaler для '{model_prefix}' и dataset='{dataset_name}'.\n"
        f"Ожидались (точно):\n  {model_path}\n  {scaler_path}\n"
        f"Доступные model-файлы:\n  " + ("\n  ".join(available_models) if available_models else "<нет>") + "\n"
        f"Доступные scaler-файлы:\n  " + ("\n  ".join(available_scalers) if available_scalers else "<нет>")
    )


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTIONS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=EMOTIONS, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def describe_model_params(model) -> Tuple[str, str]:
    model_type = type(model).__name__
    lines: List[str] = [f"Тип модели: {model_type}"]
    extra: List[str] = []

    if isinstance(model, LogisticRegression):
        lines += [
            f"solver={model.solver}, C={model.C}, max_iter={model.max_iter}, random_state={model.random_state}",
            f"n_iter={model.n_iter_}",
            f"coef_shape={model.coef_.shape}",
        ]
        coef_abs = np.abs(model.coef_).mean(axis=0)
        top_idx = np.argsort(coef_abs)[-10:][::-1]
        extra.append("Топ-10 наиболее значимых признаков по |coef| (среднее по классам):")
        for rank, idx in enumerate(top_idx, 1):
            extra.append(f"  {rank:2d}. feature_{idx}: {coef_abs[idx]:.6f}")

    elif isinstance(model, SVC):
        lines += [
            f"kernel={model.kernel}, C={model.C}, gamma={model.gamma}",
            f"n_support_total={int(model.n_support_.sum())}",
            f"n_support_by_class={dict(zip(model.classes_, model.n_support_))}",
        ]
        extra.append(
            f"dual_coef stats: min={model.dual_coef_.min():.6f}, max={model.dual_coef_.max():.6f}, "
            f"mean={model.dual_coef_.mean():.6f}, std={model.dual_coef_.std():.6f}"
        )

    elif isinstance(model, RandomForestClassifier):
        lines += [
            f"n_estimators={model.n_estimators}, criterion={model.criterion}, max_features={model.max_features}",
            f"max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, min_samples_leaf={model.min_samples_leaf}",
        ]
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        extra.append("Топ-10 наиболее значимых признаков по feature_importances_:")
        for rank, idx in enumerate(top_idx, 1):
            extra.append(f"  {rank:2d}. feature_{idx}: {importances[idx]:.6f}")
        extra.append(
            f"feature_importances stats: min={importances.min():.6f}, max={importances.max():.6f}, "
            f"mean={importances.mean():.6f}, std={importances.std():.6f}"
        )
    else:
        lines.append("Параметры: нет отдельного обработчика для данного типа модели")

    return "\n".join(lines), "\n".join(extra)


def evaluate_single_model(
    model_key: str,
    model,
    scaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> EvaluationResult:
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_metrics = get_metrics(y_train, train_pred)
    test_metrics = get_metrics(y_test, test_pred)

    train_report_text = classification_report(y_train, train_pred, labels=EMOTIONS, target_names=EMOTIONS, zero_division=0)
    test_report_text = classification_report(y_test, test_pred, labels=EMOTIONS, target_names=EMOTIONS, zero_division=0)
    test_report_dict = classification_report(
        y_test, test_pred, labels=EMOTIONS, target_names=EMOTIONS, zero_division=0, output_dict=True
    )

    train_cm = confusion_matrix(y_train, train_pred, labels=EMOTIONS)
    test_cm = confusion_matrix(y_test, test_pred, labels=EMOTIONS)
    params_text, extra_text = describe_model_params(model)

    return EvaluationResult(
        model_key=model_key,
        model_type=type(model).__name__,
        params_text=params_text,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_report_text=train_report_text,
        test_report_text=test_report_text,
        test_report_dict=test_report_dict,
        train_cm=train_cm,
        test_cm=test_cm,
        extra_text=extra_text,
    )


def save_text_report(output_dir: Path, result: EvaluationResult) -> None:
    report_path = output_dir / f"{result.model_key}_detailed_report.txt"
    lines = [
        "=" * 80,
        f"MODEL: {result.model_key}",
        "=" * 80,
        "",
        "ПАРАМЕТРЫ МОДЕЛИ:",
        result.params_text,
        "",
        "ДОПОЛНИТЕЛЬНЫЕ ОЦЕНКИ ВАЖНОСТИ:",
        result.extra_text or "Нет дополнительных оценок",
        "",
        "TRAIN METRICS:",
    ]
    for k, v in result.train_metrics.items():
        lines.append(f"  {k}: {v:.6f}")
    lines += [
        "",
        "TEST METRICS:",
    ]
    for k, v in result.test_metrics.items():
        lines.append(f"  {k}: {v:.6f}")
    lines += [
        "",
        "TRAIN CLASSIFICATION REPORT:",
        result.train_report_text,
        "",
        "TEST CLASSIFICATION REPORT:",
        result.test_report_text,
        "",
        "TRAIN CONFUSION MATRIX:",
        np.array2string(result.train_cm),
        "",
        "TEST CONFUSION MATRIX:",
        np.array2string(result.test_cm),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_csv(output_dir: Path, result: EvaluationResult) -> None:
    for split, cm in [("train", result.train_cm), ("test", result.test_cm)]:
        path = output_dir / f"{result.model_key}_{split}_confusion_matrix.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("true\\pred," + ",".join(EMOTIONS) + "\n")
            for i, label in enumerate(EMOTIONS):
                row = ",".join(str(int(x)) for x in cm[i])
                f.write(f"{label},{row}\n")


def plot_metric_comparison(results: List[EvaluationResult], output_dir: Path) -> None:
    model_names = [r.model_key for r in results]
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    x = np.arange(len(model_names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, split in enumerate(["train_metrics", "test_metrics"]):
        ax = axes[ax_idx]
        for i, metric in enumerate(metrics):
            values = [getattr(r, split)[metric] for r in results]
            ax.bar(x + (i - 1.5) * width, values, width=width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("score")
        ax.set_title("Train metrics" if split == "train_metrics" else "Test metrics")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_f1_comparison(results: List[EvaluationResult], output_dir: Path) -> None:
    model_names = [r.model_key for r in results]
    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, emotion in enumerate(EMOTIONS):
        values = [r.test_report_dict[emotion]["f1-score"] for r in results]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=emotion)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 on test set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_test_confusions(results: List[EvaluationResult], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        cm = result.test_cm
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(EMOTIONS)))
        ax.set_yticks(np.arange(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS, rotation=35, ha="right")
        ax.set_yticklabels(EMOTIONS)
        ax.set_title(result.model_key)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_comparison_tables(results: List[EvaluationResult], output_dir: Path) -> None:
    csv_path = output_dir / "model_comparison_metrics.csv"
    headers = [
        "model",
        "type",
        "train_accuracy",
        "train_balanced_accuracy",
        "train_f1_macro",
        "train_f1_weighted",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r.model_key,
                r.model_type,
                f"{r.train_metrics['accuracy']:.6f}",
                f"{r.train_metrics['balanced_accuracy']:.6f}",
                f"{r.train_metrics['f1_macro']:.6f}",
                f"{r.train_metrics['f1_weighted']:.6f}",
                f"{r.test_metrics['accuracy']:.6f}",
                f"{r.test_metrics['balanced_accuracy']:.6f}",
                f"{r.test_metrics['f1_macro']:.6f}",
                f"{r.test_metrics['f1_weighted']:.6f}",
            ]
            f.write(",".join(row) + "\n")


def pick_best_model(results: List[EvaluationResult]) -> EvaluationResult:
    # Основной критерий: test f1_macro
    # Доп. критерии при равенстве: test balanced_accuracy -> test accuracy
    return max(
        results,
        key=lambda r: (
            r.test_metrics["f1_macro"],
            r.test_metrics["balanced_accuracy"],
            r.test_metrics["accuracy"],
        ),
    )


def save_best_summary(best: EvaluationResult, results: List[EvaluationResult], output_dir: Path) -> None:
    sorted_results = sorted(
        results,
        key=lambda r: (r.test_metrics["f1_macro"], r.test_metrics["balanced_accuracy"], r.test_metrics["accuracy"]),
        reverse=True,
    )
    lines = [
        "=" * 80,
        "ИТОГОВОЕ СРАВНЕНИЕ BASELINE-МОДЕЛЕЙ",
        "=" * 80,
        "",
        "Критерий выбора лучшей модели:",
        "1) максимальный test f1_macro",
        "2) при равенстве: максимальный test balanced_accuracy",
        "3) при равенстве: максимальный test accuracy",
        "",
        "РАНЖИРОВАНИЕ МОДЕЛЕЙ (по test quality):",
    ]
    for i, r in enumerate(sorted_results, 1):
        lines.append(
            f"{i}. {r.model_key}: f1_macro={r.test_metrics['f1_macro']:.6f}, "
            f"balanced_accuracy={r.test_metrics['balanced_accuracy']:.6f}, "
            f"accuracy={r.test_metrics['accuracy']:.6f}, "
            f"f1_weighted={r.test_metrics['f1_weighted']:.6f}"
        )
    lines += [
        "",
        "ЛУЧШАЯ МОДЕЛЬ:",
        f"{best.model_key} ({best.model_type})",
        "",
        "Подробные метрики лучшей модели на TEST:",
    ]
    for k, v in best.test_metrics.items():
        lines.append(f"  {k}: {v:.6f}")
    lines += [
        "",
        "Per-class test metrics для лучшей модели:",
    ]
    for emotion in EMOTIONS:
        vals = best.test_report_dict[emotion]
        lines.append(
            f"  {emotion:8s}: precision={vals['precision']:.6f}, "
            f"recall={vals['recall']:.6f}, f1={vals['f1-score']:.6f}, support={int(vals['support'])}"
        )
    (output_dir / "summary_best_model.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Детальная оценка сохраненных baseline-моделей и выбор лучшей."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="combine_balanced_train",
        help="Имя train-датасета (без .jsonl), по умолчанию combine_balanced_train",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="combine_balanced_test",
        help="Имя test-датасета (без .jsonl), по умолчанию combine_balanced_test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Директория сохранения результатов (txt/csv/png). По умолчанию audio_models/baseline/results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = resolve_models_dir()

    base_path = DATASET_PATH / "processed_dataset_090"
    train_manifest = base_path / "aggregated_dataset" / f"{args.dataset}.jsonl"
    test_manifest = base_path / "aggregated_dataset" / f"{args.test_dataset}.jsonl"
    if not train_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError(
            f"Манифесты не найдены:\n  train={train_manifest}\n  test={test_manifest}"
        )

    print(f"Используется директория моделей: {models_dir}")
    print(f"Train manifest: {train_manifest}")
    print(f"Test manifest:  {test_manifest}")
    print(f"Результаты будут сохранены в: {output_dir}")

    print("\nЗагрузка train/test признаков...")
    X_train, y_train = load_features_from_manifest(train_manifest, base_path.parent)
    X_test, y_test = load_features_from_manifest(test_manifest, base_path.parent)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    all_results: List[EvaluationResult] = []
    for model_key in MODEL_PREFIXES:
        print(f"\n{'=' * 80}")
        print(f"ОЦЕНКА МОДЕЛИ: {model_key}")
        print(f"{'=' * 80}")
        model, scaler, model_path, scaler_path = load_model_and_scaler(models_dir, model_key, args.dataset)
        print(f"Модель: {model_path}")
        print(f"Scaler: {scaler_path}")
        result = evaluate_single_model(model_key, model, scaler, X_train, y_train, X_test, y_test)
        all_results.append(result)
        save_text_report(output_dir, result)
        save_confusion_matrix_csv(output_dir, result)
        print(
            f"TEST: f1_macro={result.test_metrics['f1_macro']:.4f}, "
            f"balanced_accuracy={result.test_metrics['balanced_accuracy']:.4f}, "
            f"accuracy={result.test_metrics['accuracy']:.4f}"
        )

    best = pick_best_model(all_results)
    write_comparison_tables(all_results, output_dir)
    save_best_summary(best, all_results, output_dir)
    plot_metric_comparison(all_results, output_dir)
    plot_class_f1_comparison(all_results, output_dir)
    plot_test_confusions(all_results, output_dir)

    print(f"\n{'=' * 80}")
    print("ИТОГ")
    print(f"{'=' * 80}")
    for r in sorted(
        all_results,
        key=lambda x: (x.test_metrics["f1_macro"], x.test_metrics["balanced_accuracy"], x.test_metrics["accuracy"]),
        reverse=True,
    ):
        print(
            f"{r.model_key:20s} | test_f1_macro={r.test_metrics['f1_macro']:.4f} | "
            f"test_bal_acc={r.test_metrics['balanced_accuracy']:.4f} | test_acc={r.test_metrics['accuracy']:.4f}"
        )
    print(f"\nЛучшая модель: {best.model_key} ({best.model_type})")
    print(f"Сводный файл: {output_dir / 'summary_best_model.txt'}")


if __name__ == "__main__":
    main()
