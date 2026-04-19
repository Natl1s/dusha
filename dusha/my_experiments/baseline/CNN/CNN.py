import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _exec_config(config_path: Path) -> dict:
    config_ns = {"__file__": str(config_path)}
    exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns


# Импорт base_path из experiments/configs/data.config (как в baseline-скриптах)
_data_config_path = (
    Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
)
_data_config_ns = _exec_config(_data_config_path)
DATASET_PATH = _data_config_ns["base_path"]

_train_data_config_path = Path(__file__).parent.parent.parent / "train_data.config"
_train_data_config_ns = _exec_config(_train_data_config_path)
TRAIN_DATA_PATH = Path(_train_data_config_ns["train_data_path"])

_test_data_config_path = Path(__file__).parent.parent.parent / "test_data.config"
_test_data_config_ns = _exec_config(_test_data_config_path)
TEST_DATA_PATH = Path(_test_data_config_ns["test_data_path"])

EMO2LABEL = {"angry": 0, "sad": 1, "neutral": 2, "positive": 3}
LABEL2EMO = {v: k for k, v in EMO2LABEL.items()}
TARGET_NAMES = [LABEL2EMO[i] for i in range(len(LABEL2EMO))]

MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def weighted_accuracy(y_true, y_pred, n_classes: int = 4) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for cls_idx in range(n_classes):
        cls_mask = y_true == cls_idx
        if cls_mask.sum() == 0:
            continue
        recalls.append((y_pred[cls_mask] == cls_idx).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def parse_label(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value.isdigit():
            return int(value)
        if value in EMO2LABEL:
            return EMO2LABEL[value]
    raise ValueError(f"Не удалось распарсить label/emotion: {value}")


def resolve_aggregated_dir(dataset_path: Path) -> Path:
    candidates = [
        dataset_path / "processed_dataset_090" / "aggregated_dataset",
        dataset_path / "aggregated_dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_dataset_name(manifest_path: Path) -> str:
    return manifest_path.stem


def resolve_feature_path(record: dict, data_root: Path) -> Path:
    def _resolve_from_root(root: Path, rel_path) -> Path:
        rel_path = Path(rel_path)
        if rel_path.is_absolute():
            return rel_path
        return root / rel_path

    if "tensor" in record:
        return _resolve_from_root(data_root, record["tensor"])
    if "feature_path" in record:
        return _resolve_from_root(data_root, record["feature_path"])
    if "hash_id" in record:
        return data_root / "features" / f"{record['hash_id']}.npy"
    raise ValueError(
        "В записи JSONL нет пути к фичам. Нужен один из ключей: "
        "'tensor', 'feature_path', 'hash_id'."
    )


class JsonlFeaturesDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "label" in row:
                    label = parse_label(row["label"])
                elif "emotion" in row:
                    label = parse_label(row["emotion"])
                else:
                    raise ValueError(f"В {manifest_path} отсутствует label/emotion: {row}")
                feature_path = resolve_feature_path(row, data_root=data_root)
                self.samples.append((feature_path, label))

        if not self.samples:
            raise ValueError(f"Пустой манифест: {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        x = torch.from_numpy(arr)

        # Ожидаем [1, 64, T], но поддерживаем частые вариации.
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] != 1 and x.shape[-1] == 1:
            x = x.permute(2, 0, 1)
        elif x.ndim != 3:
            raise ValueError(f"Неподдерживаемая форма тензора {x.shape} в {path}")

        return x, torch.tensor(label, dtype=torch.long)


def pad_collate_fn(batch):
    xs, ys = zip(*batch)
    max_t = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        delta = max_t - x.shape[-1]
        if delta > 0:
            x = nn.functional.pad(x, pad=(0, delta, 0, 0))
        padded.append(x)
    return torch.stack(padded), torch.stack(ys)


class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_model(
    model,
    dataset_name: str,
    model_name: str = MODEL_NAME,
    training_params=None,
    test_metrics=None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    backup_path = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pt"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), backup_path)
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
    print(f"\nМодель сохранена: {model_path.resolve()}")
    print(f"Бэкап: {backup_path.resolve()}")
    print(f"Отчёт: {report_path.resolve()}")


def evaluate_split(model, loader, criterion, device):
    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_targets = []
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * x.size(0)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    mean_loss = running_loss / len(loader.dataset)

    metrics = {
        "loss": float(mean_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "WA": float(weighted_accuracy(y_true, y_pred, n_classes=len(TARGET_NAMES))),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    return metrics, y_true, y_pred, probs, logits


def print_metrics(title, metrics, y_true, y_pred):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(TARGET_NAMES))),
            target_names=TARGET_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Запрошено обучение на GPU (--device cuda), но CUDA недоступна."
            )
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def train_cnn(
    train_manifest: Path,
    test_manifest: Path,
    data_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    save: bool,
    device_arg: str,
):
    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    print(f"Обучение запущено на устройстве: {device}")
    if use_cuda:
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(gpu_index)} (cuda:{gpu_index})")
    print(f"Train manifest: {train_manifest}")
    print(f"Test manifest:  {test_manifest}")

    train_ds = JsonlFeaturesDataset(train_manifest, data_root=data_root)
    test_ds = JsonlFeaturesDataset(test_manifest, data_root=data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=pad_collate_fn,
    )

    model = EmotionCNN(n_classes=len(TARGET_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(f"Эпох: {epochs}, batch_size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}")

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        running_preds = []
        running_targets = []
        num_batches = len(train_loader)

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen_samples += x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_preds.append(preds.detach().cpu().numpy())
            running_targets.append(y.detach().cpu().numpy())

            bar_width = 30
            filled = int(bar_width * batch_idx / num_batches)
            bar = "#" * filled + "-" * (bar_width - filled)
            mean_batch_loss = running_loss / max(seen_samples, 1)
            print(
                f"\rEpoch {epoch:03d}/{epochs} [{bar}] {batch_idx}/{num_batches} "
                f"loss={mean_batch_loss:.4f}",
                end="",
                flush=True,
            )
        print()

        train_pred = np.concatenate(running_preds, axis=0)
        train_true = np.concatenate(running_targets, axis=0)
        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        test_metrics, _, _, _, _ = evaluate_split(model, test_loader, criterion, device)
        test_f1 = test_metrics["f1_macro"]
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"test_f1={test_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_metrics, y_true, y_pred, _, _ = evaluate_split(
        model, test_loader, criterion, device
    )
    print_metrics("ФИНАЛЬНАЯ ОЦЕНКА НА TEST", final_test_metrics, y_true, y_pred)

    dataset_name = get_dataset_name(train_manifest)
    if save:
        final_report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(TARGET_NAMES))),
            target_names=TARGET_NAMES,
            digits=4,
            zero_division=0,
        )
        final_confusion_matrix = confusion_matrix(y_true, y_pred)
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": str(device),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "data_root": str(data_root),
        }
        test_metrics = {
            **final_test_metrics,
            "test_classification_report_text": final_report_text,
            "test_confusion_matrix": final_confusion_matrix.tolist(),
        }
        save_model(
            model,
            dataset_name=dataset_name,
            training_params=training_params,
            test_metrics=test_metrics,
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="CNN baseline для классификации эмоций на combine_balanced_*_small."
    )
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=TRAIN_DATA_PATH.parent,
        help="Путь к папке aggregated_dataset.",
    )
    parser.add_argument(
        "--train-manifest-name",
        type=str,
        default=TRAIN_DATA_PATH.name,
        help="Имя train manifest файла.",
    )
    parser.add_argument(
        "--test-manifest-name",
        type=str,
        default=TEST_DATA_PATH.name,
        help="Имя test manifest файла.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох.")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay для Adam."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Устройство обучения. По умолчанию --device cuda (обязателен GPU).",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель.")
    args = parser.parse_args()

    train_manifest = args.aggregated_dir / args.train_manifest_name
    test_manifest = args.aggregated_dir / args.test_manifest_name

    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest не найден: {train_manifest}")
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test manifest не найден: {test_manifest}")

    data_root_candidates = [args.aggregated_dir.parent, args.aggregated_dir.parent.parent]
    data_root = next(
        (candidate for candidate in data_root_candidates if (candidate / "features").exists()),
        args.aggregated_dir.parent,
    )

    train_cnn(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save=not args.no_save,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
