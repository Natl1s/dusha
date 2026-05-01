"""
Дообучение DeepPavlov/rubert-base-cased для классификации эмоций.

Пайплайн:
- загрузка train/test из LMDB через train_data.config/test_data.config
- токенизация через AutoTokenizer
- модель: BERT -> [CLS] -> Linear -> GELU -> Dropout -> Linear
- двухэтапное обучение:
  1) 1 эпоха: заморожен весь BERT, обучается только classifier
  2) оставшиеся эпохи: разморожены encoder.layer.6-11
- вывод метрик и confusion matrix
- сохранение модели и отчёта
"""

import argparse
import builtins
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from my_experiments.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as exc:
    raise ImportError(
        "Требуется библиотека transformers. Установите: pip install transformers"
    ) from exc


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


_data_config_path = Path(__file__).parent.parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = _exec_config(_data_config_path)
DATASET_PATH = _data_config_ns["base_path"]

_train_data_config_path = Path(__file__).parent.parent.parent / "train_data.config"
_train_data_config_ns = _exec_config(_train_data_config_path)
TRAIN_DATA_PATH = Path(_train_data_config_ns["train_data_path"])

_test_data_config_path = Path(__file__).parent.parent.parent / "test_data.config"
_test_data_config_ns = _exec_config(_test_data_config_path)
TEST_DATA_PATH = Path(_test_data_config_ns["test_data_path"])

MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem
DEFAULT_BACKBONE_NAME = "DeepPavlov/rubert-base-cased"

TARGET_NAMES = ["angry", "sad", "neutral", "positive"]
EMO2IDX = {name: i for i, name in enumerate(TARGET_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошено обучение на GPU, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def get_dataset_name(train_manifest_path: Path) -> str:
    return Path(train_manifest_path).stem


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_texts_from_manifest(manifest_path: Path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


class TransformerEmotionDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = []
        for label in labels:
            if label not in EMO2IDX:
                raise ValueError(f"Неизвестная метка эмоции: {label}")
            self.labels.append(EMO2IDX[label])
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
        )


class EmotionClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float = 0.1,
        classifier_hidden_size: int | None = None,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = int(self.bert.config.hidden_size)
        if classifier_hidden_size is None:
            classifier_hidden_size = bert_hidden_size
        self.classifier_hidden_size = int(classifier_hidden_size)
        self.head = nn.Sequential(
            nn.Linear(bert_hidden_size, self.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.classifier_hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.head(cls)
        return logits


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma должен быть >= 0, получено: {gamma}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing должен быть в интервале [0, 1), получено: {label_smoothing}"
            )
        if weight is not None and weight.ndim != 1:
            raise ValueError("weight для FocalLoss должен быть 1D тензором")
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0.0:
            target_probs = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            target_probs.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            target_probs = torch.zeros_like(log_probs)
            target_probs.scatter_(1, targets.unsqueeze(1), 1.0)

        ce_per_class = -(target_probs * log_probs)
        if self.weight is not None:
            ce_per_class = ce_per_class * self.weight.unsqueeze(0)
        ce_loss = ce_per_class.sum(dim=1)
        pt = torch.exp((log_probs * target_probs).sum(dim=1))
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * ce_loss).mean()


def build_class_weights(labels_raw: np.ndarray) -> np.ndarray:
    label_indices = np.array([EMO2IDX[label] for label in labels_raw], dtype=np.int64)
    counts = np.bincount(label_indices, minlength=len(TARGET_NAMES)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            "В train-части после split отсутствуют некоторые классы. "
            "Невозможно корректно рассчитать class weights."
        )
    n_samples = counts.sum()
    n_classes = len(TARGET_NAMES)
    weights = n_samples / (n_classes * counts)
    return weights.astype(np.float32)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def evaluate_split(model, loader, criterion, device: torch.device, desc: str):
    model.eval()
    running_loss = 0.0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            running_loss += loss.item() * input_ids.size(0)
            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = running_loss / len(loader.dataset)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TARGET_NAMES))),
        target_names=TARGET_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES))))
    return metrics, report_text, cm, y_true, y_pred


def save_model(
    model: nn.Module,
    tokenizer,
    dataset_name: str,
    checkpoint_payload: dict,
    training_params: dict,
    test_metrics: dict,
    model_name: str = MODEL_NAME,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    model_path_timestamped = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pt"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    torch.save(checkpoint_payload, model_path)
    torch.save(checkpoint_payload, model_path_timestamped)
    tokenizer.save_pretrained(MODELS_DIR / f"{full_model_name}_tokenizer")

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
    print(f"✓ Токенайзер: {(MODELS_DIR / f'{full_model_name}_tokenizer').absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
    print(f"{'=' * 60}")


def load_model(dataset_name: str, model_name: str = MODEL_NAME, map_location: str | torch.device = "cpu"):
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена! Проверьте наличие файла:\n  {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=map_location)
    model_params = checkpoint["model_params"]
    model = EmotionClassifier(
        model_name=model_params["backbone_name"],
        num_classes=model_params["n_classes"],
        dropout=model_params["dropout"],
        classifier_hidden_size=model_params.get("classifier_hidden_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer_dir = MODELS_DIR / f"{full_model_name}_tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_params["backbone_name"])
    print(f"✓ Модель загружена из {model_path}")
    return model, tokenizer, checkpoint


def model_exists(dataset_name: str, model_name: str = MODEL_NAME) -> bool:
    full_model_name = f"{model_name}_{dataset_name}"
    return (MODELS_DIR / f"{full_model_name}_model.pt").exists()


def _build_loader(dataset, batch_size: int, shuffle: bool, use_cuda: bool):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


def _set_stage_trainability(model: EmotionClassifier, stage: int) -> None:
    for param in model.bert.parameters():
        param.requires_grad = False

    if stage == 2:
        for name, param in model.bert.named_parameters():
            if any(f"encoder.layer.{layer_idx}" in name for layer_idx in range(6, 12)):
                param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True


def _build_optimizer_and_scheduler(
    model: EmotionClassifier,
    lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float,
):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_rubert(
    backbone_name: str = DEFAULT_BACKBONE_NAME,
    save: bool = True,
    epochs: int = 5,
    stage1_epochs: int = 1,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    max_len: int = 128,
    dropout: float = 0.1,
    classifier_hidden_size: int | None = None,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    loss_name: str = "ce",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    use_class_weights: bool = True,
    val_size: float = 0.1,
    fp16: bool = False,
    seed: int = 42,
    device_arg: str = "auto",
):
    if epochs < 1:
        raise ValueError(f"epochs должен быть >= 1, получено: {epochs}")
    if stage1_epochs < 0:
        raise ValueError(f"stage1_epochs должен быть >= 0, получено: {stage1_epochs}")
    if stage1_epochs > epochs:
        raise ValueError(
            f"stage1_epochs ({stage1_epochs}) не может быть больше epochs ({epochs})"
        )
    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps должен быть >= 1, получено: {grad_accum_steps}")
    if classifier_hidden_size is not None and classifier_hidden_size < 1:
        raise ValueError(
            f"classifier_hidden_size должен быть >= 1 или None, получено: {classifier_hidden_size}"
        )
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size должен быть в интервале (0, 1), получено: {val_size}")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"warmup_ratio должен быть в интервале [0, 1), получено: {warmup_ratio}")
    if loss_name not in {"ce", "focal"}:
        raise ValueError(f"loss_name должен быть 'ce' или 'focal', получено: {loss_name}")
    if focal_gamma < 0:
        raise ValueError(f"focal_gamma должен быть >= 0, получено: {focal_gamma}")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing должен быть в интервале [0, 1), получено: {label_smoothing}")

    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    amp_device_type = "cuda" if use_cuda else "cpu"
    use_fp16 = bool(fp16 and use_cuda)
    if fp16 and not use_cuda:
        print("⚠ fp16 запрошен, но CUDA недоступна. fp16 отключён.")
    print(f"Обучение запущено на устройстве: {device}")

    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)
    print(f"\n📊 Датасет: {dataset_name}\n")

    print(f"{'=' * 60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'=' * 60}")
    train_texts, y_train_raw = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train_raw)}")
    print(f"Распределение классов в train: {np.unique(y_train_raw, return_counts=True)}")

    train_texts, val_texts, y_train_raw, y_val_raw = train_test_split(
        train_texts,
        y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw,
    )
    print(f"Train после split: {len(y_train_raw)}")
    print(f"Val после split:   {len(y_val_raw)}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'=' * 60}")
    test_texts, y_test_raw = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test_raw)}")
    print(f"Распределение классов в test: {np.unique(y_test_raw, return_counts=True)}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ТОКЕНАЙЗЕРА И МОДЕЛИ")
    print(f"{'=' * 60}")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = EmotionClassifier(
        model_name=backbone_name,
        num_classes=len(TARGET_NAMES),
        dropout=dropout,
        classifier_hidden_size=classifier_hidden_size,
    ).to(device)

    train_ds = TransformerEmotionDataset(train_texts, y_train_raw, tokenizer, max_len=max_len)
    val_ds = TransformerEmotionDataset(val_texts, y_val_raw, tokenizer, max_len=max_len)
    test_ds = TransformerEmotionDataset(test_texts, y_test_raw, tokenizer, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(val_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
    test_loader = _build_loader(test_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    class_weights_np = build_class_weights(y_train_raw)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor if use_class_weights else None,
            label_smoothing=label_smoothing,
        )
    else:
        criterion = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights_tensor if use_class_weights else None,
            label_smoothing=label_smoothing,
        )
    scaler = torch.amp.GradScaler(amp_device_type, enabled=use_fp16)

    stage2_epochs = max(0, epochs - stage1_epochs)
    steps_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
    stage1_total_steps = steps_per_epoch * stage1_epochs
    stage2_total_steps = steps_per_epoch * stage2_epochs

    _set_stage_trainability(model, stage=1)
    optimizer_stage1, scheduler_stage1 = _build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        total_steps=max(1, stage1_total_steps),
        warmup_ratio=warmup_ratio,
    )
    optimizer_stage2 = None
    scheduler_stage2 = None
    if stage2_epochs > 0:
        _set_stage_trainability(model, stage=2)
        optimizer_stage2, scheduler_stage2 = _build_optimizer_and_scheduler(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            total_steps=max(1, stage2_total_steps),
            warmup_ratio=warmup_ratio,
        )
        _set_stage_trainability(model, stage=1)

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер val:   {len(val_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(
        f"Эпох: {epochs}, stage1_epochs: {stage1_epochs}, batch_size: {batch_size}, "
        f"grad_accum_steps: {grad_accum_steps}, max_len: {max_len}"
    )
    print(
        f"backbone: {backbone_name}, dropout: {dropout}, lr: {lr}, "
        f"weight_decay: {weight_decay}, warmup_ratio: {warmup_ratio}, fp16: {use_fp16}"
    )
    print(
        f"loss: {loss_name}, focal_gamma: {focal_gamma}, label_smoothing: {label_smoothing}, "
        f"class_weights: {'on' if use_class_weights else 'off'}"
    )
    print(f"class_weights_values: {dict(zip(TARGET_NAMES, class_weights_np.tolist()))}")
    print("epoch | stage | train_loss | train_f1 | val_loss | val_f1")

    best_state = None
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch <= stage1_epochs:
            stage = 1
            _set_stage_trainability(model, stage=1)
            optimizer = optimizer_stage1
            scheduler = scheduler_stage1
        else:
            stage = 2
            _set_stage_trainability(model, stage=2)
            if optimizer_stage2 is None or scheduler_stage2 is None:
                raise RuntimeError("Stage 2 optimizer/scheduler не инициализирован.")
            optimizer = optimizer_stage2
            scheduler = scheduler_stage2

        model.train()
        running_loss = 0.0
        seen_samples = 0
        all_train_preds = []
        all_train_targets = []

        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Train {epoch:02d}/{epochs} (stage {stage})", leave=False)
        for step, (input_ids, attention_mask, labels) in enumerate(progress, start=1):
            input_ids = input_ids.to(device, non_blocking=use_cuda)
            attention_mask = attention_mask.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            with torch.amp.autocast(amp_device_type, enabled=use_fp16):
                logits = model(input_ids, attention_mask)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / grad_accum_steps

            scaler.scale(loss).backward()

            should_step = (step % grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += raw_loss.item() * input_ids.size(0)
            seen_samples += input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_train_preds.append(preds.detach().cpu().numpy())
            all_train_targets.append(labels.detach().cpu().numpy())
            progress.set_postfix(loss=f"{running_loss / max(1, seen_samples):.4f}")

        train_preds = np.concatenate(all_train_preds, axis=0)
        train_true = np.concatenate(all_train_targets, axis=0)
        train_loss = running_loss / len(train_ds)
        train_f1 = f1_score(train_true, train_preds, average="macro", zero_division=0)

        val_metrics, _, _, _, _ = evaluate_split(model, val_loader, criterion, device, desc="Eval Val")
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": train_loss,
                "train_f1_macro": float(train_f1),
                "val_loss": val_metrics["loss"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"{epoch:02d} | {stage} | {train_loss:.4f} | {train_f1:.4f} | "
            f"{val_metrics['loss']:.4f} | {val_metrics['f1_macro']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TRAIN")
    print(f"{'=' * 60}")
    train_metrics, train_report_text, train_cm, _, _ = evaluate_split(
        model, train_loader, criterion, device, desc="Final Train Eval"
    )
    print(train_report_text)
    print("Матрица ошибок (train):")
    print(train_cm)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА VAL")
    print(f"{'=' * 60}")
    val_metrics, val_report_text, val_cm, _, _ = evaluate_split(
        model, val_loader, criterion, device, desc="Final Val Eval"
    )
    print(val_report_text)
    print("Матрица ошибок (val):")
    print(val_cm)

    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST")
    print(f"{'=' * 60}")
    test_metrics, test_report_text, test_cm, _, _ = evaluate_split(
        model, test_loader, criterion, device, desc="Final Test Eval"
    )
    print(test_report_text)
    print("Матрица ошибок (test):")
    print(test_cm)

    if save:
        model_params = {
            "backbone_name": backbone_name,
            "dropout": dropout,
            "classifier_hidden_size": model.classifier_hidden_size,
            "n_classes": len(TARGET_NAMES),
            "max_len": max_len,
        }
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "target_names": TARGET_NAMES,
            "model_params": model_params,
        }
        training_params = {
            "epochs": epochs,
            "stage1_epochs": stage1_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "max_len": max_len,
            "dropout": dropout,
            "classifier_hidden_size": model.classifier_hidden_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "loss_name": loss_name,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
            "use_class_weights": use_class_weights,
            "class_weights": class_weights_np.tolist() if use_class_weights else None,
            "val_size": val_size,
            "fp16": use_fp16,
            "seed": seed,
            "device": str(device),
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "history": history,
            "best_val_f1_macro": float(best_val_f1),
        }
        export_metrics = {
            **test_metrics,
            "test_classification_report_text": test_report_text,
            "test_confusion_matrix": test_cm.tolist(),
            "train_metrics": train_metrics,
            "train_classification_report_text": train_report_text,
            "train_confusion_matrix": train_cm.tolist(),
            "val_metrics": val_metrics,
            "val_classification_report_text": val_report_text,
            "val_confusion_matrix": val_cm.tolist(),
        }
        save_model(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            checkpoint_payload=checkpoint_payload,
            training_params=training_params,
            test_metrics=export_metrics,
        )

    return model, dataset_name


def load_and_evaluate(device_arg: str = "auto"):
    device = resolve_device(device_arg)
    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    dataset_name = get_dataset_name(train_manifest)

    model, tokenizer, checkpoint = load_model(dataset_name, map_location=device)
    model = model.to(device)
    model.eval()

    model_params = checkpoint["model_params"]
    max_len = int(model_params["max_len"])

    train_texts, y_train_raw = load_texts_from_manifest(train_manifest)
    test_texts, y_test_raw = load_texts_from_manifest(test_manifest)

    train_ds = TransformerEmotionDataset(train_texts, y_train_raw, tokenizer, max_len=max_len)
    test_ds = TransformerEmotionDataset(test_texts, y_test_raw, tokenizer, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=32, shuffle=False, use_cuda=device.type == "cuda")
    test_loader = _build_loader(test_ds, batch_size=32, shuffle=False, use_cuda=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    train_metrics, train_report_text, train_cm, _, _ = evaluate_split(
        model, train_loader, criterion, device, desc="Final Train Eval"
    )
    test_metrics, test_report_text, test_cm, _, _ = evaluate_split(
        model, test_loader, criterion, device, desc="Final Test Eval"
    )

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    print(train_report_text)
    print("Матрица ошибок (train):")
    print(train_cm)
    print(train_metrics)

    print(f"\n{'=' * 60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'=' * 60}")
    print(test_report_text)
    print("Матрица ошибок (test):")
    print(test_cm)
    print(test_metrics)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение или загрузка модели RuBERT для классификации эмоций"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto"],
        default="auto",
        help="train - обучить новую модель, load - загрузить существующую, auto - загрузить если есть",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель после обучения")
    parser.add_argument("--backbone-name", type=str, default=DEFAULT_BACKBONE_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout для головы (рекомендуется тюнить: 0.1, 0.2, 0.3).",
    )
    parser.add_argument(
        "--classifier-hidden-size",
        type=int,
        default=None,
        help="Размер скрытого слоя головы (None = hidden_size BERT).",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--loss-name", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Отключить class weights в функции потерь.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--fp16", action="store_true", help="Использовать mixed precision (CUDA only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Устройство обучения.",
    )
    args = parser.parse_args()

    dataset_name = get_dataset_name(TRAIN_DATA_PATH)
    if args.mode == "train":
        train_rubert(
            backbone_name=args.backbone_name,
            save=not args.no_save,
            epochs=args.epochs,
            stage1_epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            max_len=args.max_len,
            dropout=args.dropout,
            classifier_hidden_size=args.classifier_hidden_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            loss_name=args.loss_name,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            use_class_weights=not args.no_class_weights,
            val_size=args.val_size,
            fp16=args.fp16,
            seed=args.seed,
            device_arg=args.device,
        )
    elif args.mode == "load":
        load_and_evaluate(device_arg=args.device)
    else:
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            load_and_evaluate(device_arg=args.device)
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            train_rubert(
                backbone_name=args.backbone_name,
                save=not args.no_save,
                epochs=args.epochs,
                stage1_epochs=args.stage1_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                max_len=args.max_len,
                dropout=args.dropout,
                classifier_hidden_size=args.classifier_hidden_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                loss_name=args.loss_name,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                use_class_weights=not args.no_class_weights,
                val_size=args.val_size,
                fp16=args.fp16,
                seed=args.seed,
                device_arg=args.device,
            )
