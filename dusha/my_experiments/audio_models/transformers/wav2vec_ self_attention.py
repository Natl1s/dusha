import argparse
import json
import math
import pickle
import random
from collections import deque
from contextlib import nullcontext
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
from tqdm import tqdm
from transformers import Wav2Vec2Model, get_cosine_schedule_with_warmup

from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index


def _exec_config(config_path: Path) -> dict:
    config_ns = {"__file__": str(config_path)}
    exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns


_data_config_path = (
    Path(__file__).parent.parent.parent.parent / "experiments" / "configs" / "data.config"
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
TARGET_SAMPLE_RATE = 16000

MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = "wav2vec2_xlsr300m_self_attention"
UNFREEZE_LAST_N = 4
HEAD_DROPOUT = 0.1
WARM_START_CHECKPOINT = (
    MODELS_DIR / "wav2vec2_xlsr300m_self_attention_combine_balanced_train_small_best.ckpt"
)
WARM_START_EPOCH = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def get_dataset_name(train_lmdb_path: Path) -> str:
    return train_lmdb_path.stem


def validate_lmdb_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} LMDB не найден: {path}")
    if path.is_dir():
        data_mdb = path / "data.mdb"
        if not data_mdb.exists():
            raise ValueError(
                f"{kind} LMDB указывает на директорию без data.mdb: {path}\n"
                "Ожидается путь к конкретному .lmdb файлу (или LMDB-директории с data.mdb)."
            )
    elif path.suffix.lower() != ".lmdb":
        raise ValueError(
            f"{kind} LMDB должен иметь расширение .lmdb: {path}\n"
            "Скорее всего передан путь к каталогу датасета, а не к файлу LMDB."
        )


def compute_class_weights_from_lmdb(lmdb_path: Path, n_classes: int) -> torch.Tensor:
    env = open_lmdb_readonly(lmdb_path)
    try:
        total = get_lmdb_length(env)
        class_counts = np.zeros((n_classes,), dtype=np.int64)
        with env.begin() as txn:
            for idx in range(total):
                raw = txn.get(str(idx).encode("utf-8"))
                if raw is None:
                    raise KeyError(f"В LMDB отсутствует ключ {idx}")
                payload = pickle.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError(f"Некорректный payload у ключа {idx}: ожидается dict")
                label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
                label_idx = parse_label_to_index(label_raw)
                class_counts[label_idx] += 1
    finally:
        env.close()

    if np.any(class_counts == 0):
        missing = [TARGET_NAMES[i] for i, c in enumerate(class_counts) if c == 0]
        raise ValueError(f"В train LMDB отсутствуют классы: {missing}")

    # Inverse-frequency weights normalized to mean=1 for stable CE scale.
    inv_freq = class_counts.sum() / (n_classes * class_counts.astype(np.float64))
    inv_freq = inv_freq / inv_freq.mean()
    return torch.tensor(inv_freq, dtype=torch.float32)


def _normalize_waveform(raw_waveform) -> np.ndarray:
    arr = np.asarray(raw_waveform)
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)

    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        if peak > 1.0:
            arr = arr / peak
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(arr, -1.0, 1.0)


def _crop_or_pad(
    waveform: np.ndarray,
    is_train: bool,
    min_crop_sec: float,
    max_crop_sec: float,
) -> tuple[np.ndarray, int]:
    if max_crop_sec < min_crop_sec:
        raise ValueError(f"max_crop_sec < min_crop_sec: {max_crop_sec} < {min_crop_sec}")

    target_sec = random.uniform(min_crop_sec, max_crop_sec) if is_train else max_crop_sec
    target_len = int(round(target_sec * TARGET_SAMPLE_RATE))
    if target_len <= 0:
        raise ValueError(f"Некорректная целевая длина: {target_len}")

    src_len = int(waveform.shape[0])
    if src_len >= target_len:
        if is_train:
            start = random.randint(0, src_len - target_len)
        else:
            start = max((src_len - target_len) // 2, 0)
        cropped = waveform[start : start + target_len]
        return cropped.astype(np.float32), target_len

    padded = np.zeros((target_len,), dtype=np.float32)
    padded[:src_len] = waveform
    return padded, src_len


class LmdbWaveDataset(Dataset):
    def __init__(
        self,
        lmdb_path: Path,
        is_train: bool,
        min_crop_sec: float = 5.0,
        max_crop_sec: float = 6.0,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)
        self.length = get_lmdb_length(self.env)
        if self.length <= 0:
            raise ValueError(f"Пустой LMDB: {self.lmdb_path}")
        self.is_train = is_train
        self.min_crop_sec = min_crop_sec
        self.max_crop_sec = max_crop_sec

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            raw = txn.get(str(int(idx)).encode("utf-8"))
        if raw is None:
            raise KeyError(f"В LMDB отсутствует ключ {idx}")
        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Некорректный payload у idx={idx}: ожидается dict")

        waveform_raw = payload.get("waveform", payload.get("audio", payload.get("wav")))
        if waveform_raw is None:
            raise KeyError(
                f"В payload LMDB отсутствует waveform/audio/wav (idx={idx}). "
                "Ожидается raw audio для wav2vec2."
            )

        sample_rate = int(payload.get("waveform_sr", payload.get("sample_rate", TARGET_SAMPLE_RATE)))
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Некорректная sample_rate в idx={idx}: {sample_rate}. "
                f"Ожидается {TARGET_SAMPLE_RATE} Hz."
            )

        waveform = _normalize_waveform(waveform_raw)
        waveform, valid_len = _crop_or_pad(
            waveform,
            is_train=self.is_train,
            min_crop_sec=self.min_crop_sec,
            max_crop_sec=self.max_crop_sec,
        )
        if not np.isfinite(waveform).all():
            waveform = np.zeros_like(waveform, dtype=np.float32)

        label_raw = payload.get("y", payload.get("label", payload.get("emotion")))
        label = parse_label_to_index(label_raw)
        return (
            torch.from_numpy(waveform),
            torch.tensor(valid_len, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def wave_collate_fn(batch):
    waves, valid_lens, labels = zip(*batch)
    max_len = max(wave.size(0) for wave in waves)
    padded_waves = waves[0].new_zeros((len(waves), max_len))
    for i, wave in enumerate(waves):
        cur_len = wave.size(0)
        padded_waves[i, :cur_len] = wave

    valid_lens_t = torch.stack(valid_lens).to(dtype=torch.long)
    labels_t = torch.stack(labels).to(dtype=torch.long)
    return padded_waves, valid_lens_t, labels_t


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(hidden_states).squeeze(-1)
        mask = attn_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        scores = scores - scores.max(dim=1, keepdim=True).values
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class Wav2VecSelfAttentionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-xls-r-300m",
        n_classes: int = 4,
        unfreeze_last_n: int = UNFREEZE_LAST_N,
        pooling_type: str = "attention",
    ):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.encoder.gradient_checkpointing_enable()
        self.unfreeze_last_n = unfreeze_last_n
        self.pooling_type = pooling_type

        hidden_size = self.encoder.config.hidden_size
        self.pooler = AttentionPooling(hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(hidden_size, n_classes),
        )

        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def enable_last_transformer_layers(self) -> None:
        self.freeze_encoder()
        layers = self.encoder.encoder.layers
        trainable = min(self.unfreeze_last_n, len(layers))
        for layer in layers[-trainable:]:
            for p in layer.parameters():
                p.requires_grad = True

    def _build_mask_after_feature_extractor(self, attention_mask: torch.Tensor, feature_len: int) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1)
        out_lengths = self.encoder._get_feat_extract_output_lengths(lengths).to(torch.long)
        out_lengths = torch.clamp(out_lengths, max=feature_len)
        mask = torch.zeros(
            (attention_mask.size(0), feature_len),
            dtype=torch.long,
            device=attention_mask.device,
        )
        for i, cur_len in enumerate(out_lengths):
            mask[i, : int(cur_len.item())] = 1
        return mask

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_has_trainable_params = any(p.requires_grad for p in self.encoder.parameters())
        encoder_ctx = nullcontext() if encoder_has_trainable_params else torch.no_grad()
        # Encoder is forced to fp32 for numerical stability; AMP stays effective for the head.
        with torch.autocast(device_type=input_values.device.type, enabled=False):
            with encoder_ctx:
                outputs = self.encoder(
                    input_values=input_values.float(),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        hidden_states = torch.stack(outputs.hidden_states[-4:], dim=0)
        fused = hidden_states.mean(dim=0)
        if self.pooling_type == "mean":
            pooled = fused.mean(dim=1)
        else:
            feat_mask = self._build_mask_after_feature_extractor(attention_mask, fused.size(1))
            pooled = self.pooler(fused, feat_mask)
        return self.classifier(pooled)


def save_model(
    model: nn.Module,
    dataset_name: str,
    training_params=None,
    test_metrics=None,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{MODEL_NAME}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pt"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"

    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "saved_at": timestamp,
    }
    torch.save(payload, model_path)

    report_lines = [
        f"model_name: {MODEL_NAME}",
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
    print(f"Отчёт: {report_path.resolve()}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    dataset_name: str,
    epoch: int,
    global_step: int,
    best_f1: float,
    checkpoint_kind: str,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    full_model_name = f"{MODEL_NAME}_{dataset_name}"
    checkpoint_path = MODELS_DIR / f"{full_model_name}_{checkpoint_kind}.ckpt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_f1": best_f1,
    }
    torch.save(payload, checkpoint_path)


def evaluate_split(model, loader, criterion, device, use_amp: bool):
    model.eval()
    all_logits, all_probs, all_preds, all_targets = [], [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for waves, valid_lens, y in loader:
            waves = waves.to(device, non_blocking=device.type == "cuda")
            valid_lens = valid_lens.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")

            attention_mask = (
                torch.arange(waves.size(1), device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
            ).long()
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(waves, attention_mask)
                logits = torch.clamp(logits, -20, 20)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            running_loss += loss.item() * waves.size(0)

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
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
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
            raise RuntimeError("Запрошено обучение на GPU, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def _build_loader(dataset, batch_size: int, shuffle: bool, use_cuda: bool, num_workers: int):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "collate_fn": wave_collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def train_wav2vec(
    train_lmdb: Path,
    test_lmdb: Path,
    pretrained_name: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
    warmup_ratio: float,
    min_crop_sec: float,
    max_crop_sec: float,
    eval_every_n_epochs: int,
    seed: int,
    num_workers: int,
    save: bool,
    device_arg: str,
    pooling_type: str,
    log_every_n_steps: int,
    loss_ma_window: int,
    use_amp: bool,
):
    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    use_amp = bool(use_amp and use_cuda)

    print(f"Обучение запущено на устройстве: {device}")
    if use_cuda:
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(gpu_index)} (cuda:{gpu_index})")
    print(f"Train LMDB: {train_lmdb}")
    print(f"Test LMDB:  {test_lmdb}")

    print("Class weights временно отключены для дебага NaN.")

    train_ds = LmdbWaveDataset(
        train_lmdb, is_train=True, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec
    )
    test_ds = LmdbWaveDataset(
        test_lmdb, is_train=False, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec
    )
    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True, use_cuda=use_cuda, num_workers=num_workers)
    test_loader = _build_loader(test_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda, num_workers=num_workers)

    dataset_name = get_dataset_name(train_lmdb)
    print(f"Загрузка предобученной модели: {pretrained_name}")
    model = Wav2VecSelfAttentionClassifier(
        pretrained_name=pretrained_name,
        n_classes=len(TARGET_NAMES),
        unfreeze_last_n=UNFREEZE_LAST_N,
        pooling_type=pooling_type,
    ).to(device)

    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
            {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
        ]
    )

    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_updates * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if not WARM_START_CHECKPOINT.exists():
        raise FileNotFoundError(f"Warm-start checkpoint не найден: {WARM_START_CHECKPOINT}")
    checkpoint = torch.load(WARM_START_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"Warm-start checkpoint загружен: {WARM_START_CHECKPOINT}")

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(
        f"Эпох: {epochs}, batch_size: {batch_size}, grad_accum: {gradient_accumulation_steps}, "
        f"effective_batch: {batch_size * gradient_accumulation_steps}"
    )
    print(
        f"lr_encoder: {lr_encoder}, lr_head: {lr_head}, weight_decay: {weight_decay}, "
        f"warmup_ratio: {warmup_ratio}, fp16: {use_amp}, pooling: {pooling_type}, head_dropout: {HEAD_DROPOUT}"
    )
    print("class_weights (CE): disabled (debug mode)")
    print(f"step logging: every {log_every_n_steps} updates, moving_avg_window: {loss_ma_window}")
    print("step | loss | acc | grad_norm | lr")
    print(
        "Стратегия: Epoch 1-2 train head only, "
        f"Epoch 3+ train head + last {model.unfreeze_last_n} transformer layers"
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_f1 = float(checkpoint.get("best_f1", -1.0))
    global_step = int(checkpoint.get("global_step", 0))
    start_epoch = WARM_START_EPOCH
    if epochs < start_epoch:
        raise ValueError(
            f"--epochs должен быть >= {start_epoch}, т.к. обучение стартует с {start_epoch}-й эпохи."
        )
    print(
        f"Старт обучения с эпохи {start_epoch} "
        f"(пропуск эпох 1-{start_epoch - 1} за счёт warm-start)."
    )

    for epoch in range(start_epoch, epochs + 1):
        if epoch == 3:
            if use_cuda:
                torch.cuda.empty_cache()
            model.enable_last_transformer_layers()
            print(
                f"\n[Stage switch] Разморозка последних {model.unfreeze_last_n} "
                "слоёв wav2vec2 encoder."
            )
        backward_chunk_size = 1 if epoch >= 3 else batch_size

        model.train()
        running_loss = 0.0
        seen_samples = 0
        running_preds = []
        running_targets = []
        running_correct = 0
        loss_ma_queue = deque()
        loss_ma_sum = 0.0

        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch:02d}/{epochs}",
            leave=False,
        )

        for batch_idx, (waves, valid_lens, y) in progress:
            waves = waves.to(device, non_blocking=use_cuda)
            valid_lens = valid_lens.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            attention_mask = (torch.arange(waves.size(1), device=device).unsqueeze(0) < valid_lens.unsqueeze(1)).long()

            batch_size_cur = waves.size(0)
            chunk_size = min(backward_chunk_size, batch_size_cur)
            batch_failed = False
            batch_loss_mean = 0.0
            batch_preds = []
            batch_targets = []

            for start in range(0, batch_size_cur, chunk_size):
                end = min(start + chunk_size, batch_size_cur)
                chunk_waves = waves[start:end]
                chunk_valid_lens = valid_lens[start:end]
                chunk_y = y[start:end]
                chunk_mask = attention_mask[start:end]
                chunk_weight = (end - start) / batch_size_cur

                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        chunk_logits = model(chunk_waves, chunk_mask)
                        chunk_logits = torch.clamp(chunk_logits, -20, 20)
                        chunk_raw_loss = criterion(chunk_logits, chunk_y)
                        chunk_loss = (chunk_raw_loss * chunk_weight) / gradient_accumulation_steps

                    if not torch.isfinite(chunk_loss):
                        print("NaN detected, skipping batch")
                        batch_failed = True
                        break

                    scaler.scale(chunk_loss).backward()
                except torch.OutOfMemoryError:
                    print("OOM on forward/backward, skipping batch")
                    batch_failed = True
                    break

                batch_loss_mean += chunk_raw_loss.item() * chunk_weight
                chunk_preds = torch.argmax(chunk_logits, dim=1)
                running_correct += (chunk_preds == chunk_y).sum().item()
                batch_preds.append(chunk_preds.detach().cpu().numpy())
                batch_targets.append(chunk_y.detach().cpu().numpy())

            if batch_failed:
                optimizer.zero_grad(set_to_none=True)
                if use_cuda:
                    torch.cuda.empty_cache()
                continue

            running_loss += batch_loss_mean * batch_size_cur
            seen_samples += batch_size_cur
            running_preds.append(np.concatenate(batch_preds, axis=0))
            running_targets.append(np.concatenate(batch_targets, axis=0))
            loss_ma_queue.append(batch_loss_mean)
            loss_ma_sum += batch_loss_mean
            if len(loss_ma_queue) > loss_ma_window:
                loss_ma_sum -= loss_ma_queue.popleft()

            should_step = (batch_idx % gradient_accumulation_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                optimizer_stepped = False
                grad_norm_value = 0.0
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # With AMP the optimizer step can be skipped on overflow; in that case
                # we should not advance LR schedule or global update step.
                optimizer_stepped = (not use_amp) or (scaler.get_scale() >= prev_scale)

                optimizer.zero_grad(set_to_none=True)
                if optimizer_stepped:
                    scheduler.step()
                    global_step += 1

                    loss_ma = loss_ma_sum / len(loss_ma_queue)
                    running_acc = running_correct / max(seen_samples, 1)

                    if global_step % log_every_n_steps == 0:
                        print(
                            f"{global_step} | {loss_ma:.4f} | {running_acc:.4f} | "
                            f"{grad_norm_value:.4f} | {optimizer.param_groups[0]['lr']:.2e}"
                        )

            progress.set_postfix(loss=f"{running_loss / max(seen_samples, 1):.4f}", step=global_step)

        if not running_preds:
            raise RuntimeError(
                "За эпоху не обработано ни одного batch (все пропущены из-за OOM/NaN). "
                "Уменьшите batch_size/crop или число размораживаемых слоёв."
            )

        train_pred = np.concatenate(running_preds, axis=0)
        train_true = np.concatenate(running_targets, axis=0)
        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        do_eval = (epoch % eval_every_n_epochs == 0) or (epoch == epochs)
        if do_eval:
            test_metrics, _, _, _, _ = evaluate_split(model, test_loader, criterion, device, use_amp=use_amp)
            test_f1 = test_metrics["f1_macro"]
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        dataset_name=dataset_name,
                        epoch=epoch,
                        global_step=global_step,
                        best_f1=best_f1,
                        checkpoint_kind="best",
                    )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
                f"test_f1={test_f1:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                f"test=skip (eval_every_n_epochs={eval_every_n_epochs})"
            )

        if save:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                dataset_name=dataset_name,
                epoch=epoch,
                global_step=global_step,
                best_f1=best_f1,
                checkpoint_kind="last",
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test_metrics, y_true, y_pred, _, _ = evaluate_split(
        model, test_loader, criterion, device, use_amp=use_amp
    )
    print_metrics("ФИНАЛЬНАЯ ОЦЕНКА НА TEST", final_test_metrics, y_true, y_pred)

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
            "pretrained_name": pretrained_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "lr_encoder": lr_encoder,
            "lr_head": lr_head,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "min_crop_sec": min_crop_sec,
            "max_crop_sec": max_crop_sec,
            "unfreeze_last_n": model.unfreeze_last_n,
            "eval_every_n_epochs": eval_every_n_epochs,
            "gradient_checkpointing": True,
            "fp16": use_amp,
            "seed": seed,
            "num_workers": num_workers,
            "device": str(device),
            "pooling_type": pooling_type,
            "head_dropout": HEAD_DROPOUT,
            "class_weights": "disabled_for_nan_debug",
            "log_every_n_steps": log_every_n_steps,
            "loss_ma_window": loss_ma_window,
            "train_lmdb": str(train_lmdb),
            "test_lmdb": str(test_lmdb),
        }
        test_metrics = {
            **final_test_metrics,
            "test_classification_report_text": final_report_text,
            "test_confusion_matrix": final_confusion_matrix.tolist(),
        }
        save_model(
            model=model,
            dataset_name=dataset_name,
            training_params=training_params,
            test_metrics=test_metrics,
        )

    return model


def run_one_batch_overfit_test(
    train_lmdb: Path,
    pretrained_name: str,
    batch_size: int,
    min_crop_sec: float,
    max_crop_sec: float,
    seed: int,
    num_workers: int,
    device_arg: str,
    pooling_type: str,
    steps: int,
    lr_encoder: float,
    lr_head: float,
):
    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    print(f"One-batch overfit test на устройстве: {device}")
    print(f"pooling: {pooling_type}, steps: {steps}, lr_encoder: {lr_encoder}, lr_head: {lr_head}")

    train_ds = LmdbWaveDataset(
        train_lmdb, is_train=True, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec
    )
    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True, use_cuda=use_cuda, num_workers=num_workers)
    waves, valid_lens, labels = next(iter(train_loader))
    waves = waves.to(device, non_blocking=use_cuda)
    valid_lens = valid_lens.to(device, non_blocking=use_cuda)
    labels = labels.to(device, non_blocking=use_cuda)
    attention_mask = (torch.arange(waves.size(1), device=device).unsqueeze(0) < valid_lens.unsqueeze(1)).long()

    model = Wav2VecSelfAttentionClassifier(
        pretrained_name=pretrained_name,
        n_classes=len(TARGET_NAMES),
        pooling_type=pooling_type,
    ).to(device)

    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": encoder_params, "lr": lr_encoder},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    # Overfit sanity check: disable dropout randomness, but keep grads and optimizer steps.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    for i in range(steps):
        logits = model(waves, attention_mask)
        logits = torch.clamp(logits, -20, 20)
        if torch.isnan(logits).any():
            print(f"NaN logits на шаге {i}")
            break
        loss = criterion(logits, labels)
        print(i, loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(
        description="Wav2Vec2-XLS-R-300M + self-attention pooling для классификации эмоций из LMDB."
    )
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=TRAIN_DATA_PATH.parent,
        help="Путь к папке с LMDB файлами.",
    )
    parser.add_argument(
        "--train-lmdb-name",
        type=str,
        default=TRAIN_DATA_PATH.name,
        help="Имя train LMDB файла.",
    )
    parser.add_argument(
        "--test-lmdb-name",
        type=str,
        default=TEST_DATA_PATH.name,
        help="Имя test LMDB файла.",
    )
    parser.add_argument(
        "--pretrained-name",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="HF имя предобученной wav2vec2 модели.",
    )
    parser.add_argument("--epochs", type=int, default=18, help="Количество эпох.")
    parser.add_argument("--batch-size", type=int, default=4, help="Размер батча.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Шаги накопления градиента.",
    )
    parser.add_argument("--lr-encoder", type=float, default=5e-6, help="LR для encoder.")
    parser.add_argument("--lr-head", type=float, default=1e-5, help="LR для classifier head.")
    parser.add_argument(
        "--pooling-type",
        type=str,
        choices=["attention", "mean"],
        default="mean",
        help="Тип pooling: attention (основной) или mean (быстрый sanity test).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Доля warmup шагов для cosine scheduler.",
    )
    parser.add_argument("--min-crop-sec", type=float, default=4.0, help="Минимальная длина аудио (сек).")
    parser.add_argument("--max-crop-sec", type=float, default=6.0, help="Максимальная длина аудио (сек).")
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=2,
        help="Как часто запускать eval по test (в эпохах).",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Количество workers DataLoader.")
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=20,
        help="Логировать step-метрики каждые N optimizer updates.",
    )
    parser.add_argument(
        "--loss-ma-window",
        type=int,
        default=50,
        help="Окно moving average для loss в step-логах.",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Включить mixed precision (fp16) на CUDA для снижения потребления VRAM.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Отключить mixed precision (fp16).",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--one-batch-overfit-test",
        action="store_true",
        help="Запустить overfit sanity test на одном батче без обычного обучения.",
    )
    parser.add_argument(
        "--one-batch-steps",
        type=int,
        default=50,
        help="Количество шагов в one-batch overfit test.",
    )
    parser.add_argument(
        "--one-batch-lr-encoder",
        type=float,
        default=1e-6,
        help="LR encoder для one-batch overfit test.",
    )
    parser.add_argument(
        "--one-batch-lr-head",
        type=float,
        default=1e-5,
        help="LR head для one-batch overfit test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="cuda",
        help="Устройство обучения. По умолчанию --device cuda (обязателен GPU).",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель.")
    args = parser.parse_args()

    train_lmdb = args.aggregated_dir / args.train_lmdb_name
    test_lmdb = args.aggregated_dir / args.test_lmdb_name

    validate_lmdb_path(train_lmdb, kind="Train")
    validate_lmdb_path(test_lmdb, kind="Test")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps должен быть > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs должен быть > 0")
    if args.eval_every_n_epochs <= 0:
        raise ValueError("--eval-every-n-epochs должен быть > 0")
    if args.min_crop_sec <= 0 or args.max_crop_sec <= 0:
        raise ValueError("--min-crop-sec и --max-crop-sec должны быть > 0")
    if args.max_crop_sec < args.min_crop_sec:
        raise ValueError("--max-crop-sec должен быть >= --min-crop-sec")
    if args.log_every_n_steps <= 0:
        raise ValueError("--log-every-n-steps должен быть > 0")
    if args.loss_ma_window <= 0:
        raise ValueError("--loss-ma-window должен быть > 0")
    if args.one_batch_steps <= 0:
        raise ValueError("--one-batch-steps должен быть > 0")

    if args.one_batch_overfit_test:
        run_one_batch_overfit_test(
            train_lmdb=train_lmdb,
            pretrained_name=args.pretrained_name,
            batch_size=args.batch_size,
            min_crop_sec=args.min_crop_sec,
            max_crop_sec=args.max_crop_sec,
            seed=args.seed,
            num_workers=args.num_workers,
            device_arg=args.device,
            pooling_type=args.pooling_type,
            steps=args.one_batch_steps,
            lr_encoder=args.one_batch_lr_encoder,
            lr_head=args.one_batch_lr_head,
        )
        return

    train_wav2vec(
        train_lmdb=train_lmdb,
        test_lmdb=test_lmdb,
        pretrained_name=args.pretrained_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_crop_sec=args.min_crop_sec,
        max_crop_sec=args.max_crop_sec,
        eval_every_n_epochs=args.eval_every_n_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        save=not args.no_save,
        device_arg=args.device,
        pooling_type=args.pooling_type,
        log_every_n_steps=args.log_every_n_steps,
        loss_ma_window=args.loss_ma_window,
        use_amp=args.fp16,
    )


if __name__ == "__main__":
    main()
