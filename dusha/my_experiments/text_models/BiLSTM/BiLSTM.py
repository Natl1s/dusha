"""
FastText Embeddings + BiLSTM для классификации эмоций по тексту.

Скрипт повторяет базовый пайплайн Embeddings_LogReg:
- загрузка train/test из LMDB через конфиги train_data.config/test_data.config
- предобработка текста
- работа с FastText embeddings
- прогрессбары обучения
- вывод метрик и confusion matrix
- сохранение модели и отчёта
"""

import argparse
import builtins
import json
import random
import re
from collections import Counter
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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from my_experiments.lmdb_utils import load_texts_from_lmdb as _load_texts_from_lmdb


def print(*args, **kwargs):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if args:
        builtins.print(prefix, *args, **kwargs)
    else:
        builtins.print(prefix, **kwargs)


try:
    from gensim.models.fasttext import load_facebook_model

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠ ВНИМАНИЕ: gensim не установлен!")
    print("Установите: pip install gensim")
    print("Или: poetry add gensim")


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
DEFAULT_EMBEDDINGS_PATH = Path.home() / "fasttext_models" / "cc.ru.300.bin"

TARGET_NAMES = ["angry", "sad", "neutral", "positive"]
EMO2IDX = {name: i for i, name in enumerate(TARGET_NAMES)}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


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
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_texts_from_manifest(manifest_path: Path):
    return _load_texts_from_lmdb(Path(manifest_path), preprocess_fn=preprocess_text)


def load_fasttext_model(embeddings_path: Path):
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Библиотека gensim не установлена!\n"
            "Установите: pip install gensim\n"
            "Или: poetry add gensim"
        )

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Файл с embeddings не найден: {embeddings_path}")

    print(f"Загрузка FastText embeddings из {embeddings_path}...")
    print("⏳ Это может занять несколько минут...")
    model = load_facebook_model(str(embeddings_path))
    print("✓ Embeddings загружены!")
    print(f"  - Размерность вектора: {model.wv.vector_size}")
    print(f"  - Количество слов в словаре: {len(model.wv)}")
    return model


def build_vocab(texts: list[str], max_vocab_size: int, min_freq: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in word2idx:
            continue
        word2idx[token] = len(word2idx)
        if len(word2idx) >= max_vocab_size:
            break
    return word2idx


def build_embedding_matrix(word2idx: dict[str, int], fasttext_model) -> np.ndarray:
    dim = int(fasttext_model.wv.vector_size)
    matrix = np.zeros((len(word2idx), dim), dtype=np.float32)
    for token, idx in tqdm(word2idx.items(), desc="Инициализация embedding matrix"):
        if idx == PAD_IDX:
            continue
        matrix[idx] = fasttext_model.wv[token]
    return matrix


def encode_text(text: str, word2idx: dict[str, int], max_len: int) -> tuple[np.ndarray, int]:
    tokens = text.split()
    ids = [word2idx.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    length = max(1, len(ids))
    if len(ids) < max_len:
        ids.extend([PAD_IDX] * (max_len - len(ids)))
    return np.asarray(ids, dtype=np.int64), length


class TextSequenceDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, word2idx: dict[str, int], max_len: int):
        self.input_ids = []
        self.lengths = []
        self.labels = []

        for text, label in zip(texts, labels):
            ids, length = encode_text(text, word2idx, max_len=max_len)
            self.input_ids.append(ids)
            self.lengths.append(length)
            if label not in EMO2IDX:
                raise ValueError(f"Неизвестная метка эмоции: {label}")
            self.labels.append(EMO2IDX[label])

        self.input_ids = np.stack(self.input_ids)
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BiLSTMEmotionClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        freeze_embeddings: bool,
        n_classes: int,
        pooling_mode: str = "mean_max",
    ):
        super().__init__()
        if pooling_mode not in {"mean_max", "last_hidden"}:
            raise ValueError(f"Неподдерживаемый pooling_mode: {pooling_mode}")
        self.pooling_mode = pooling_mode
        emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            emb_tensor, freeze=freeze_embeddings, padding_idx=PAD_IDX
        )
        emb_dim = emb_tensor.shape[1]
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_size * 4 if pooling_mode == "mean_max" else hidden_size * 2
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        if self.pooling_mode == "last_hidden":
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            feats = torch.cat([forward_last, backward_last], dim=1)
        else:
            sequence_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
            max_steps = sequence_out.size(1)
            time_idx = torch.arange(max_steps, device=lengths.device).unsqueeze(0)
            mask = (time_idx < lengths.unsqueeze(1)).unsqueeze(-1)
            sum_pool = (sequence_out * mask).sum(dim=1)
            mean_pool = sum_pool / lengths.clamp(min=1).unsqueeze(1)
            min_value = torch.finfo(sequence_out.dtype).min
            max_pool = sequence_out.masked_fill(~mask, min_value).max(dim=1).values
            feats = torch.cat([mean_pool, max_pool], dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)


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
        for input_ids, lengths, labels in tqdm(loader, desc=desc, leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
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
    model = BiLSTMEmotionClassifier(
        embedding_matrix=checkpoint["embedding_matrix"],
        hidden_size=model_params["hidden_size"],
        num_layers=model_params["num_layers"],
        dropout=model_params["dropout"],
        freeze_embeddings=model_params["freeze_embeddings"],
        n_classes=model_params["n_classes"],
        pooling_mode=model_params.get("pooling_mode", "last_hidden"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Модель загружена из {model_path}")
    return model, checkpoint


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


def train_bilstm(
    embeddings_path: Path | None = None,
    save: bool = True,
    epochs: int = 12,
    batch_size: int = 64,
    max_len: int = 64,
    max_vocab_size: int = 50000,
    min_freq: int = 1,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    freeze_embeddings: bool = False,
    lr: float = 1e-3,
    lr_embeddings: float | None = None,
    weight_decay: float = 1e-5,
    val_size: float = 0.1,
    seed: int = 42,
    device_arg: str = "auto",
):
    if embeddings_path is None:
        embeddings_path = DEFAULT_EMBEDDINGS_PATH

    set_seed(seed)
    device = resolve_device(device_arg)
    use_cuda = device.type == "cuda"
    print(f"Обучение запущено на устройстве: {device}")

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'=' * 60}")
    fasttext_model = load_fasttext_model(embeddings_path)

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

    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size должен быть в интервале (0, 1), получено: {val_size}")

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
    print("ПОСТРОЕНИЕ СЛОВАРЯ И EMBEDDING MATRIX")
    print(f"{'=' * 60}")
    word2idx = build_vocab(train_texts, max_vocab_size=max_vocab_size, min_freq=min_freq)
    embedding_matrix = build_embedding_matrix(word2idx, fasttext_model)
    print(f"✓ Размер словаря: {len(word2idx)}")
    print(f"✓ Размер embedding matrix: {embedding_matrix.shape}")

    train_ds = TextSequenceDataset(train_texts, y_train_raw, word2idx, max_len=max_len)
    val_ds = TextSequenceDataset(val_texts, y_val_raw, word2idx, max_len=max_len)
    test_ds = TextSequenceDataset(test_texts, y_test_raw, word2idx, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(val_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
    test_loader = _build_loader(test_ds, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = BiLSTMEmotionClassifier(
        embedding_matrix=embedding_matrix,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
        n_classes=len(TARGET_NAMES),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if lr_embeddings is None:
        lr_embeddings = lr * 0.1
    if freeze_embeddings:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        non_embedding_params = [
            param
            for name, param in model.named_parameters()
            if not name.startswith("embedding.")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.embedding.parameters(), "lr": lr_embeddings},
                {"params": non_embedding_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )

    print(f"\nРазмер train: {len(train_ds)}")
    print(f"Размер val:   {len(val_ds)}")
    print(f"Размер test:  {len(test_ds)}")
    print(
        f"Эпох: {epochs}, batch_size: {batch_size}, max_len: {max_len}, "
        f"hidden_size: {hidden_size}, num_layers: {num_layers}, dropout: {dropout}"
    )
    print(
        f"freeze_embeddings: {freeze_embeddings}, lr: {lr}, "
        f"lr_embeddings: {0.0 if freeze_embeddings else lr_embeddings}"
    )
    print("epoch | train_loss | train_f1 | val_loss | val_f1")

    best_state = None
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        all_train_preds = []
        all_train_targets = []

        progress = tqdm(train_loader, desc=f"Train {epoch:02d}/{epochs}", leave=False)
        for input_ids, lengths, labels in progress:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
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
            f"{epoch:02d} | {train_loss:.4f} | {train_f1:.4f} | "
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
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "freeze_embeddings": freeze_embeddings,
            "n_classes": len(TARGET_NAMES),
            "max_len": max_len,
            "pooling_mode": "mean_max",
        }
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "embedding_matrix": embedding_matrix,
            "word2idx": word2idx,
            "target_names": TARGET_NAMES,
            "model_params": model_params,
        }
        training_params = {
            "embeddings_path": str(embeddings_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "max_len": max_len,
            "max_vocab_size": max_vocab_size,
            "min_freq": min_freq,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "freeze_embeddings": freeze_embeddings,
            "lr": lr,
            "lr_embeddings": 0.0 if freeze_embeddings else float(lr_embeddings),
            "weight_decay": weight_decay,
            "val_size": val_size,
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

    model, checkpoint = load_model(dataset_name, map_location=device)
    model = model.to(device)
    model.eval()

    word2idx = checkpoint["word2idx"]
    model_params = checkpoint["model_params"]
    max_len = int(model_params["max_len"])

    train_texts, y_train_raw = load_texts_from_manifest(train_manifest)
    test_texts, y_test_raw = load_texts_from_manifest(test_manifest)

    train_ds = TextSequenceDataset(train_texts, y_train_raw, word2idx, max_len=max_len)
    test_ds = TextSequenceDataset(test_texts, y_test_raw, word2idx, max_len=max_len)
    train_loader = _build_loader(train_ds, batch_size=64, shuffle=False, use_cuda=device.type == "cuda")
    test_loader = _build_loader(test_ds, batch_size=64, shuffle=False, use_cuda=device.type == "cuda")
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
        description="Обучение или загрузка модели FastText Embeddings + BiLSTM для классификации эмоций"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "load", "auto"],
        default="auto",
        help="train - обучить новую модель, load - загрузить существующую, auto - загрузить если есть",
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять модель после обучения")
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help=f"Путь к FastText embeddings (.bin). По умолчанию: {DEFAULT_EMBEDDINGS_PATH}",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-embeddings",
        type=float,
        default=None,
        help="LR для embedding-слоя (по умолчанию lr*0.1 при разморозке).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-embeddings",
        dest="freeze_embeddings",
        action="store_true",
        help="Заморозить embedding слой.",
    )
    parser.add_argument(
        "--no-freeze-embeddings",
        dest="freeze_embeddings",
        action="store_false",
        help="Разрешить дообучение embedding слоя.",
    )
    parser.set_defaults(freeze_embeddings=False)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Устройство обучения.",
    )
    args = parser.parse_args()

    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Требуется библиотека gensim. Установите: pip install gensim или poetry add gensim"
        )

    dataset_name = get_dataset_name(TRAIN_DATA_PATH)
    if args.mode == "train":
        train_bilstm(
            embeddings_path=args.embeddings_path,
            save=not args.no_save,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_len=args.max_len,
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            freeze_embeddings=args.freeze_embeddings,
            lr=args.lr,
            lr_embeddings=args.lr_embeddings,
            weight_decay=args.weight_decay,
            val_size=args.val_size,
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
            train_bilstm(
                embeddings_path=args.embeddings_path,
                save=not args.no_save,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_len=args.max_len,
                max_vocab_size=args.max_vocab_size,
                min_freq=args.min_freq,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                freeze_embeddings=args.freeze_embeddings,
                lr=args.lr,
                lr_embeddings=args.lr_embeddings,
                weight_decay=args.weight_decay,
                val_size=args.val_size,
                seed=args.seed,
                device_arg=args.device,
            )
