import argparse
import builtins
import json
import pickle
import random
import re
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
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from my_experiments.audio_models.CNN.CNN_BiLSTM import EmotionCNNBiLSTM
from my_experiments.lmdb_utils import get_lmdb_length, open_lmdb_readonly, parse_label_to_index
from my_experiments.text_models.transformers.RuBERT import EmotionClassifier


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

DEFAULT_AUDIO_MODEL_PATH = (
    PROJECT_MY_EXP_DIR
    / "audio_models"
    / "CNN"
    / "models_params"
    / "CNN_BiLSTM_combine_balanced_train_model.pt"
)
DEFAULT_TEXT_MODEL_PATH = (
    PROJECT_MY_EXP_DIR
    / "text_models"
    / "transformers"
    / "models_params"
    / "RuBERT_dusha_resd_train_model.pt"
)

TARGET_NAMES = ["angry", "sad", "neutral", "positive"]


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
            raise RuntimeError("Запрошено устройство cuda, но CUDA недоступна.")
        return torch.device("cuda:0")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Неподдерживаемое устройство: {device_arg}")


def _extract_text(payload: dict) -> str:
    text_keys = ("speaker_text", "text", "transcript", "utterance")
    for key in text_keys:
        if key in payload:
            text = str(payload[key]).strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                return text
    return ""


def _prepare_audio_tensor(payload: dict, idx: int) -> torch.Tensor:
    if "x" not in payload:
        raise KeyError(f"В payload нет ключа 'x' (idx={idx})")
    x = torch.from_numpy(np.asarray(payload["x"], dtype=np.float32))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3 and x.shape[0] != 1 and x.shape[-1] == 1:
        x = x.permute(2, 0, 1)
    elif x.ndim != 3:
        raise ValueError(f"Неподдерживаемая форма аудиотензора {tuple(x.shape)} (idx={idx})")
    return x


class FusionLmdbDataset(Dataset):
    def __init__(self, lmdb_path: Path, tokenizer, max_len: int):
        self.lmdb_path = Path(lmdb_path)
        self.env = open_lmdb_readonly(self.lmdb_path)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

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
                try:
                    _prepare_audio_tensor(payload, idx)
                except (KeyError, TypeError, ValueError):
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

    def __getitem__(self, item_idx):
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

        audio_x = _prepare_audio_tensor(payload, lmdb_idx)
        label = parse_label_to_index(payload.get("y", payload.get("label", payload.get("emotion"))))
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return audio_x, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


def fusion_collate_fn(batch):
    audio_xs, input_ids_list, attn_masks_list, labels = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in audio_xs], dtype=torch.long)
    max_t = int(lengths.max().item())
    padded_audio = []
    for x in audio_xs:
        delta = max_t - x.shape[-1]
        if delta > 0:
            x = torch.nn.functional.pad(x, pad=(0, delta, 0, 0))
        padded_audio.append(x)
    return (
        torch.stack(padded_audio),
        lengths,
        torch.stack(input_ids_list),
        torch.stack(attn_masks_list),
        torch.stack(labels),
    )


def _parse_training_params_from_report(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"training_params:\s*(\{.*?\})\s*test_metrics:", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def load_audio_model(audio_model_path: Path, device: torch.device) -> EmotionCNNBiLSTM:
    model_path = Path(audio_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден аудио-чекпоинт: {model_path}")

    report_path = model_path.parent / f"{model_path.stem}_training_report.txt"
    training_params = _parse_training_params_from_report(report_path)

    model = EmotionCNNBiLSTM(
        n_classes=len(TARGET_NAMES),
        lstm_hidden_size=int(training_params.get("lstm_hidden_size", 128)),
        lstm_layers=int(training_params.get("lstm_layers", 2)),
        lstm_dropout=float(training_params.get("lstm_dropout", 0.2)),
        bidirectional=bool(training_params.get("bidirectional", True)),
    )
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys:
        print(f"WARNING: missing keys in audio state_dict: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"WARNING: unexpected keys in audio state_dict: {incompat.unexpected_keys}")
    model = model.to(device)
    model.eval()
    return model


def load_text_model(
    text_model_path: Path,
    tokenizer_dir: Path | None,
    device: torch.device,
) -> tuple[EmotionClassifier, AutoTokenizer]:
    model_path = Path(text_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден текстовый чекпоинт: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_params" not in checkpoint:
        raise ValueError("Ожидается checkpoint словарь от RuBERT со структурой model_params/model_state_dict.")

    model_params = checkpoint["model_params"]
    model = EmotionClassifier(
        model_name=model_params["backbone_name"],
        num_classes=model_params["n_classes"],
        dropout=model_params["dropout"],
        classifier_hidden_size=model_params.get("classifier_hidden_size"),
    )
    incompat = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompat.missing_keys:
        print(f"WARNING: missing keys in text state_dict: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"WARNING: unexpected keys in text state_dict: {incompat.unexpected_keys}")
    model = model.to(device)
    model.eval()

    if tokenizer_dir is None:
        stem = model_path.stem
        if stem.endswith("_model"):
            tokenizer_dir = model_path.parent / (stem[:-6] + "_tokenizer")
        else:
            tokenizer_dir = model_path.parent / (stem + "_tokenizer")
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_params["backbone_name"])
    return model, tokenizer


def collect_model_probs(
    audio_model: EmotionCNNBiLSTM,
    text_model: EmotionClassifier,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    probs_audio_all = []
    probs_text_all = []
    with torch.no_grad():
        for audio_x, lengths, input_ids, attention_mask, labels in tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
        ):
            audio_x = audio_x.to(device)
            lengths = lengths.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            audio_logits = audio_model(audio_x, lengths)
            text_logits = text_model(input_ids, attention_mask)
            probs_audio = torch.softmax(audio_logits, dim=1)
            probs_text = torch.softmax(text_logits, dim=1)

            y_true_all.append(labels.cpu().numpy())
            probs_audio_all.append(probs_audio.cpu().numpy())
            probs_text_all.append(probs_text.cpu().numpy())

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


def print_eval(title: str, y_true: np.ndarray, y_pred: np.ndarray, metrics: dict):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
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
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_NAMES)))))


def main():
    parser = argparse.ArgumentParser(description="Late Fusion (soft voting) для аудио+текст моделей.")
    parser.add_argument("--train-lmdb", type=Path, default=DEFAULT_TRAIN_LMDB)
    parser.add_argument("--test-lmdb", type=Path, default=DEFAULT_TEST_LMDB)
    parser.add_argument("--audio-model-path", type=Path, default=DEFAULT_AUDIO_MODEL_PATH)
    parser.add_argument("--text-model-path", type=Path, default=DEFAULT_TEXT_MODEL_PATH)
    parser.add_argument(
        "--text-tokenizer-dir",
        type=Path,
        default=None,
        help="Опционально: путь к директории токенайзера. Если не задан, определяется рядом с text-model-path.",
    )
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    args = parser.parse_args()

    if not args.train_lmdb.exists():
        raise FileNotFoundError(f"Train LMDB не найден: {args.train_lmdb}")
    if not args.test_lmdb.exists():
        raise FileNotFoundError(f"Test LMDB не найден: {args.test_lmdb}")
    if not args.audio_model_path.exists():
        raise FileNotFoundError(f"Аудио модель не найдена: {args.audio_model_path}")
    if not args.text_model_path.exists():
        raise FileNotFoundError(f"Текстовая модель не найдена: {args.text_model_path}")
    if not (0.0 < args.val_size < 1.0):
        raise ValueError(f"val-size должен быть в (0,1), получено: {args.val_size}")
    if not (0.0 < args.alpha_step <= 1.0):
        raise ValueError(f"alpha-step должен быть в (0,1], получено: {args.alpha_step}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Устройство: {device}")
    print(f"Train LMDB: {args.train_lmdb}")
    print(f"Test LMDB:  {args.test_lmdb}")
    print(f"Audio model: {args.audio_model_path}")
    print(f"Text model:  {args.text_model_path}")

    audio_model = load_audio_model(args.audio_model_path, device=device)
    text_model, tokenizer = load_text_model(
        args.text_model_path,
        tokenizer_dir=args.text_tokenizer_dir,
        device=device,
    )

    train_ds = FusionLmdbDataset(args.train_lmdb, tokenizer=tokenizer, max_len=args.max_len)
    test_ds = FusionLmdbDataset(args.test_lmdb, tokenizer=tokenizer, max_len=args.max_len)

    train_indices = np.arange(len(train_ds))
    _, val_indices = train_test_split(
        train_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_ds.labels,
    )
    val_ds = Subset(train_ds, val_indices.tolist())

    use_cuda = device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=fusion_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=fusion_collate_fn,
    )

    print(f"Размер train (валидные мультимодальные): {len(train_ds)}")
    print(f"Размер val: {len(val_ds)}")
    print(f"Размер test (валидные мультимодальные): {len(test_ds)}")

    y_val, p_audio_val, p_text_val = collect_model_probs(
        audio_model, text_model, val_loader, device, desc="Инференс val"
    )
    y_test, p_audio_test, p_text_test = collect_model_probs(
        audio_model, text_model, test_loader, device, desc="Инференс test"
    )

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 4)
    best_alpha = None
    best_val_f1 = -1.0
    print("\nПодбор alpha по val (macro-F1):")
    for alpha in tqdm(alphas, desc="Поиск alpha", unit="alpha"):
        val_eval = evaluate_fusion(y_val, p_audio_val, p_text_val, float(alpha))
        val_f1 = val_eval["metrics"]["f1_macro"]
        tqdm.write(f"alpha={alpha:.2f} -> val_f1_macro={val_f1:.6f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_alpha = float(alpha)

    assert best_alpha is not None
    print(f"\nЛучший alpha: {best_alpha:.2f} (val_f1_macro={best_val_f1:.6f})")

    val_best = evaluate_fusion(y_val, p_audio_val, p_text_val, best_alpha)
    test_best = evaluate_fusion(y_test, p_audio_test, p_text_test, best_alpha)
    print_eval("LATE FUSION @ VAL", y_val, val_best["y_pred"], val_best["metrics"])
    print_eval("LATE FUSION @ TEST", y_test, test_best["y_pred"], test_best["metrics"])


if __name__ == "__main__":
    main()
