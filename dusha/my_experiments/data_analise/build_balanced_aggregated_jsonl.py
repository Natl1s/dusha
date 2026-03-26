import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TARGET_EMOTIONS = ("angry", "sad", "neutral", "positive")


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write(os.linesep)


def filter_target_emotions(records: Sequence[dict]) -> List[dict]:
    return [r for r in records if r.get("emotion") in TARGET_EMOTIONS]


def count_by_emotion(records: Sequence[dict]) -> Dict[str, int]:
    counter = Counter(r.get("emotion") for r in records)
    return {emo: int(counter.get(emo, 0)) for emo in TARGET_EMOTIONS}


def print_dataset_stats(dataset_name: str, records: Sequence[dict]) -> None:
    counts = count_by_emotion(records)
    print(f"\n{dataset_name}")
    print(f"  total: {len(records)}")
    for emo in TARGET_EMOTIONS:
        print(f"  {emo}: {counts[emo]}")


def sample_records(records: Sequence[dict], k: int, rng: random.Random) -> List[dict]:
    if k <= 0:
        return []
    if k >= len(records):
        return list(records)
    return rng.sample(list(records), k=k)


def build_balanced_full(records: Sequence[dict], rng: random.Random) -> List[dict]:
    by_emotion: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        by_emotion[row["emotion"]].append(row)

    non_neutral_counts = [len(by_emotion[e]) for e in TARGET_EMOTIONS if e != "neutral"]
    if any(c == 0 for c in non_neutral_counts):
        raise ValueError(
            "Не удалось собрать full-набор: один из не-neutral классов пустой "
            f"(counts={ {e: len(by_emotion[e]) for e in TARGET_EMOTIONS} })"
        )

    min_non_neutral = min(non_neutral_counts)
    neutral_cap = 2 * min_non_neutral

    selected: List[dict] = []
    for emo in TARGET_EMOTIONS:
        emo_records = by_emotion[emo]
        if emo == "neutral":
            selected.extend(sample_records(emo_records, min(len(emo_records), neutral_cap), rng))
        else:
            selected.extend(emo_records)

    rng.shuffle(selected)
    return selected


def _scaled_targets_with_same_ratio(
    counts: Dict[str, int], ratio: float, min_per_present_class: int = 1
) -> Dict[str, int]:
    present = [e for e in TARGET_EMOTIONS if counts.get(e, 0) > 0]
    if not present:
        return {e: 0 for e in TARGET_EMOTIONS}

    total = sum(counts[e] for e in present)
    target_total = int(round(total * ratio))
    target_total = max(target_total, len(present) * min_per_present_class)
    target_total = min(target_total, total)

    raw = {e: counts[e] * ratio for e in present}
    floor_counts = {
        e: min(counts[e], max(min_per_present_class, int(raw[e])))
        for e in present
    }

    current = sum(floor_counts.values())

    if current < target_total:
        # Добавляем элементы по убыванию дробной части, пока не доберем нужный total.
        remainders = sorted(
            present,
            key=lambda e: (raw[e] - int(raw[e]), counts[e]),
            reverse=True,
        )
        idx = 0
        while current < target_total:
            e = remainders[idx % len(remainders)]
            if floor_counts[e] < counts[e]:
                floor_counts[e] += 1
                current += 1
            idx += 1
            if idx > 1000000:
                raise RuntimeError("Слишком много итераций при распределении target counts")

    elif current > target_total:
        # Убираем элементы из классов с наибольшим текущим размером, но не ниже min_per_present_class.
        removable = sorted(present, key=lambda e: floor_counts[e], reverse=True)
        idx = 0
        while current > target_total:
            e = removable[idx % len(removable)]
            if floor_counts[e] > min_per_present_class:
                floor_counts[e] -= 1
                current -= 1
            idx += 1
            if idx > 1000000:
                raise RuntimeError("Слишком много итераций при уменьшении target counts")

    targets = {e: 0 for e in TARGET_EMOTIONS}
    targets.update(floor_counts)
    return targets


def build_balanced_small(records: Sequence[dict], ratio: float, rng: random.Random) -> List[dict]:
    by_emotion: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        by_emotion[row["emotion"]].append(row)

    counts = {e: len(by_emotion[e]) for e in TARGET_EMOTIONS}
    targets = _scaled_targets_with_same_ratio(counts, ratio=ratio, min_per_present_class=1)

    selected: List[dict] = []
    for emo in TARGET_EMOTIONS:
        selected.extend(sample_records(by_emotion[emo], targets[emo], rng))

    rng.shuffle(selected)
    return selected


def load_split_pair(aggregated_dir: Path, split: str) -> List[dict]:
    crowd_path = aggregated_dir / f"crowd_{split}.jsonl"
    podcast_path = aggregated_dir / f"podcast_{split}.jsonl"

    crowd = read_jsonl(crowd_path)
    podcast = read_jsonl(podcast_path)
    merged = crowd + podcast
    filtered = filter_target_emotions(merged)

    dropped = len(merged) - len(filtered)
    if dropped > 0:
        print(f"[{split}] Исключено записей с эмоциями вне {TARGET_EMOTIONS}: {dropped}")

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Собирает сбалансированные объединенные наборы в aggregated_dataset: "
            "combine_balanced_train/test и *_small (30%)."
        )
    )
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=Path("/home/natlis/PycharmProjects/dusha_new/dusha/dataset/processed_dataset_090/aggregated_dataset"),
        help="Путь к папке aggregated_dataset",
    )
    parser.add_argument("--small-ratio", type=float, default=0.3, help="Доля для small-наборов")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимого сэмплинга")

    args = parser.parse_args()
    rng = random.Random(args.seed)

    aggregated_dir = args.aggregated_dir
    train_merged = load_split_pair(aggregated_dir, "train")
    test_merged = load_split_pair(aggregated_dir, "test")

    combine_balanced_train = build_balanced_full(train_merged, rng=rng)
    combine_balanced_test = build_balanced_full(test_merged, rng=rng)

    combine_balanced_train_small = build_balanced_small(
        combine_balanced_train, ratio=args.small_ratio, rng=rng
    )
    combine_balanced_test_small = build_balanced_small(
        combine_balanced_test, ratio=args.small_ratio, rng=rng
    )

    outputs = {
        "combine_balanced_train": combine_balanced_train,
        "combine_balanced_test": combine_balanced_test,
        "combine_balanced_train_small": combine_balanced_train_small,
        "combine_balanced_test_small": combine_balanced_test_small,
    }

    for name, rows in outputs.items():
        out_path = aggregated_dir / f"{name}.jsonl"
        write_jsonl(out_path, rows)
        print_dataset_stats(name, rows)
        print(f"  saved_to: {out_path}")


if __name__ == "__main__":
    main()
