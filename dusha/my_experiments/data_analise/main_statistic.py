import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dusha.experiments.core.utils import load_jsonl_as_df


def analyze_emotion_distribution(manifest_path):
    """Анализ распределения эмоций в манифесте"""
    # Загрузка JSONL манифеста
    df = load_jsonl_as_df(manifest_path)

    # Подсчет записей по эмоциям
    emotion_counts = df['emotion'].value_counts()
    label_counts = df['label'].value_counts().sort_index()

    print(f"Всего записей: {len(df)}")
    print("\nРаспределение по текстовым меткам эмоций:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count / len(df) * 100:.1f}%)")

    print("\nРаспределение по числовым меткам:")
    emotion_names = {0: 'angry', 1: 'sad', 2: 'neutral', 3: 'positive'}
    for label, count in label_counts.items():
        emotion_name = emotion_names.get(label, f'unknown_{label}')
        print(f"  {label} ({emotion_name}): {count} ({count / len(df) * 100:.1f}%)")

        # Статистика по длительности
    print(f"\nСтатистика по длительности аудио:")
    print(f"  Минимальная: {df['wav_length'].min():.2f} сек")
    print(f"  Максимальная: {df['wav_length'].max():.2f} сек")
    print(f"  Средняя: {df['wav_length'].mean():.2f} сек")
    print(f"  Медиана: {df['wav_length'].median():.2f} сек")

    return df


def print_priority_balance_report(df, dataset_name):
    """Короткий приоритетный отчет по дисбалансу классов."""
    label_counts = df['label'].value_counts().sort_index()
    label_ratio = label_counts / label_counts.sum()

    max_count = int(label_counts.max())
    min_count = int(label_counts.min())
    majority_label = int(label_counts.idxmax())
    majority_ratio = float(label_ratio.max())
    imbalance_ratio = (max_count / min_count) if min_count > 0 else np.inf

    # Нормализованная энтропия: 1.0 ~= почти равномерное распределение
    entropy = float(-(label_ratio * np.log(label_ratio + 1e-12)).sum())
    max_entropy = float(np.log(len(label_counts))) if len(label_counts) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    print(f"\n[Приоритетный баланс] {dataset_name}")
    print(f"  Классов: {len(label_counts)}")
    print(f"  Мажоритарный класс: {majority_label} ({majority_ratio * 100:.1f}%)")
    print(f"  Соотношение max/min: {imbalance_ratio:.2f}")
    print(f"  Норм. энтропия: {normalized_entropy:.3f}")

    for label, count in label_counts.items():
        print(f"    label={label}: {count} ({label_ratio[label] * 100:.1f}%)")


def print_train_test_balance_shift(results):
    """Сравнение train/test по долям классов для crowd/podcast/combined."""
    compare_pairs = [
        ('crowd', 'train_crowd', 'test_crowd'),
        ('podcast', 'train_podcast', 'test_podcast'),
        ('combined', 'train_combined', 'test_combined'),
    ]

    print("\n=== Сдвиг распределений train/test (по долям классов) ===")
    for group_name, train_key, test_key in compare_pairs:
        train_df = results.get(train_key)
        test_df = results.get(test_key)

        if train_df is None or test_df is None:
            print(f"  [{group_name}] пропуск: отсутствует train или test")
            continue

        train_ratio = train_df['label'].value_counts(normalize=True)
        test_ratio = test_df['label'].value_counts(normalize=True)
        labels = sorted(set(train_ratio.index).union(set(test_ratio.index)))

        print(f"  [{group_name}]")
        for label in labels:
            tr = float(train_ratio.get(label, 0.0))
            te = float(test_ratio.get(label, 0.0))
            delta_pp = (te - tr) * 100.0
            print(
                f"    label={label}: train={tr * 100:.1f}% | "
                f"test={te * 100:.1f}% | delta={delta_pp:+.1f} п.п."
            )


def load_sample_features(df, manifest_path, n_samples=3, show_plots=True):
    """Загрузка и вывод примеров mel-спектрограмм из конкретного манифеста."""
    manifest_path = Path(manifest_path)
    print(f"\nЗагрузка {n_samples} примеров mel-спектрограмм из {manifest_path.name}:")

    for i in range(min(n_samples, len(df))):
        row = df.iloc[i]
        feature_path = (manifest_path.parent / row['tensor']).resolve()

        try:
            mel_spec = np.load(feature_path)
            print(f"  Пример {i + 1}: {row['emotion']}")
            print(f"    ID: {row['id']}")
            print(f"    Tensor в манифесте: {row['tensor']}")
            print(f"    Абсолютный путь: {feature_path}")
            print(f"    Форма спектрограммы: {mel_spec.shape}")
            print(f"    Длительность: {row['wav_length']} сек")
            print(f"    Диапазон значений: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")

            if show_plots:
                if mel_spec.ndim == 3:
                    to_show = mel_spec[0]
                elif mel_spec.ndim == 2:
                    to_show = mel_spec
                else:
                    print(f"    Неожиданная размерность ({mel_spec.ndim}), пропускаю визуализацию")
                    continue

                plt.figure(figsize=(8, 3))
                plt.imshow(to_show, aspect='auto', origin='lower')
                plt.colorbar(label='amplitude')
                plt.title(f"{manifest_path.stem}: {row['emotion']} ({row['id']})")
                plt.xlabel('time frames')
                plt.ylabel('mel bins')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"    Ошибка загрузки {feature_path}: {e}")


def analyze_multiple_datasets(base_path, n_mel_examples=2, show_plots=True):
    """Анализ нескольких датасетов"""
    base_path = Path(base_path)

    # Пути к различным манифестам
    manifests = {
        'train_crowd': base_path / 'train' / 'crowd_train.jsonl',
        'train_podcast': base_path / 'train' / 'podcast_train.jsonl',
        'test_crowd': base_path / 'test' / 'crowd_test.jsonl',
        'test_podcast': base_path / 'test' / 'podcast_test.jsonl',
        'train_combined': base_path / 'train' / 'train.jsonl',
        'test_combined': base_path / 'test' / 'test.jsonl'
    }

    results = {}

    for name, manifest_path in manifests.items():
        if manifest_path.exists():
            print(f"\n=== Анализ датасета: {name} ===")
            df = analyze_emotion_distribution(manifest_path)
            print_priority_balance_report(df, name)
            results[name] = df

            # Загрузка примеров features только для обучающих датасетов
            if 'train' in name and name != 'train_combined':
                load_sample_features(
                    df,
                    manifest_path=manifest_path,
                    n_samples=n_mel_examples,
                    show_plots=show_plots,
                )
        else:
            print(f"\nФайл не найден: {manifest_path}")

    print_train_test_balance_shift(results)

    return results


# Использование
if __name__ == "__main__":
    # Укажите путь к обработанному датасету
    base_path = Path('/home/natlis/PycharmProjects/dusha_new/dusha/dataset/processed_dataset_090')

    # Анализ всех доступных датасетов
    results = analyze_multiple_datasets(base_path, n_mel_examples=2, show_plots=True)

    # Дополнительно: визуализация распределения эмоций
    if 'train_combined' in results:
        df = results['train_combined']

        plt.figure(figsize=(10, 6))
        emotion_counts = df['emotion'].value_counts()
        plt.bar(emotion_counts.index, emotion_counts.values)
        plt.title('Распределение эмоций в обучающем датасете')
        plt.xlabel('Эмоция')
        plt.ylabel('Количество записей')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()