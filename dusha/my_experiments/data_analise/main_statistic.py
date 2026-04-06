import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dusha.experiments.core.utils import load_jsonl_as_df

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']


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


def analyze_multiple_datasets(base_path, save_plots=False):
    """
    Анализ нескольких датасетов
    
    Args:
        base_path: Путь к обработанному датасету
        save_plots: Сохранять ли графики в папку visualizations/
    """
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
        else:
            print(f"\nФайл не найден: {manifest_path}")

    print_train_test_balance_shift(results)

    return results


# Использование
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Статистический анализ датасетов'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Сохранять графики в папку visualizations/'
    )
    
    args = parser.parse_args()
    
    # Путь к обработанному датасету из data.config
    base_path = DATASET_PATH / 'processed_dataset_090'
    
    # Создаем директорию для сохранения графиков
    output_dir = None
    if args.save_plots:
        output_dir = Path(__file__).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        print(f"\nГрафики будут сохранены в: {output_dir}")

    # Анализ всех доступных датасетов
    results = analyze_multiple_datasets(base_path, save_plots=args.save_plots)

    # Визуализация распределения эмоций
    if 'train_combined' in results:
        df = results['train_combined']

        # График 1: Распределение эмоций (bar chart)
        plt.figure(figsize=(10, 6))
        emotion_counts = df['emotion'].value_counts()
        plt.bar(emotion_counts.index, emotion_counts.values, color=['#FF6B6B', '#4ECDC4', '#95A5A6', '#52C76E'])
        plt.title('Распределение эмоций в обучающем датасете', fontsize=14, fontweight='bold')
        plt.xlabel('Эмоция', fontsize=12)
        plt.ylabel('Количество записей', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if args.save_plots:
            save_path = output_dir / "emotion_distribution_train.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ График сохранен: {save_path}")
        
        plt.show()
        
        # График 2: Распределение длительности аудио по эмоциям
        plt.figure(figsize=(12, 6))
        emotion_names = ['angry', 'sad', 'neutral', 'positive']
        colors = ['#FF6B6B', '#4ECDC4', '#95A5A6', '#52C76E']
        
        for emotion, color in zip(emotion_names, colors):
            emotion_df = df[df['emotion'] == emotion]
            if len(emotion_df) > 0:
                plt.hist(emotion_df['wav_length'], bins=50, alpha=0.6, label=emotion.capitalize(), color=color)
        
        plt.title('Распределение длительности аудио по эмоциям', fontsize=14, fontweight='bold')
        plt.xlabel('Длительность (сек)', fontsize=12)
        plt.ylabel('Количество', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if args.save_plots:
            save_path = output_dir / "duration_distribution_by_emotion.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ График сохранен: {save_path}")
        
        plt.show()
        
        # График 3: Boxplot длительности по эмоциям
        plt.figure(figsize=(10, 6))
        emotion_data = [df[df['emotion'] == emotion]['wav_length'].values 
                       for emotion in emotion_names]
        
        bp = plt.boxplot(emotion_data, labels=[e.capitalize() for e in emotion_names], 
                        patch_artist=True, showmeans=True)
        
        # Раскрашиваем boxplot
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        plt.title('Распределение длительности аудио (boxplot)', fontsize=14, fontweight='bold')
        plt.ylabel('Длительность (сек)', fontsize=12)
        plt.xlabel('Эмоция', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if args.save_plots:
            save_path = output_dir / "duration_boxplot_by_emotion.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ График сохранен: {save_path}")
        
        plt.show()
    
    if args.save_plots:
        print(f"\n{'='*60}")
        print("ВСЕ ГРАФИКИ СОХРАНЕНЫ")
        print(f"{'='*60}")
        print(f"Директория: {output_dir}")
        print("Файлы:")
        for plot_file in sorted(output_dir.glob("*.png")):
            print(f"  - {plot_file.name}")
        print(f"{'='*60}")