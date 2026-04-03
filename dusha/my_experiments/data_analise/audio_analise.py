"""
Анализ аудио признаков: mel-спектрограммы, MFCC, визуализация

Скрипт выполняет глубокий анализ аудио признаков по эмоциям:
- Загрузка mel-спектрограмм из .npy файлов
- Построение усредненных mel-спектрограмм по 4 эмоциям
- Вычисление и визуализация средних значений MFCC по эмоциям
- PCA визуализация для понимания разделимости классов
- t-SNE визуализация для выявления кластеров

Использование:
    poetry run python dusha/my_experiments/data_analise/audio_analise.py
    poetry run python dusha/my_experiments/data_analise/audio_analise.py --dataset combine_balanced_train_small
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Опциональные импорты
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Заглушка для tqdm если не установлен
    def tqdm(iterable, **kwargs):
        return iterable

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid' if SEABORN_AVAILABLE else 'default')

# Маппинг эмоций
EMOTION_LABELS = {
    'angry': 0,
    'sad': 1,
    'neutral': 2,
    'positive': 3
}

EMOTION_NAMES = ['angry', 'sad', 'neutral', 'positive']
EMOTION_COLORS = {
    'angry': 'red',
    'sad': 'blue',
    'neutral': 'gray',
    'positive': 'green'
}


def load_features_from_manifest(manifest_path, base_path=None, max_samples=None):
    """
    Загрузка mel-спектрограмм из JSONL манифеста.
    
    Args:
        manifest_path: Путь к .jsonl файлу с метаданными
        base_path: Базовый путь к датасету (если None, используется родительская папка манифеста)
        max_samples: Максимальное количество сэмплов для загрузки (None = все)
        
    Returns:
        features: Список mel-спектрограмм
        labels: Массив меток эмоций
        metadata: Список словарей с метаданными
    """
    features = []
    labels = []
    metadata = []
    
    if base_path is None:
        base_path = Path(manifest_path).parent.parent.parent
    
    print(f"Загрузка данных из {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        if max_samples:
            lines = lines[:max_samples]
        
        for line in tqdm(lines, desc="Загрузка mel-спектрограмм"):
            try:
                data = json.loads(line.strip())
                
                # Путь к файлу с признаками
                hash_id = data.get('hash_id')
                if hash_id:
                    # Для aggregated_dataset используем hash_id
                    npy_path = base_path / 'features' / f'{hash_id}.npy'
                else:
                    # Для других датасетов используем поле id
                    npy_path = base_path / 'features' / f"{data['id']}.npy"
                
                if not npy_path.exists():
                    continue
                
                # Загрузка mel-спектрограммы
                mel_spec = np.load(npy_path)
                
                features.append(mel_spec)
                labels.append(data['emotion'])
                metadata.append(data)
                
            except Exception as e:
                continue
    
    print(f"✓ Загружено {len(features)} mel-спектрограмм")
    
    return features, np.array(labels), metadata


def compute_average_mel_spectrograms(features, labels):
    """
    Вычисляет усредненные mel-спектрограммы для каждой эмоции.
    
    Процесс:
    1. Группируем спектрограммы по эмоциям
    2. Приводим все к единой временной длине (усреднение или обрезка)
    3. Усредняем спектрограммы внутри каждой группы
    
    Args:
        features: Список mel-спектрограмм (каждая может быть разной длины)
        labels: Массив меток эмоций
        
    Returns:
        avg_spectrograms: Словарь {emotion: averaged_mel_spectrogram}
        emotion_counts: Словарь {emotion: count}
    """
    print(f"\n{'='*60}")
    print("ВЫЧИСЛЕНИЕ УСРЕДНЕННЫХ MEL-СПЕКТРОГРАММ")
    print(f"{'='*60}")
    
    # Группируем спектрограммы по эмоциям
    emotion_spectrograms = {emotion: [] for emotion in EMOTION_NAMES}
    
    for mel_spec, label in zip(features, labels):
        if label in emotion_spectrograms:
            # Убираем batch dimension если есть
            if mel_spec.ndim == 3:
                mel_spec = mel_spec[0]
            emotion_spectrograms[label].append(mel_spec)
    
    # Вычисляем средние спектрограммы
    avg_spectrograms = {}
    emotion_counts = {}
    
    for emotion, spectrograms in emotion_spectrograms.items():
        if not spectrograms:
            print(f"⚠ Нет данных для эмоции: {emotion}")
            continue
        
        emotion_counts[emotion] = len(spectrograms)
        
        # Находим целевой размер (медиана длин)
        lengths = [spec.shape[1] for spec in spectrograms]
        target_length = int(np.median(lengths))
        
        # Приводим все спектрограммы к одной длине
        normalized_specs = []
        for spec in spectrograms:
            if spec.shape[1] > target_length:
                # Обрезаем
                spec = spec[:, :target_length]
            elif spec.shape[1] < target_length:
                # Дополняем нулями
                pad_width = ((0, 0), (0, target_length - spec.shape[1]))
                spec = np.pad(spec, pad_width, mode='constant', constant_values=-80)
            normalized_specs.append(spec)
        
        # Усредняем
        avg_spec = np.mean(normalized_specs, axis=0)
        avg_spectrograms[emotion] = avg_spec
        
        print(f"  {emotion:8s}: {emotion_counts[emotion]} сэмплов, "
              f"форма усредненной спектрограммы: {avg_spec.shape}")
    
    return avg_spectrograms, emotion_counts


def plot_average_mel_spectrograms(avg_spectrograms, save_path=None):
    """
    Визуализация усредненных mel-спектрограмм для каждой эмоции.
    
    Args:
        avg_spectrograms: Словарь {emotion: averaged_mel_spectrogram}
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ УСРЕДНЕННЫХ MEL-СПЕКТРОГРАММ")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTION_NAMES):
        if emotion not in avg_spectrograms:
            continue
        
        ax = axes[idx]
        mel_spec = avg_spectrograms[emotion]
        
        # Отображаем спектрограмму
        im = ax.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        
        ax.set_title(f'{emotion.capitalize()} (усредненная)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Временные фреймы', fontsize=11)
        ax.set_ylabel('Mel bins', fontsize=11)
        
        # Добавляем colorbar
        plt.colorbar(im, ax=ax, label='Амплитуда (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def extract_mfcc_from_mel(mel_spec, n_mfcc=13):
    """
    Извлекает MFCC коэффициенты из mel-спектрограммы.
    
    Упрощенная версия: берем DCT от mel-спектрограммы.
    В реальности librosa делает это более сложным способом.
    
    Args:
        mel_spec: Mel-спектрограмма (n_mels, time)
        n_mfcc: Количество MFCC коэффициентов
        
    Returns:
        mfcc: MFCC коэффициенты (n_mfcc, time)
    """
    from scipy.fftpack import dct
    
    # Применяем DCT-II к каждому временному фрейму
    mfcc = dct(mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    
    return mfcc


def compute_average_mfcc(features, labels, n_mfcc=13):
    """
    Вычисляет средние значения MFCC по эмоциям.
    
    Args:
        features: Список mel-спектрограмм
        labels: Массив меток эмоций
        n_mfcc: Количество MFCC коэффициентов
        
    Returns:
        avg_mfcc: Словарь {emotion: averaged_mfcc}
        mfcc_stats: Словарь {emotion: {'mean': mean_vector, 'std': std_vector}}
    """
    print(f"\n{'='*60}")
    print("ВЫЧИСЛЕНИЕ СРЕДНИХ ЗНАЧЕНИЙ MFCC")
    print(f"{'='*60}")
    
    # Группируем MFCC по эмоциям
    emotion_mfcc = {emotion: [] for emotion in EMOTION_NAMES}
    
    for mel_spec, label in tqdm(zip(features, labels), total=len(features), desc="Извлечение MFCC"):
        if label not in emotion_mfcc:
            continue
        
        # Убираем batch dimension если есть
        if mel_spec.ndim == 3:
            mel_spec = mel_spec[0]
        
        # Извлекаем MFCC
        mfcc = extract_mfcc_from_mel(mel_spec, n_mfcc=n_mfcc)
        
        # Усредняем по времени
        mfcc_mean = np.mean(mfcc, axis=1)
        emotion_mfcc[label].append(mfcc_mean)
    
    # Вычисляем статистики
    avg_mfcc = {}
    mfcc_stats = {}
    
    for emotion, mfcc_list in emotion_mfcc.items():
        if not mfcc_list:
            continue
        
        mfcc_array = np.array(mfcc_list)
        mean_mfcc = np.mean(mfcc_array, axis=0)
        std_mfcc = np.std(mfcc_array, axis=0)
        
        avg_mfcc[emotion] = mean_mfcc
        mfcc_stats[emotion] = {
            'mean': mean_mfcc,
            'std': std_mfcc,
            'count': len(mfcc_list)
        }
        
        print(f"  {emotion:8s}: {len(mfcc_list)} сэмплов, MFCC форма: {mean_mfcc.shape}")
    
    return avg_mfcc, mfcc_stats


def plot_average_mfcc(avg_mfcc, mfcc_stats, save_path=None):
    """
    Визуализация средних значений MFCC по эмоциям.
    
    Args:
        avg_mfcc: Словарь {emotion: averaged_mfcc}
        mfcc_stats: Словарь {emotion: {'mean': mean_vector, 'std': std_vector}}
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ СРЕДНИХ ЗНАЧЕНИЙ MFCC")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Средние значения MFCC
    ax1 = axes[0]
    for emotion in EMOTION_NAMES:
        if emotion not in avg_mfcc:
            continue
        
        mfcc_mean = mfcc_stats[emotion]['mean']
        mfcc_std = mfcc_stats[emotion]['std']
        
        x = np.arange(len(mfcc_mean))
        ax1.plot(x, mfcc_mean, marker='o', label=emotion.capitalize(), 
                color=EMOTION_COLORS[emotion], linewidth=2)
        ax1.fill_between(x, mfcc_mean - mfcc_std, mfcc_mean + mfcc_std,
                        alpha=0.2, color=EMOTION_COLORS[emotion])
    
    ax1.set_xlabel('MFCC коэффициент', fontsize=12)
    ax1.set_ylabel('Среднее значение', fontsize=12)
    ax1.set_title('Средние значения MFCC по эмоциям', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # График 2: Heatmap MFCC
    ax2 = axes[1]
    mfcc_matrix = []
    emotion_order = []
    
    for emotion in EMOTION_NAMES:
        if emotion in avg_mfcc:
            mfcc_matrix.append(avg_mfcc[emotion])
            emotion_order.append(emotion.capitalize())
    
    if mfcc_matrix:
        mfcc_matrix = np.array(mfcc_matrix)
        
        im = ax2.imshow(mfcc_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax2.set_yticks(range(len(emotion_order)))
        ax2.set_yticklabels(emotion_order)
        ax2.set_xlabel('MFCC коэффициент', fontsize=12)
        ax2.set_title('Heatmap средних MFCC', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Значение')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def prepare_features_for_visualization(features, labels):
    """
    Подготовка признаков для PCA и t-SNE визуализации.
    
    Преобразует mel-спектрограммы в векторы фиксированной длины
    через усреднение по временной оси.
    
    Args:
        features: Список mel-спектрограмм
        labels: Массив меток эмоций
        
    Returns:
        X: Матрица признаков (n_samples, n_features)
        y: Массив числовых меток
        label_names: Массив строковых меток
    """
    print(f"\n{'='*60}")
    print("ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ ВИЗУАЛИЗАЦИИ")
    print(f"{'='*60}")
    
    feature_vectors = []
    label_names = []
    
    for mel_spec, label in tqdm(zip(features, labels), total=len(features), 
                                desc="Преобразование в векторы"):
        # Убираем batch dimension если есть
        if mel_spec.ndim == 3:
            mel_spec = mel_spec[0]
        
        # Усредняем по временной оси и вычисляем std
        mean_part = mel_spec.mean(axis=1)
        std_part = mel_spec.std(axis=1)
        
        # Объединяем mean и std
        feature_vector = np.concatenate([mean_part, std_part])
        
        feature_vectors.append(feature_vector)
        label_names.append(label)
    
    X = np.array(feature_vectors)
    y = np.array([EMOTION_LABELS[label] for label in label_names])
    
    print(f"✓ Матрица признаков: {X.shape}")
    print(f"  - Количество сэмплов: {X.shape[0]}")
    print(f"  - Размерность признаков: {X.shape[1]}")
    
    return X, y, np.array(label_names)


def plot_pca_visualization(X, y, label_names, save_path=None):
    """
    PCA визуализация признаков в 2D пространстве.
    
    PCA (Principal Component Analysis) находит главные компоненты,
    которые объясняют максимальную дисперсию в данных.
    
    Args:
        X: Матрица признаков (n_samples, n_features)
        y: Массив числовых меток
        label_names: Массив строковых меток
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("PCA ВИЗУАЛИЗАЦИЯ")
    print(f"{'='*60}")
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA до 2 компонент
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Информация о объясненной дисперсии
    explained_var = pca.explained_variance_ratio_
    print(f"✓ PCA выполнена")
    print(f"  - Объясненная дисперсия PC1: {explained_var[0]*100:.2f}%")
    print(f"  - Объясненная дисперсия PC2: {explained_var[1]*100:.2f}%")
    print(f"  - Суммарная объясненная дисперсия: {explained_var.sum()*100:.2f}%")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for emotion in EMOTION_NAMES:
        mask = label_names == emotion
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion.capitalize(),
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% дисперсии)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% дисперсии)', fontsize=12)
    ax.set_title('PCA визуализация mel-спектрограмм по эмоциям', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def plot_tsne_visualization(X, y, label_names, save_path=None):
    """
    t-SNE визуализация признаков в 2D пространстве.
    
    t-SNE (t-Distributed Stochastic Neighbor Embedding) лучше сохраняет
    локальную структуру данных и часто выявляет кластеры лучше, чем PCA.
    
    Args:
        X: Матрица признаков (n_samples, n_features)
        y: Массив числовых меток
        label_names: Массив строковых меток
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("t-SNE ВИЗУАЛИЗАЦИЯ")
    print(f"{'='*60}")
    print("⏳ t-SNE может занять несколько минут...")
    
    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE до 2 компонент
    # perplexity: баланс между локальной и глобальной структурой (5-50)
    # learning_rate: скорость обучения (обычно 10-1000)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    print(f"✓ t-SNE выполнена")
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for emotion in EMOTION_NAMES:
        mask = label_names == emotion
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion.capitalize(),
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.set_title('t-SNE визуализация mel-спектрограмм по эмоциям', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def analyze_audio_features(manifest_path, max_samples=None, save_plots=False):
    """
    Основная функция анализа аудио признаков.
    
    Выполняет полный анализ:
    1. Загрузка данных
    2. Усредненные mel-спектрограммы
    3. Средние значения MFCC
    4. PCA визуализация
    5. t-SNE визуализация
    
    Args:
        manifest_path: Путь к манифесту
        max_samples: Максимальное количество сэмплов (None = все)
        save_plots: Сохранять ли графики на диск
    """
    print(f"\n{'='*60}")
    print("АНАЛИЗ АУДИО ПРИЗНАКОВ")
    print(f"{'='*60}")
    print(f"Манифест: {manifest_path}")
    if max_samples:
        print(f"Ограничение: {max_samples} сэмплов")
    
    # Создаем директорию для сохранения графиков
    output_dir = None
    if save_plots:
        output_dir = Path(__file__).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        print(f"Графики будут сохранены в: {output_dir}")
    
    # 1. Загрузка данных
    features, labels, metadata = load_features_from_manifest(
        manifest_path,
        base_path=DATASET_PATH,
        max_samples=max_samples
    )
    
    if len(features) == 0:
        print("❌ Не удалось загрузить данные!")
        return
    
    # Вывод статистики
    print(f"\nСтатистика по эмоциям:")
    unique, counts = np.unique(labels, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"  {emotion:8s}: {count} ({count/len(labels)*100:.1f}%)")
    
    # 2. Усредненные mel-спектрограммы
    avg_spectrograms, emotion_counts = compute_average_mel_spectrograms(features, labels)
    save_path = output_dir / "avg_mel_spectrograms.png" if save_plots else None
    plot_average_mel_spectrograms(avg_spectrograms, save_path=save_path)
    
    # 3. Средние значения MFCC
    avg_mfcc, mfcc_stats = compute_average_mfcc(features, labels, n_mfcc=13)
    save_path = output_dir / "avg_mfcc.png" if save_plots else None
    plot_average_mfcc(avg_mfcc, mfcc_stats, save_path=save_path)
    
    # 4. Подготовка признаков для визуализации
    X, y, label_names = prepare_features_for_visualization(features, labels)
    
    # 5. PCA визуализация
    save_path = output_dir / "pca_visualization.png" if save_plots else None
    plot_pca_visualization(X, y, label_names, save_path=save_path)
    
    # 6. t-SNE визуализация
    save_path = output_dir / "tsne_visualization.png" if save_plots else None
    plot_tsne_visualization(X, y, label_names, save_path=save_path)
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*60}")
    
    if save_plots:
        print(f"\n✓ Все графики сохранены в: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Анализ аудио признаков: mel-спектрограммы, MFCC, PCA, t-SNE'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='combine_balanced_train_small',
        help='Название датасета (имя файла без .jsonl)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Максимальное количество сэмплов для анализа (по умолчанию: все)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Сохранять графики в папку visualizations/'
    )
    
    args = parser.parse_args()
    
    # Путь к манифесту
    base_path = DATASET_PATH / 'processed_dataset_090'
    manifest_path = base_path / 'aggregated_dataset' / f'{args.dataset}.jsonl'
    
    if not manifest_path.exists():
        print(f"❌ Манифест не найден: {manifest_path}")
        print("\nДоступные манифесты:")
        manifest_dir = base_path / 'aggregated_dataset'
        if manifest_dir.exists():
            for f in manifest_dir.glob('*.jsonl'):
                print(f"  - {f.stem}")
        exit(1)
    
    # Запуск анализа
    analyze_audio_features(
        manifest_path,
        max_samples=args.max_samples,
        save_plots=args.save_plots
    )
