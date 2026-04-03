#!/usr/bin/env python3
# emotion_spectrogram_analysis_fixed.py
"""
✅ ИСПРАВЛЕНА ОШИБКА "setting an array element with a sequence"
- Убедимся, что ВСЕ спектрограммы имеют форму [100, 128]
- Добавим проверки и отладку
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import gc
import warnings
from models.choose_dataset import *

warnings.filterwarnings('ignore')

# =============================================================================
# ⚡ ПАРАМЕТРЫ
# =============================================================================
MAX_SAMPLES = 5000
TARGET_TIME = 100
TARGET_MELS = 128
TSNE_SAMPLES = 300
BATCH_SIZE = 100
SAMPLE_RATE = 16000

# =============================================================================
# 📁 ПУТИ
# =============================================================================
BASE_PATH = Path('/home/natlis/PycharmProjects/dusha/dusha/data_processing/dataset')
JSONL_PATH = BASE_PATH / 'processed_dataset_090' / 'train' / DATASET_NAME
WAV_BASE_PATHS = {
    'crowd_train': BASE_PATH / 'crowd_train' / 'wavs',
}

print(f"📁 JSONL: {JSONL_PATH}")


# =============================================================================
# 🔍 ЗАГРУЗКА JSONL
# =============================================================================
def load_jsonl_sampled(jsonl_path, max_samples=MAX_SAMPLES):
    data = []
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                    count += 1
                except:
                    continue
    print(f"📖 Загружено {len(data)} сэмплов")
    return pd.DataFrame(data)


# =============================================================================
# 🎵 ✅ ИСПРАВЛЕННОЕ ИЗВЛЕЧЕНИЕ + НОРМАЛИЗАЦИЯ
# =============================================================================
def extract_and_normalize_mel(wav_path, target_time=TARGET_TIME, target_mels=TARGET_MELS, sr=SAMPLE_RATE):
    """Извлекает И СРАЗУ нормализует к [target_time, target_mels]"""
    try:
        # Загрузка аудио
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)

        # Mel-спектрограмма
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=target_mels,
            n_fft=1024, hop_length=256
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        # ✅ НОРМАЛИЗАЦИЯ ВРЕМЕНИ ПРЯМО ЗДЕСЬ
        time_frames, freq_bins = mel_db.shape

        # Время: обрезаем/дополняем
        if time_frames > target_time:
            mel_norm = mel_db[:target_time, :]
        else:
            pad_time = target_time - time_frames
            mel_norm = np.pad(mel_db, ((0, pad_time), (0, 0)), mode='constant')

        # ✅ ГАРАНТИРУЕМ форму [100, 128]
        if mel_norm.shape != (target_time, target_mels):
            # print(f"⚠️ Исправляем форму {mel_norm.shape} -> {(target_time, target_mels)}")
            # Обрезаем или дополняем по обоим измерениям
            result = np.zeros((target_time, target_mels), dtype=np.float32)
            time_to_copy = min(target_time, mel_norm.shape[0])
            freq_to_copy = min(target_mels, mel_norm.shape[1])
            result[:time_to_copy, :freq_to_copy] = mel_norm[:time_to_copy, :freq_to_copy]
            return result

        return mel_norm

    except Exception as e:
        print(f"⚠️ Ошибка {wav_path}: {e}")
        # Возвращаем пустую спектрограмму правильной формы
        return np.zeros((target_time, target_mels), dtype=np.float32)


def load_spectrograms_from_wav(df, wav_base_paths, batch_size=BATCH_SIZE):
    """✅ БЕЗОПАСНАЯ загрузка - все спектрограммы одинаковой формы"""
    all_specs = []  # Список нормализованных спектрограмм
    failed_files = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_specs = []

        print(f"🔄 Батч {i // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")

        for idx, row in batch_df.iterrows():
            filename = row.get('id', row.get('filename', ''))
            audio_path = row.get('audio_path', '')

            # Поиск WAV файла
            wav_file = None
            for dataset_type, wav_base in wav_base_paths.items():
                possible_paths = [
                    wav_base / f"{filename}.wav",
                    wav_base / f"{filename}.WAV",
                    wav_base / audio_path if audio_path else None
                ]
                for p in possible_paths:
                    if p and p.exists():
                        wav_file = p
                        break
                if wav_file:
                    break

            if wav_file:
                # ✅ Извлекаем + нормализуем СРАЗУ
                mel_norm = extract_and_normalize_mel(wav_file)

                # ✅ ПРОВЕРКА формы
                if mel_norm.shape != (TARGET_TIME, TARGET_MELS):
                    print(f"❌ Неправильная форма {mel_norm.shape} для файла {filename}")
                    print(f"   Исправляем...")
                    mel_norm = np.zeros((TARGET_TIME, TARGET_MELS), dtype=np.float32)

                batch_specs.append(mel_norm)
            else:
                print(f"❌ WAV не найден: {filename} (путь: {audio_path})")
                # Пустая спектрограмма правильной формы
                empty_spec = np.zeros((TARGET_TIME, TARGET_MELS), dtype=np.float32)
                batch_specs.append(empty_spec)
                failed_files.append(filename)

        # ✅ Все спектрограммы в батче должны быть одинаковой формы!
        # Проверяем первую спектрограмму в батче
        if batch_specs:
            expected_shape = batch_specs[0].shape
            for j, spec in enumerate(batch_specs):
                if spec.shape != expected_shape:
                    print(f"⚠️ Исправляем спектрограмму {j}: {spec.shape} -> {expected_shape}")
                    # Создаем спектрограмму правильной формы
                    fixed_spec = np.zeros(expected_shape, dtype=np.float32)
                    time_to_copy = min(expected_shape[0], spec.shape[0])
                    freq_to_copy = min(expected_shape[1], spec.shape[1])
                    fixed_spec[:time_to_copy, :freq_to_copy] = spec[:time_to_copy, :freq_to_copy]
                    batch_specs[j] = fixed_spec

            # ✅ ТЕПЕРЬ БЕЗОПАСНО создаем массив
            try:
                batch_array = np.array(batch_specs)
                print(f"   ✅ Батч {i // batch_size + 1}: {batch_array.shape}")
                all_specs.append(batch_array)
            except Exception as e:
                print(f"❌ Ошибка создания массива батча: {e}")
                print(f"   Формы в батче: {[spec.shape for spec in batch_specs]}")
                # Создаем массив вручную
                batch_array = np.zeros((len(batch_specs), TARGET_TIME, TARGET_MELS), dtype=np.float32)
                for j, spec in enumerate(batch_specs):
                    batch_array[j] = spec[:TARGET_TIME, :TARGET_MELS]
                all_specs.append(batch_array)

        del batch_specs
        gc.collect()

    if failed_files:
        print(f"⚠️ Всего не найдено файлов: {len(failed_files)}")

    # ✅ Финальное объединение
    if all_specs:
        result = np.vstack(all_specs)
        print(f"✅ Спектрограммы: {result.shape} (все [{TARGET_TIME}, {TARGET_MELS}])")
        return result
    else:
        print("❌ Нет спектрограмм для обработки")
        return np.zeros((0, TARGET_TIME, TARGET_MELS), dtype=np.float32)


# =============================================================================
# 🎯 ГЛАВНЫЙ АНАЛИЗ
# =============================================================================
def analyze_emotions():
    print("🚀 Анализ эмоций из WAV файлов...")

    # 1. Загрузка метаданных
    df = load_jsonl_sampled(JSONL_PATH)
    print("📊 Эмоции:", df['emotion'].value_counts().to_dict())

    # 2. ✅ БЕЗОПАСНАЯ загрузка спектрограмм
    X = load_spectrograms_from_wav(df, WAV_BASE_PATHS)
    y = df['emotion'].values[:len(X)]

    if len(X) == 0:
        print("❌ Нет данных для анализа!")
        return

    # 3. Проверка формы
    print(f"✅ Готово: X.shape={X.shape}, y.shape={y.shape}")
    print(f"   Уникальные формы в X: {np.unique([x.shape for x in X], axis=0)}")

    # =============================================================================
    # 📊 ВИЗУАЛИЗАЦИИ
    # =============================================================================

    # 1. Усреднённые спектрограммы
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    emotions = sorted(pd.unique(y))[:4]

    for i, emotion in enumerate(emotions):
        mask = y == emotion
        if mask.sum() > 0:
            mean_spec = X[mask].mean(axis=0)
            im = axes[i].imshow(mean_spec, aspect='auto', cmap='viridis', origin='lower')
            axes[i].set_title(f'{emotion}\n({mask.sum()})', fontweight='bold')
            plt.colorbar(im, ax=axes[i])
        else:
            axes[i].text(0.5, 0.5, f'Нет данных\nдля {emotion}',
                         ha='center', va='center', fontsize=12)
            axes[i].set_title(emotion)

    plt.suptitle('🧠 Усреднённые мел-спектрограммы', y=0.98)
    plt.tight_layout()
    plt.savefig('mean_spectrograms.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Анализ энергии
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(emotions)))
    for i, emotion in enumerate(emotions):
        mask = y == emotion
        if mask.sum() > 0:
            energy = X[mask].sum(axis=1).mean(axis=0)
            ax1.plot(energy, color=colors[i], linewidth=3, label=f'{emotion} ({mask.sum()})')

            # Высокие частоты
            total_energy = X[y == emotion].sum()
            if total_energy > 0:
                high_ratios = X[y == emotion, :, 64:].sum() / total_energy
            else:
                high_ratios = 0
            ax2.bar(emotion, high_ratios, color=colors[i], alpha=0.8)

    ax1.legend()
    ax1.set_title('📈 Энергия по частотным бинам')
    ax2.set_title('🎵 Высокие частоты')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('energy_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. PCA + t-SNE
    print("🔍 PCA/t-SNE...")
    subset_size = min(TSNE_SAMPLES, len(X))
    subset_idx = np.random.choice(len(X), subset_size, replace=False)

    X_subset = X[subset_idx].reshape(subset_size, -1)
    y_subset = y[subset_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=min(15, subset_size // 3), random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for i, emotion in enumerate(emotions):
        mask = y_subset == emotion
        if mask.sum() > 0:
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'{emotion} ({mask.sum()})', s=60, alpha=0.7)
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'{emotion} ({mask.sum()})', s=60, alpha=0.7)

    ax1.legend()
    ax1.set_title(f'PCA ({pca.explained_variance_ratio_.sum():.1%})')
    ax2.set_title('t-SNE')

    plt.tight_layout()
    plt.savefig('pca_tsne_emotions.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("🎉 АНАЛИЗ ЗАВЕРШЁН!")


# =============================================================================
# 🚀 ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    if not JSONL_PATH.exists():
        print(f"❌ JSONL не найден: {JSONL_PATH}")
    else:
        analyze_emotions()