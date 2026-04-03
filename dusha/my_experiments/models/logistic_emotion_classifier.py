#!/usr/bin/env python3
# logistic_emotion_classifier.py
"""
🎯 Классификация эмоций с помощью логистической регрессии
"""
from choose_dataset import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

# =============================================================================
# ⚡ ОБЩИЕ ПАРАМЕТРЫ
# =============================================================================
MAX_SAMPLES = 5000
TARGET_TIME = 100
TARGET_MELS = 128
SAMPLE_RATE = 16000
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_SIZE = TARGET_TIME * TARGET_MELS

# =============================================================================
# 📁 ПУТИ
# =============================================================================
BASE_PATH = Path('/home/natlis/PycharmProjects/dusha/dusha/data_processing/dataset')
JSONL_PATH = BASE_PATH / 'processed_dataset_090' / 'train' / DATASET_NAME
WAV_BASE_PATHS = {'crowd_train': BASE_PATH / 'crowd_train' / 'wavs'}
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(exist_ok=True)

print(f"📁 JSONL: {JSONL_PATH}")


# =============================================================================
# 🔧 УТИЛИТЫ (идентичные SVM)
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


def extract_and_normalize_mel(wav_path, target_time=TARGET_TIME, target_mels=TARGET_MELS, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=target_mels, n_fft=1024, hop_length=256
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        current_time, current_mels = mel_db.shape
        result = np.zeros((target_time, target_mels), dtype=np.float32)

        time_to_copy = min(target_time, current_time)
        mels_to_copy = min(target_mels, current_mels)
        result[:time_to_copy, :mels_to_copy] = mel_db[:time_to_copy, :mels_to_copy]

        return result.flatten()

    except Exception as e:
        print(f"⚠️ Ошибка {wav_path}: {e}")
        return np.zeros(FEATURE_SIZE, dtype=np.float32)


def load_features(df, wav_base_paths, verbose=True):
    X = []
    y = []
    failed_files = []
    found_files = 0

    for idx, row in df.iterrows():
        if verbose and idx % 100 == 0:
            print(f"🔄 Обработано {idx}/{len(df)} файлов...")

        filename = row.get('id', row.get('filename', ''))
        emotion = row.get('emotion', '')

        if not emotion:
            continue

        wav_file = None
        for wav_base in wav_base_paths.values():
            possible_paths = [
                wav_base / f"{filename}",
                wav_base / f"{filename}.wav",
                wav_base / f"{filename}.WAV"
            ]

            for path in possible_paths:
                if path.exists():
                    wav_file = path
                    break

            if wav_file:
                break

        if wav_file:
            features = extract_and_normalize_mel(wav_file)

            if len(features) != FEATURE_SIZE:
                if verbose:
                    print(f"⚠️ Исправляем размер {len(features)} -> {FEATURE_SIZE}")
                fixed_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
                size_to_copy = min(len(features), FEATURE_SIZE)
                fixed_features[:size_to_copy] = features[:size_to_copy]
                features = fixed_features

            X.append(features)
            y.append(emotion)
            found_files += 1
        else:
            failed_files.append(filename)
            X.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
            y.append(emotion)

    if verbose:
        print(f"✅ Найдено файлов: {found_files}")
        if failed_files:
            print(f"⚠️ Не найдено файлов: {len(failed_files)}")

    X_array = np.zeros((len(X), FEATURE_SIZE), dtype=np.float32)
    for i, features in enumerate(X):
        X_array[i] = features[:FEATURE_SIZE]

    return X_array, np.array(y), failed_files


# =============================================================================
# 🎯 ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
# =============================================================================
def train_logistic():
    print("🚀 Запуск обучения логистической регрессии...")

    # 1. Загрузка данных
    df = load_jsonl_sampled(JSONL_PATH, MAX_SAMPLES)
    print("\n📊 Распределение эмоций:")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)

    min_samples = 10
    valid_emotions = emotion_counts[emotion_counts >= min_samples].index
    df = df[df['emotion'].isin(valid_emotions)]
    print(f"\n📈 Используем {len(valid_emotions)} эмоций с ≥{min_samples} примерами")

    # 2. Извлечение признаков
    print("\n🎵 Извлечение признаков...")
    X, y, failed_files = load_features(df, WAV_BASE_PATHS)

    if len(X) == 0:
        print("❌ Нет данных для обучения!")
        return

    print(f"\n✅ Признаки: X.shape={X.shape}, y.shape={y.shape}")

    # 3. Кодирование меток
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"📊 Классы: {le.classes_}")

    # 4. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"\n📊 Разделение данных:")
    print(f"   Обучающая выборка: {X_train.shape[0]} примеров")
    print(f"   Тестовая выборка: {X_test.shape[0]} примеров")

    # 5. Масштабирование
    print("\n📊 Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. PCA
    use_pca = True
    if use_pca:
        print("\n🔍 Применение PCA...")
        pca = PCA(n_components=100, random_state=RANDOM_STATE)  # Фиксированное количество
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"   Уменьшено: {X.shape[1]} -> {X_train_scaled.shape[1]} признаков")

    # 7. Обучение логистической регрессии
    print("\n🤖 Обучение логистической регрессии...")

    logistic = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=RANDOM_STATE,
        verbose=1
    )

    logistic.fit(X_train_scaled, y_train)

    # 8. Оценка
    print("\n📊 Оценка модели...")
    y_pred = logistic.predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("📊 РЕЗУЛЬТАТЫ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ:")
    print(f"{'=' * 60}")
    print(f"✅ Точность: {accuracy:.4f}")

    print("\n📈 Отчёт по классификации:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))

    # 9. Визуализация
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Матрица ошибок
    im = axes[0].imshow(cm, cmap='Greens', aspect='auto')
    axes[0].set_title(f'Логистическая регрессия\nТочность: {accuracy:.3f}',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(le.classes_)))
    axes[0].set_yticks(range(len(le.classes_)))
    axes[0].set_xticklabels(le.classes_, rotation=45, ha='right')
    axes[0].set_yticklabels(le.classes_)

    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, cm[i, j], ha='center', va='center', color=color, fontweight='bold')

    plt.colorbar(im, ax=axes[0])

    # Коэффициенты модели
    if len(le.classes_) <= 5:  # Показываем только для небольшого числа классов
        coef_df = pd.DataFrame(
            logistic.coef_,
            columns=[f'Призн.{i}' for i in range(logistic.coef_.shape[1])],
            index=le.classes_
        )

        # Топ-10 признаков по важности
        mean_abs_coef = np.abs(logistic.coef_).mean(axis=0)
        top_indices = np.argsort(mean_abs_coef)[-10:]
        top_features = coef_df.iloc[:, top_indices]

        # Визуализация тепловой карты
        import seaborn as sns
        sns.heatmap(top_features, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                    ax=axes[1], cbar_kws={'label': 'Коэффициент'})
        axes[1].set_title('Топ-10 признаков по важности', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Признаки')
        axes[1].set_ylabel('Классы')

    plt.tight_layout()
    plt.savefig('logistic_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. GridSearch
    perform_gridsearch = True
    if perform_gridsearch:
        print("\n🔍 GridSearchCV...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [500, 1000]
        }

        grid_search = GridSearchCV(
            LogisticRegression(multi_class='multinomial', random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs= 18
        )

        grid_search.fit(X_train_scaled, y_train)
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучшая точность: {grid_search.best_score_:.4f}")

    # 11. Сохранение модели
    print("\n💾 Сохранение модели...")
    model_data = {
        'model': logistic,
        'scaler': scaler,
        'label_encoder': le,
        'use_pca': use_pca,
        'pca': pca if use_pca else None,
        'feature_size': FEATURE_SIZE,
        'classes': le.classes_.tolist(),
        'accuracy': accuracy,
        'test_data': {'X_test': X_test, 'y_test': y_test}
    }

    joblib.dump(model_data, MODEL_PATH / 'logistic_model.pkl')
    print(f"✅ Модель сохранена: {MODEL_PATH / 'logistic_model.pkl'}")

    print("\n🎉 Обучение логистической регрессии завершено!")
    return model_data


# =============================================================================
# 🚀 ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    if not JSONL_PATH.exists():
        print(f"❌ JSONL не найден: {JSONL_PATH}")
    else:
        train_logistic()