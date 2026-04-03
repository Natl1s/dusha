#!/usr/bin/env python3
# random_forest_emotion_classifier.py
"""
🎯 Классификация эмоций с помощью Random Forest
"""

from choose_dataset import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from sklearn.ensemble import RandomForestClassifier
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
# 🔧 УТИЛИТЫ (идентичные другим моделям)
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
# 🎯 ОБУЧЕНИЕ RANDOM FOREST
# =============================================================================
def train_random_forest():
    print("🚀 Запуск обучения Random Forest...")

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

    # 5. Random Forest обычно не требует масштабирования, но сделаем для единообразия
    print("\n📊 Подготовка данных...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. PCA не обязателен для Random Forest, но можно использовать для уменьшения размера
    use_pca = False  # Random Forest хорошо работает с большим количеством признаков
    if use_pca:
        print("\n🔍 Применение PCA...")
        pca = PCA(n_components=100, random_state=RANDOM_STATE)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"   Уменьшено: {X.shape[1]} -> {X_train_scaled.shape[1]} признаков")

    # 7. Обучение Random Forest
    print("\n🌳 Обучение Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs= 18,
        verbose=1
    )

    rf.fit(X_train_scaled, y_train)

    # 8. Оценка
    print("\n📊 Оценка модели...")
    y_pred = rf.predict(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print("📊 РЕЗУЛЬТАТЫ RANDOM FOREST:")
    print(f"{'=' * 60}")
    print(f"✅ Точность: {accuracy:.4f}")

    print("\n📈 Отчёт по классификации:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))

    # 9. Визуализация
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Матрица ошибок
    im = axes[0].imshow(cm, cmap='Oranges', aspect='auto')
    axes[0].set_title(f'Random Forest\nТочность: {accuracy:.3f}',
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

    # Важность признаков
    feature_importance = rf.feature_importances_

    # Топ-20 важных признаков
    top_n = min(20, len(feature_importance))
    top_indices = np.argsort(feature_importance)[-top_n:]
    top_importance = feature_importance[top_indices]

    axes[1].barh(range(top_n), top_importance, color='orange', alpha=0.7)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([f'Призн.{i}' for i in top_indices])
    axes[1].set_title(f'Топ-{top_n} важных признаков', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Важность')

    # Важность по времени (усредненная)
    if feature_importance.shape[0] == FEATURE_SIZE:
        # Преобразуем к форме спектрограммы
        importance_2d = feature_importance.reshape(TARGET_TIME, TARGET_MELS)
        time_importance = importance_2d.mean(axis=1)

        axes[2].plot(time_importance, linewidth=3, color='darkorange')
        axes[2].set_title('Важность по временным срезам', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Временной кадр (0-100)')
        axes[2].set_ylabel('Средняя важность')
        axes[2].grid(True, alpha=0.3)

        # Отмечаем наиболее важные временные точки
        top_time_indices = np.argsort(time_importance)[-3:]
        for idx in top_time_indices:
            axes[2].axvline(x=idx, color='red', linestyle='--', alpha=0.5)
            axes[2].text(idx, time_importance[idx] * 1.05, f'{idx}',
                         ha='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. GridSearch
    perform_gridsearch = True
    if perform_gridsearch:
        print("\n🔍 GridSearchCV...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs= 18),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs= 18
        )

        grid_search.fit(X_train_scaled, y_train)
        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучшая точность: {grid_search.best_score_:.4f}")

        # Используем лучшую модель
        best_rf = grid_search.best_estimator_
        best_rf.fit(X_train_scaled, y_train)
        y_pred_best = best_rf.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test, y_pred_best)
        print(f"Точность лучшей модели: {best_accuracy:.4f}")

        # Обновляем модель
        rf = best_rf
        accuracy = best_accuracy

    # 11. Сохранение модели
    print("\n💾 Сохранение модели...")
    model_data = {
        'model': rf,
        'scaler': scaler,
        'label_encoder': le,
        'use_pca': use_pca,
        'pca': pca if use_pca else None,
        'feature_size': FEATURE_SIZE,
        'classes': le.classes_.tolist(),
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'test_data': {'X_test': X_test, 'y_test': y_test}
    }

    joblib.dump(model_data, MODEL_PATH / 'random_forest_model.pkl')
    print(f"✅ Модель сохранена: {MODEL_PATH / 'random_forest_model.pkl'}")

    # 12. Анализ важности по классам
    print("\n🔍 Анализ важности признаков по классам:")

    # Собираем информацию о важности признаков
    if hasattr(rf, 'feature_importances_'):
        top_10_indices = np.argsort(feature_importance)[-10:]
        print("\nТоп-10 наиболее важных признаков:")
        for idx in reversed(top_10_indices):
            print(f"   Признак {idx}: {feature_importance[idx]:.6f}")

    print("\n🎉 Обучение Random Forest завершено!")
    return model_data


# =============================================================================
# 🚀 ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    if not JSONL_PATH.exists():
        print(f"❌ JSONL не найден: {JSONL_PATH}")
    else:
        train_random_forest()