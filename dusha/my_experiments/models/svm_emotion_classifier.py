#!/usr/bin/env python3
# svm_emotion_classifier_fixed.py
"""
🎯 Классификация эмоций с помощью SVM (ИСПРАВЛЕНА ОШИБКА ФОРМЫ)
"""

from choose_dataset import *


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

# =============================================================================
# ⚡ ПАРАМЕТРЫ
# =============================================================================
MAX_SAMPLES = 5000  # Уменьшим для теста
TARGET_TIME = 100
TARGET_MELS = 128
SAMPLE_RATE = 16000
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_SIZE = TARGET_TIME * TARGET_MELS  # 100 * 128 = 12800

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
# 🔧 УТИЛИТЫ
# =============================================================================
def load_jsonl_sampled(jsonl_path, max_samples=MAX_SAMPLES):
    """Загрузка данных из JSONL"""
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
    """Извлечение и нормализация мел-спектрограммы"""
    try:
        # Загрузка аудио
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)

        # Mel-спектрограмма
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=target_mels,
            n_fft=1024, hop_length=256
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)

        # ✅ ГАРАНТИРУЕМ форму [target_time, target_mels]
        current_time, current_mels = mel_db.shape

        # Создаем результат правильной формы
        result = np.zeros((target_time, target_mels), dtype=np.float32)

        # Копируем данные
        time_to_copy = min(target_time, current_time)
        mels_to_copy = min(target_mels, current_mels)
        result[:time_to_copy, :mels_to_copy] = mel_db[:time_to_copy, :mels_to_copy]

        return result.flatten()  # ✅ Теперь всегда 12800 элементов

    except Exception as e:
        print(f"⚠️ Ошибка {wav_path}: {e}")
        # Возвращаем нулевой вектор правильного размера
        return np.zeros(FEATURE_SIZE, dtype=np.float32)


def load_features(df, wav_base_paths, verbose=True):
    """Загрузка признаков из аудиофайлов"""
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

        # Поиск WAV файла
        wav_file = None
        for wav_base in wav_base_paths.values():
            # Пробуем разные варианты имени файла
            possible_names = [
                filename,
                f"{filename}.wav",
                f"{filename}.WAV",
                str(filename).replace('.wav', ''),
                str(filename).replace('.WAV', '')
            ]

            for name in possible_names:
                possible_path = wav_base / name
                if possible_path.exists():
                    wav_file = possible_path
                    break
                # Пробуем добавить расширение .wav
                possible_path = wav_base / f"{name}.wav"
                if possible_path.exists():
                    wav_file = possible_path
                    break

            if wav_file:
                break

        if wav_file:
            features = extract_and_normalize_mel(wav_file)

            # ✅ ПРОВЕРКА размера вектора
            if len(features) != FEATURE_SIZE:
                if verbose:
                    print(f"⚠️ Исправляем размер {len(features)} -> {FEATURE_SIZE} для {filename}")
                # Создаем вектор правильного размера
                fixed_features = np.zeros(FEATURE_SIZE, dtype=np.float32)
                size_to_copy = min(len(features), FEATURE_SIZE)
                fixed_features[:size_to_copy] = features[:size_to_copy]
                features = fixed_features

            X.append(features)
            y.append(emotion)
            found_files += 1
        else:
            failed_files.append(filename)
            # Добавляем нулевой вектор для поддержания размера
            X.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
            y.append(emotion)

    if verbose:
        print(f"✅ Найдено файлов: {found_files}")
        if failed_files:
            print(f"⚠️ Не найдено файлов: {len(failed_files)}")
        if X:
            # Проверяем размерности
            print(f"📏 Размер первого вектора: {len(X[0])}")
            print(f"📏 Ожидаемый размер: {FEATURE_SIZE}")

    # ✅ Создаем массивы с явным указанием размера
    X_array = np.zeros((len(X), FEATURE_SIZE), dtype=np.float32)
    for i, features in enumerate(X):
        X_array[i] = features[:FEATURE_SIZE]

    return X_array, np.array(y)


# =============================================================================
# 🎯 ОБУЧЕНИЕ SVM
# =============================================================================
def train_svm():
    print("🚀 Запуск обучения SVM...")

    # 1. Загрузка данных
    df = load_jsonl_sampled(JSONL_PATH, MAX_SAMPLES)
    print("\n📊 Распределение эмоций:")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)

    # Оставляем только эмоции с достаточным количеством примеров
    min_samples = 10
    valid_emotions = emotion_counts[emotion_counts >= min_samples].index
    df = df[df['emotion'].isin(valid_emotions)]
    print(f"\n📈 Используем {len(valid_emotions)} эмоций с ≥{min_samples} примерами")

    # 2. Извлечение признаков
    print("\n🎵 Извлечение признаков...")
    X, y = load_features(df, WAV_BASE_PATHS)

    if len(X) == 0:
        print("❌ Нет данных для обучения!")
        return

    print(f"\n✅ Признаки: X.shape={X.shape}, y.shape={y.shape}")
    print(f"   Размерность каждого примера: {X.shape[1]}")

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

    # 5. Масштабирование признаков
    print("\n📊 Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. PCA для уменьшения размерности (опционально, для ускорения)
    use_pca = True
    if use_pca:
        print("\n🔍 Применение PCA...")
        # Сохраняем 95% дисперсии
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"   Уменьшено: {X.shape[1]} -> {X_train_scaled.shape[1]} признаков")
        print(f"   Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.3f}")

    # 7. Обучение SVM
    print("\n🤖 Обучение SVM...")

    # Начинаем с простой конфигурации
    svm = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,  # Для вероятностей
        random_state=RANDOM_STATE,
        verbose=True
    )

    svm.fit(X_train_scaled, y_train)

    # 8. Оценка модели
    print("\n📊 Оценка модели...")
    y_pred = svm.predict(X_test_scaled)
    y_pred_proba = svm.predict_proba(X_test_scaled)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 60}")
    print("📊 РЕЗУЛЬТАТЫ SVM:")
    print(f"{'=' * 60}")
    print(f"✅ Точность: {accuracy:.4f}")

    print("\n📈 Отчёт по классификации:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))

    # 9. Матрица ошибок
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Матрица ошибок
    im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_title(f'Матрица ошибок SVM\nТочность: {accuracy:.3f}',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(le.classes_)))
    axes[0].set_yticks(range(len(le.classes_)))
    axes[0].set_xticklabels(le.classes_, rotation=45, ha='right')
    axes[0].set_yticklabels(le.classes_)
    axes[0].set_xlabel('Предсказанный класс')
    axes[0].set_ylabel('Истинный класс')

    # Добавляем значения в ячейки
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, cm[i, j], ha='center', va='center',
                         color=color, fontweight='bold')

    plt.colorbar(im, ax=axes[0])

    # Количество опорных векторов по классам
    if hasattr(svm, 'n_support_'):
        support_counts = svm.n_support_
        axes[1].bar(range(len(le.classes_)), support_counts,
                    color='skyblue', alpha=0.7)
        axes[1].set_title('Количество опорных векторов по классам',
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Классы')
        axes[1].set_ylabel('Количество SV')
        axes[1].set_xticks(range(len(le.classes_)))
        axes[1].set_xticklabels(le.classes_, rotation=45)

        # Добавляем значения на столбцы
        for i, count in enumerate(support_counts):
            axes[1].text(i, count + max(support_counts) * 0.01, str(count),
                         ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('svm_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. Анализ точности по классам
    print("\n📊 Анализ точности по классам:")
    class_accuracies = {}
    for i, emotion in enumerate(le.classes_):
        mask = y_test_labels == emotion
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_accuracies[emotion] = class_acc
            print(f"   {emotion}: {class_acc:.4f} ({mask.sum()} примеров)")

    # 11. GridSearch для оптимизации гиперпараметров
    perform_gridsearch = True
    if perform_gridsearch and X_train.shape[0] < 500:
        print("\n🔍 Запуск GridSearchCV для оптимизации гиперпараметров...")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs= 18
        )

        grid_search.fit(X_train_scaled, y_train)

        print(f"\n🎯 Лучшие параметры: {grid_search.best_params_}")
        print(f"🎯 Лучшая точность (кросс-валидация): {grid_search.best_score_:.4f}")

        # Обучаем модель с лучшими параметрами
        best_svm = grid_search.best_estimator_
        best_svm.fit(X_train_scaled, y_train)
        y_pred_best = best_svm.predict(X_test_scaled)
        best_accuracy = accuracy_score(y_test, y_pred_best)
        print(f"🎯 Точность на тесте (лучшая модель): {best_accuracy:.4f}")

        # Используем лучшую модель
        svm = best_svm
        accuracy = best_accuracy

    # 12. Сохранение модели
    print("\n💾 Сохранение модели...")
    model_data = {
        'model': svm,
        'scaler': scaler,
        'label_encoder': le,
        'use_pca': use_pca,
        'pca': pca if use_pca else None,
        'feature_size': FEATURE_SIZE,
        'classes': le.classes_.tolist(),
        'accuracy': accuracy
    }

    joblib.dump(model_data, MODEL_PATH / 'svm_model.pkl')
    print(f"✅ Модель сохранена: {MODEL_PATH / 'svm_model.pkl'}")

    # 13. Создание отчета
    print("\n📋 Сводный отчет:")
    print(f"{'=' * 60}")
    print(f"Модель: SVM")
    print(f"Ядро: {svm.kernel}")
    print(f"Параметр C: {svm.C}")
    print(f"Количество классов: {len(le.classes_)}")
    print(f"Итоговая точность: {accuracy:.4f}")
    print(f"Размерность признаков: {X.shape[1]} -> {X_train_scaled.shape[1]}")
    print(f"{'=' * 60}")

    print("\n🎉 Обучение SVM успешно завершено!")
    return svm, scaler, le, accuracy




# =============================================================================
# 🚀 ЗАПУСК
# =============================================================================
if __name__ == "__main__":
    if not JSONL_PATH.exists():
        print(f"❌ JSONL не найден: {JSONL_PATH}")
    else:
        train_svm()