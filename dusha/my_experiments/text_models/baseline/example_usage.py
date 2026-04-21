"""
Пример использования модели Embeddings_LogReg для предсказания эмоций.

Этот скрипт показывает, как:
1. Загрузить обученную модель
2. Загрузить FastText embeddings
3. Предсказать эмоцию для произвольного текста
"""

import sys
from pathlib import Path
import numpy as np

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent))

# Импортируем функции из основного скрипта
try:
    from Embeddings_LogReg import (
        load_fasttext_model,
        text_to_vector,
        preprocess_text,
        load_model,
        get_dataset_name,
        DATASET_PATH,
        DEFAULT_EMBEDDINGS_PATH
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    IMPORTS_OK = False


def predict_emotion(text, model, scaler, fasttext_model):
    """
    Предсказывает эмоцию для заданного текста.
    
    Args:
        text: Исходный текст
        model: Обученная модель LogisticRegression
        scaler: Обученный StandardScaler
        fasttext_model: Модель FastText
        
    Returns:
        emotion: Предсказанная эмоция
        probabilities: Вероятности для каждого класса
    """
    # Предобработка текста
    text_clean = preprocess_text(text)
    
    # Преобразование в вектор
    vector = text_to_vector(text_clean, fasttext_model)
    
    # Нормализация
    vector_scaled = scaler.transform(vector.reshape(1, -1))
    
    # Предсказание
    emotion = model.predict(vector_scaled)[0]
    probabilities = model.predict_proba(vector_scaled)[0]
    
    return emotion, probabilities


def main():
    """Пример использования модели для предсказания"""
    
    if not IMPORTS_OK:
        print("Не удалось импортировать необходимые функции")
        return
    
    print("="*60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДЕЛИ EMBEDDINGS_LOGREG")
    print("="*60)
    
    # Путь к embeddings (можно изменить)
    embeddings_path = DEFAULT_EMBEDDINGS_PATH
    
    # Проверка существования embeddings
    if not Path(embeddings_path).exists():
        print(f"\n⚠ Файл embeddings не найден: {embeddings_path}")
        print("\nСкачайте предобученные embeddings:")
        print("wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz")
        print("gunzip cc.ru.300.bin.gz")
        print(f"mkdir -p {Path(embeddings_path).parent}")
        print(f"mv cc.ru.300.bin {embeddings_path}")
        return
    
    # Загрузка FastText модели
    print("\n1. Загрузка FastText embeddings...")
    try:
        fasttext_model = load_fasttext_model(embeddings_path)
    except Exception as e:
        print(f"Ошибка загрузки embeddings: {e}")
        return
    
    # Загрузка обученной модели
    print("\n2. Загрузка обученной модели...")
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    dataset_name = get_dataset_name(train_manifest)
    
    try:
        model, scaler = load_model(dataset_name)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nСначала обучите модель:")
        print("poetry run python Embeddings_LogReg.py --mode train")
        return
    
    # Примеры текстов для предсказания
    test_texts = [
        "я очень злой и раздраженный",
        "мне грустно и печально",
        "все хорошо, нормально",
        "я счастлив и радуюсь жизни",
        "какой же ужасный день",
        "спасибо большое",
    ]
    
    print("\n3. Предсказание эмоций для примеров:")
    print("="*60)
    
    # Предсказание для каждого текста
    for i, text in enumerate(test_texts, 1):
        emotion, probs = predict_emotion(text, model, scaler, fasttext_model)
        
        print(f"\nПример {i}:")
        print(f"  Текст: '{text}'")
        print(f"  Предсказание: {emotion}")
        print(f"  Вероятности:")
        for label, prob in zip(model.classes_, probs):
            print(f"    {label:8s}: {prob:.3f}")
    
    print("\n" + "="*60)
    print("Готово!")
    print("="*60)


if __name__ == "__main__":
    main()
