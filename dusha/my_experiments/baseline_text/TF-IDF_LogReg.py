"""
TF-IDF + Logistic Regression для классификации эмоций по тексту

Скрипт обучает модель классификации эмоций на основе текстовых признаков:
- Загрузка текстов из JSONL файлов (колонка speaker_text)
- Предобработка: lowercase, удаление лишних пробелов
- Построение TF-IDF признаков (word-level и n-граммы)
- Обучение LogisticRegression с балансировкой классов
- Сохранение модели и векторизатора
- Вывод метрик качества

Использование:
    poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode train
    poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode load
    poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode auto
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import joblib
import argparse
from datetime import datetime
import re

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']

# Путь для сохранения моделей
MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem  # TF-IDF_LogReg


def save_model(model, vectorizer, dataset_name, model_name=MODEL_NAME):
    """Сохраняет модель и векторизатор в файл"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    vectorizer_path = MODELS_DIR / f"{full_model_name}_vectorizer.pkl"
    model_path_timestamped = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pkl"
    
    # Сохранение основных файлов
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Сохранение с временной меткой (для истории)
    joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path_timestamped)
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'='*60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Векторизатор: {vectorizer_path.absolute()}")
    print(f"✓ Бэкап:  {model_path_timestamped.absolute()}")
    print(f"{'='*60}")


def load_model(dataset_name, model_name=MODEL_NAME):
    """Загружает модель и векторизатор из файла"""
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    vectorizer_path = MODELS_DIR / f"{full_model_name}_vectorizer.pkl"
    
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена! Проверьте наличие файлов:\n"
            f"  {model_path}\n"
            f"  {vectorizer_path}"
        )
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print(f"✓ Модель загружена из {model_path}")
    print(f"✓ Векторизатор загружен из {vectorizer_path}")
    
    return model, vectorizer


def model_exists(dataset_name, model_name=MODEL_NAME):
    """Проверяет существование сохраненной модели"""
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    vectorizer_path = MODELS_DIR / f"{full_model_name}_vectorizer.pkl"
    return model_path.exists() and vectorizer_path.exists()


def get_dataset_name(train_manifest_path):
    """Извлекает имя датасета из пути к манифесту"""
    # Берём имя файла без расширения
    return Path(train_manifest_path).stem


def preprocess_text(text):
    """
    Предобработка текста:
    1. Приведение к нижнему регистру (lowercase)
    2. Удаление лишних пробелов
    
    Args:
        text: Исходный текст
        
    Returns:
        Обработанный текст
    """
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление лишних пробелов: заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    
    # Удаление пробелов в начале и конце строки
    text = text.strip()
    
    return text


def load_texts_from_manifest(manifest_path):
    """
    Загрузка текстов из JSONL манифеста.
    
    Читает JSON файл построчно и извлекает:
    - speaker_text: текст для обучения
    - emotion: метка класса
    
    Args:
        manifest_path: Путь к .jsonl файлу
        
    Returns:
        texts: Список текстов
        labels: Список меток эмоций
    """
    texts = []
    labels = []

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Извлекаем текст из поля speaker_text
                text = data.get('speaker_text', '')
                
                # Предобработка текста
                text = preprocess_text(text)
                
                # Пропускаем пустые тексты
                if not text:
                    continue
                
                texts.append(text)
                labels.append(data['emotion'])
                
            except json.JSONDecodeError as e:
                print(f"⚠ Ошибка парсинга JSON в строке {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"⚠ Отсутствует ключ {e} в строке {line_num}")
                continue

    return texts, np.array(labels)


def evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test):
    """
    Оценка модели на обучающей и тестовой выборках.
    
    Args:
        model: Обученная модель LogisticRegression
        vectorizer: Обученный TF-IDF векторизатор
        X_train_texts: Тексты обучающей выборки
        y_train: Метки обучающей выборки
        X_test_texts: Тексты тестовой выборки
        y_test: Метки тестовой выборки
    """
    # Преобразование текстов в TF-IDF признаки
    X_train_tfidf = vectorizer.transform(X_train_texts)
    X_test_tfidf = vectorizer.transform(X_test_texts)
    
    # Параметры модели
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Количество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"Размер матрицы коэффициентов: {model.coef_.shape}")
    print(f"Количество итераций: {model.n_iter_}")
    
    # Параметры векторизатора
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ TF-IDF ВЕКТОРИЗАТОРА")
    print(f"{'='*60}")
    print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
    print(f"Диапазон n-грамм: {vectorizer.ngram_range}")
    print(f"Максимальное количество признаков: {vectorizer.max_features}")
    print(f"Минимальная частота документов (min_df): {vectorizer.min_df}")
    print(f"Максимальная частота документов (max_df): {vectorizer.max_df}")
    
    # Оценка на обучающей выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    train_pred = model.predict(X_train_tfidf)
    print(classification_report(y_train, train_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))
    
    # Оценка на тестовой выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    test_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, test_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, test_pred))


def train_tfidf_logreg(save=True):
    """
    Обучение модели классификации эмоций на основе TF-IDF признаков и Логистической регрессии.
    
    Этапы:
    1. Загрузка текстов из JSONL файлов
    2. Предобработка текстов (lowercase, удаление пробелов)
    3. Построение TF-IDF признаков (word-level и n-граммы)
    4. Обучение LogisticRegression
    5. Оценка качества
    6. Сохранение модели
    
    Args:
        save: Сохранять ли модель после обучения
        
    Returns:
        model: Обученная модель
        vectorizer: Обученный векторизатор
        dataset_name: Имя датасета
    """
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test.jsonl'

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    # Загрузка обучающих данных
    print("Загрузка обучающих данных...")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")
    print(f"Пример текста после предобработки: '{X_train_texts[0]}'")

    # Загрузка тестовых данных
    print("\nЗагрузка тестовых данных...")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    # Построение TF-IDF признаков
    print(f"\n{'='*60}")
    print("ПОСТРОЕНИЕ TF-IDF ПРИЗНАКОВ")
    print(f"{'='*60}")
    
    # Создание TF-IDF векторизатора
    # ngram_range=(1, 2) - используем как отдельные слова (1-граммы), так и биграммы (2-граммы)
    # max_features=10000 - ограничиваем размер словаря топ-10000 признаками
    # min_df=2 - слово должно встречаться минимум в 2 документах
    # max_df=0.8 - слово не должно встречаться более чем в 80% документов (фильтр стоп-слов)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # word-level (1-граммы) + биграммы (2-граммы)
        max_features=10000,      # максимальное количество признаков
        min_df=2,                # минимальная частота документа
        max_df=0.8,              # максимальная частота документа
        sublinear_tf=True        # применяем сублинейное масштабирование TF (1 + log(tf))
    )
    
    # Обучение векторизатора и преобразование обучающих текстов
    X_train_tfidf = vectorizer.fit_transform(X_train_texts)
    print(f"✓ Размер матрицы TF-IDF признаков (train): {X_train_tfidf.shape}")
    print(f"  - Количество документов: {X_train_tfidf.shape[0]}")
    print(f"  - Количество признаков (размер словаря): {X_train_tfidf.shape[1]}")
    
    # Преобразование тестовых текстов (без обучения векторизатора)
    X_test_tfidf = vectorizer.transform(X_test_texts)
    print(f"✓ Размер матрицы TF-IDF признаков (test): {X_test_tfidf.shape}")

    # Обучение логистической регрессии
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ МОДЕЛИ LOGISTIC REGRESSION")
    print(f"{'='*60}")
    
    # Создание модели логистической регрессии
    # solver='lbfgs' - оптимизатор Limited-memory BFGS
    # max_iter=1000 - максимальное количество итераций
    # random_state=42 - фиксация случайности для воспроизводимости
    # class_weight='balanced' - автоматическая балансировка классов
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # балансировка классов
    )
    
    # Обучение модели
    model.fit(X_train_tfidf, y_train)
    print("✓ Обучение завершено!")

    # Оценка модели
    evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test)
    
    # Сохранение модели
    if save:
        save_model(model, vectorizer, dataset_name)

    return model, vectorizer, dataset_name


def load_and_evaluate():
    """
    Загружает существующую модель и оценивает её на данных.
    
    Returns:
        model: Загруженная модель
        vectorizer: Загруженный векторизатор
    """
    print(f"{'='*60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'='*60}")
    
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test.jsonl'
    
    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")
    
    # Загрузка модели
    model, vectorizer = load_model(dataset_name)
    
    # Загрузка обучающих данных
    print("\nЗагрузка обучающих данных...")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    
    # Загрузка тестовых данных
    print("\nЗагрузка тестовых данных...")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    
    # Оценка модели
    evaluate_model(model, vectorizer, X_train_texts, y_train, X_test_texts, y_test)

    return model, vectorizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Обучение или загрузка модели TF-IDF + LogReg для классификации эмоций по тексту'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'load', 'auto'],
        default='auto',
        help='Режим работы: train - обучить новую модель, load - загрузить существующую, auto - загрузить если есть, иначе обучить'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Не сохранять модель после обучения'
    )
    
    args = parser.parse_args()
    
    # Получение имени датасета для проверки существования модели
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train.jsonl'
    dataset_name = get_dataset_name(train_manifest)
    
    # Определение режима работы
    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        model, vectorizer, _ = train_tfidf_logreg(save=not args.no_save)
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        model, vectorizer = load_and_evaluate()
    else:  # auto
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            model, vectorizer = load_and_evaluate()
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            model, vectorizer, _ = train_tfidf_logreg(save=not args.no_save)
