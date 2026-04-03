"""
FastText Embeddings + Logistic Regression для классификации эмоций по тексту

Скрипт обучает модель классификации эмоций на основе усредненных word embeddings:
- Загрузка текстов из JSONL файлов (колонка speaker_text)
- Предобработка: lowercase, удаление лишних пробелов
- Использование предобученных FastText embeddings (учитывают части слов)
- Преобразование текста в вектор через усреднение векторов слов
- Обучение LogisticRegression с балансировкой классов
- Сохранение модели
- Вывод метрик качества

Требования:
    pip install gensim
    Или: poetry add gensim
    
    Предобученная модель FastText для русского языка:
    Скачать: https://fasttext.cc/docs/en/crawl-vectors.html
    Например: cc.ru.300.bin или wiki.ru.bin

Использование:
    poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train
    poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode load
    poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode auto
    
    С указанием пути к embeddings:
    poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train --embeddings-path /path/to/cc.ru.300.bin
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import joblib
import argparse
from datetime import datetime
import re

# Проверка наличия gensim
try:
    from gensim.models import KeyedVectors
    from gensim.models.fasttext import load_facebook_model
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠ ВНИМАНИЕ: gensim не установлен!")
    print("Установите: pip install gensim")
    print("Или: poetry add gensim")

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']

# Путь для сохранения моделей
MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem  # Embeddings_LogReg

# Путь к предобученным embeddings по умолчанию
# Скачать можно здесь: https://fasttext.cc/docs/en/crawl-vectors.html
DEFAULT_EMBEDDINGS_PATH = Path.home() / "fasttext_models" / "cc.ru.300.bin"


def save_model(model, scaler, dataset_name, model_name=MODEL_NAME):
    """
    Сохраняет модель и scaler в файл.
    
    Args:
        model: Обученная модель LogisticRegression
        scaler: Обученный StandardScaler
        dataset_name: Имя датасета
        model_name: Название модели
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{full_model_name}_scaler.pkl"
    model_path_timestamped = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pkl"
    
    # Сохранение основных файлов
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Сохранение с временной меткой (для истории)
    joblib.dump({'model': model, 'scaler': scaler}, model_path_timestamped)
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'='*60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Scaler: {scaler_path.absolute()}")
    print(f"✓ Бэкап:  {model_path_timestamped.absolute()}")
    print(f"{'='*60}")


def load_model(dataset_name, model_name=MODEL_NAME):
    """
    Загружает модель и scaler из файла.
    
    Args:
        dataset_name: Имя датасета
        model_name: Название модели
        
    Returns:
        model: Загруженная модель
        scaler: Загруженный scaler
    """
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{full_model_name}_scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена! Проверьте наличие файлов:\n"
            f"  {model_path}\n"
            f"  {scaler_path}"
        )
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"✓ Модель загружена из {model_path}")
    print(f"✓ Scaler загружен из {scaler_path}")
    
    return model, scaler


def model_exists(dataset_name, model_name=MODEL_NAME):
    """
    Проверяет существование сохраненной модели.
    
    Args:
        dataset_name: Имя датасета
        model_name: Название модели
        
    Returns:
        bool: True если модель существует
    """
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{full_model_name}_scaler.pkl"
    return model_path.exists() and scaler_path.exists()


def get_dataset_name(train_manifest_path):
    """
    Извлекает имя датасета из пути к манифесту.
    
    Args:
        train_manifest_path: Путь к файлу манифеста
        
    Returns:
        str: Имя датасета (имя файла без расширения)
    """
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


def load_fasttext_model(embeddings_path):
    """
    Загружает предобученную модель FastText.
    
    FastText умеет работать с OOV (out-of-vocabulary) словами,
    разбивая их на n-граммы символов.
    
    Args:
        embeddings_path: Путь к файлу с предобученными embeddings (.bin)
        
    Returns:
        model: Загруженная модель FastText
        
    Raises:
        ImportError: Если gensim не установлен
        FileNotFoundError: Если файл embeddings не найден
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Библиотека gensim не установлена!\n"
            "Установите: pip install gensim\n"
            "Или: poetry add gensim"
        )
    
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Файл с embeddings не найден: {embeddings_path}\n\n"
            f"Скачайте предобученную модель FastText для русского языка:\n"
            f"  https://fasttext.cc/docs/en/crawl-vectors.html\n\n"
            f"Например:\n"
            f"  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz\n"
            f"  gunzip cc.ru.300.bin.gz\n"
            f"  mkdir -p ~/fasttext_models\n"
            f"  mv cc.ru.300.bin ~/fasttext_models/\n"
        )
    
    print(f"Загрузка FastText embeddings из {embeddings_path}...")
    print("⏳ Это может занять несколько минут...")
    
    # Загрузка FastText модели в формате .bin (Facebook format)
    model = load_facebook_model(str(embeddings_path))
    
    print(f"✓ Embeddings загружены!")
    print(f"  - Размерность вектора: {model.wv.vector_size}")
    print(f"  - Количество слов в словаре: {len(model.wv)}")
    
    return model


def text_to_vector(text, fasttext_model):
    """
    Преобразует текст в вектор фиксированной длины.
    
    Процесс:
    1. Разбиваем текст на слова
    2. Каждое слово преобразуем в вектор через FastText
    3. Усредняем все векторы слов
    
    FastText преимущество: если слова нет в словаре (OOV),
    модель все равно создаст вектор на основе n-грамм символов.
    
    Args:
        text: Исходный текст
        fasttext_model: Модель FastText
        
    Returns:
        np.ndarray: Вектор фиксированной длины (размерность = vector_size)
    """
    # Разбиваем текст на слова
    words = text.split()
    
    # Если текст пустой, возвращаем нулевой вектор
    if not words:
        return np.zeros(fasttext_model.wv.vector_size, dtype=np.float32)
    
    # Получаем векторы для всех слов
    word_vectors = []
    for word in words:
        # FastText умеет работать с OOV словами через n-граммы
        try:
            vector = fasttext_model.wv[word]
            word_vectors.append(vector)
        except KeyError:
            # Если слово не найдено (не должно произойти с FastText, но на всякий случай)
            continue
    
    # Если не удалось получить ни одного вектора, возвращаем нулевой
    if not word_vectors:
        return np.zeros(fasttext_model.wv.vector_size, dtype=np.float32)
    
    # Усредняем все векторы слов
    # Это простой, но эффективный способ получить представление текста
    text_vector = np.mean(word_vectors, axis=0).astype(np.float32)
    
    return text_vector


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
        labels: Массив меток эмоций
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


def texts_to_vectors(texts, fasttext_model, verbose=True):
    """
    Преобразует список текстов в матрицу векторов.
    
    Args:
        texts: Список текстов
        fasttext_model: Модель FastText
        verbose: Выводить ли прогресс
        
    Returns:
        np.ndarray: Матрица векторов размером (n_samples, vector_size)
    """
    vectors = []
    
    if verbose:
        print(f"Преобразование {len(texts)} текстов в векторы...")
        from tqdm import tqdm
        texts_iter = tqdm(texts, desc="Векторизация")
    else:
        texts_iter = texts
    
    for text in texts_iter:
        vector = text_to_vector(text, fasttext_model)
        vectors.append(vector)
    
    # Преобразуем список векторов в numpy массив
    vectors_matrix = np.vstack(vectors)
    
    if verbose:
        print(f"✓ Создана матрица векторов: {vectors_matrix.shape}")
    
    return vectors_matrix


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """
    Оценка модели на обучающей и тестовой выборках.
    
    Args:
        model: Обученная модель LogisticRegression
        scaler: Обученный StandardScaler
        X_train: Векторы обучающей выборки
        y_train: Метки обучающей выборки
        X_test: Векторы тестовой выборки
        y_test: Метки тестовой выборки
    """
    # Нормализация признаков
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Параметры модели
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Количество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    print(f"Размер матрицы коэффициентов: {model.coef_.shape}")
    print(f"Количество итераций: {model.n_iter_}")
    
    # Информация о признаках
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ПРИЗНАКОВ")
    print(f"{'='*60}")
    print(f"Размерность embedding вектора: {X_train.shape[1]}")
    print(f"Количество обучающих примеров: {X_train.shape[0]}")
    print(f"Количество тестовых примеров: {X_test.shape[0]}")
    
    # Оценка на обучающей выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    train_pred = model.predict(X_train_scaled)
    print(classification_report(y_train, train_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))
    
    # Оценка на тестовой выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    test_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, test_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))
    
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_test, test_pred))


def train_embeddings_logreg(embeddings_path=None, save=True):
    """
    Обучение модели классификации эмоций на основе усредненных FastText embeddings.
    
    Этапы:
    1. Загрузка предобученных FastText embeddings
    2. Загрузка текстов из JSONL файлов
    3. Предобработка текстов (lowercase, удаление пробелов)
    4. Преобразование текстов в векторы (усреднение word embeddings)
    5. Нормализация векторов
    6. Обучение LogisticRegression
    7. Оценка качества
    8. Сохранение модели
    
    Args:
        embeddings_path: Путь к файлу FastText embeddings (.bin)
        save: Сохранять ли модель после обучения
        
    Returns:
        model: Обученная модель
        scaler: Обученный scaler
        dataset_name: Имя датасета
    """
    # Используем путь по умолчанию, если не указан
    if embeddings_path is None:
        embeddings_path = DEFAULT_EMBEDDINGS_PATH
    
    # Загрузка FastText модели
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'='*60}")
    fasttext_model = load_fasttext_model(embeddings_path)
    
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test_small.jsonl'

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"\n📊 Датасет: {dataset_name}\n")

    # Загрузка обучающих данных
    print(f"{'='*60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'='*60}")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")
    print(f"Пример текста после предобработки: '{X_train_texts[0]}'")

    # Загрузка тестовых данных
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'='*60}")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    # Преобразование текстов в векторы
    print(f"\n{'='*60}")
    print("ПРЕОБРАЗОВАНИЕ ТЕКСТОВ В ВЕКТОРЫ")
    print(f"{'='*60}")
    
    # Проверка доступности tqdm для прогресс-бара
    try:
        import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("💡 Совет: установите tqdm для отображения прогресса (pip install tqdm)")
    
    # Преобразование обучающих текстов
    X_train = texts_to_vectors(X_train_texts, fasttext_model, verbose=use_tqdm)
    
    # Преобразование тестовых текстов
    X_test = texts_to_vectors(X_test_texts, fasttext_model, verbose=use_tqdm)

    # Нормализация признаков
    print(f"\n{'='*60}")
    print("НОРМАЛИЗАЦИЯ ПРИЗНАКОВ")
    print(f"{'='*60}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Признаки нормализованы (StandardScaler)")

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
        class_weight='balanced'
    )
    
    # Обучение модели
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    # Оценка модели
    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)
    
    # Сохранение модели
    if save:
        save_model(model, scaler, dataset_name)

    return model, scaler, dataset_name


def load_and_evaluate(embeddings_path=None):
    """
    Загружает существующую модель и оценивает её на данных.
    
    Args:
        embeddings_path: Путь к файлу FastText embeddings (.bin)
        
    Returns:
        model: Загруженная модель
        scaler: Загруженный scaler
    """
    print(f"{'='*60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'='*60}")
    
    # Используем путь по умолчанию, если не указан
    if embeddings_path is None:
        embeddings_path = DEFAULT_EMBEDDINGS_PATH
    
    # Загрузка FastText модели
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА FASTTEXT EMBEDDINGS")
    print(f"{'='*60}")
    fasttext_model = load_fasttext_model(embeddings_path)
    
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test_small.jsonl'
    
    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"\n📊 Датасет: {dataset_name}\n")
    
    # Загрузка модели
    model, scaler = load_model(dataset_name)
    
    # Загрузка обучающих данных
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА ОБУЧАЮЩИХ ДАННЫХ")
    print(f"{'='*60}")
    X_train_texts, y_train = load_texts_from_manifest(train_manifest)
    print(f"Количество обучающих примеров: {len(y_train)}")
    
    # Загрузка тестовых данных
    print(f"\n{'='*60}")
    print("ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ")
    print(f"{'='*60}")
    X_test_texts, y_test = load_texts_from_manifest(test_manifest)
    print(f"Количество тестовых примеров: {len(y_test)}")
    
    # Преобразование текстов в векторы
    print(f"\n{'='*60}")
    print("ПРЕОБРАЗОВАНИЕ ТЕКСТОВ В ВЕКТОРЫ")
    print(f"{'='*60}")
    
    X_train = texts_to_vectors(X_train_texts, fasttext_model, verbose=False)
    X_test = texts_to_vectors(X_test_texts, fasttext_model, verbose=False)
    
    # Оценка модели
    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Обучение или загрузка модели FastText Embeddings + LogReg для классификации эмоций по тексту'
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
    parser.add_argument(
        '--embeddings-path',
        type=str,
        default=None,
        help=f'Путь к файлу FastText embeddings (.bin). По умолчанию: {DEFAULT_EMBEDDINGS_PATH}'
    )
    
    args = parser.parse_args()
    
    # Проверка наличия gensim
    if not GENSIM_AVAILABLE:
        print("\n" + "="*60)
        print("ОШИБКА: Требуется библиотека gensim")
        print("="*60)
        print("\nУстановите gensim:")
        print("  pip install gensim")
        print("  или")
        print("  poetry add gensim")
        print("\n" + "="*60)
        exit(1)
    
    # Получение имени датасета для проверки существования модели
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    dataset_name = get_dataset_name(train_manifest)
    
    # Определение режима работы
    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        model, scaler, _ = train_embeddings_logreg(
            embeddings_path=args.embeddings_path,
            save=not args.no_save
        )
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        model, scaler = load_and_evaluate(embeddings_path=args.embeddings_path)
    else:  # auto
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            model, scaler = load_and_evaluate(embeddings_path=args.embeddings_path)
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            model, scaler, _ = train_embeddings_logreg(
                embeddings_path=args.embeddings_path,
                save=not args.no_save
            )
