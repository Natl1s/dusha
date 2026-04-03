# Установка и использование Embeddings_LogReg

## Быстрый старт

### 1. Установка зависимостей

```bash
# Установка gensim для работы с FastText
poetry add gensim

# Опционально: tqdm для прогресс-бара
poetry add tqdm
```

### 2. Скачивание предобученных embeddings

**Вариант 1: FastText Common Crawl (рекомендуется)**

```bash
# Создаем директорию
mkdir -p ~/fasttext_models
cd ~/fasttext_models

# Скачиваем модель (2.3 GB)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz

# Распаковываем
gunzip cc.ru.300.bin.gz

# Проверяем
ls -lh cc.ru.300.bin
```

**Вариант 2: FastText Wikipedia**

```bash
# Более легкая модель
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.bin
```

**Вариант 3: RusVectores**

Посетите https://rusvectores.org/ru/models/ и скачайте любую FastText модель.

### 3. Запуск обучения

```bash
# С путем по умолчанию (~/fasttext_models/cc.ru.300.bin)
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train

# Или с указанием пути
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py \
    --mode train \
    --embeddings-path ~/fasttext_models/cc.ru.300.bin
```

## Примеры использования

### Обучить новую модель

```bash
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py \
    --mode train \
    --embeddings-path ~/fasttext_models/cc.ru.300.bin
```

### Загрузить и оценить существующую модель

```bash
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py \
    --mode load \
    --embeddings-path ~/fasttext_models/cc.ru.300.bin
```

### Автоматический режим

```bash
# Загрузит модель если она существует, иначе обучит новую
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py \
    --mode auto \
    --embeddings-path ~/fasttext_models/cc.ru.300.bin
```

### Обучение без сохранения (для тестирования)

```bash
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py \
    --mode train \
    --no-save \
    --embeddings-path ~/fasttext_models/cc.ru.300.bin
```

## Время выполнения

- **Загрузка FastText модели**: ~1-2 минуты (единоразово при запуске)
- **Векторизация текстов**: ~30-60 секунд (зависит от количества примеров)
- **Обучение LogReg**: ~10-30 секунд
- **Общее время**: ~2-4 минуты

## Размер файлов

- **Предобученные embeddings**: 2.3 GB (cc.ru.300.bin)
- **Сохраненная модель**: ~50 KB
- **Сохраненный scaler**: ~10 KB

## Устранение проблем

### Ошибка: "gensim не установлен"

```bash
poetry add gensim
```

### Ошибка: "Файл с embeddings не найден"

Проверьте путь к файлу:
```bash
ls -lh ~/fasttext_models/cc.ru.300.bin
```

Если файл не существует, скачайте его (см. раздел "Скачивание предобученных embeddings").

### Ошибка: "Memory Error" при загрузке embeddings

Модель FastText требует ~4-6 GB оперативной памяти. Если памяти недостаточно:
1. Используйте более легкую модель (wiki.ru.bin вместо cc.ru.300.bin)
2. Используйте KeyedVectors вместо полной модели (требует конвертации)

### Медленная векторизация

Установите tqdm для отображения прогресса:
```bash
poetry add tqdm
```

## Альтернативные источники embeddings

1. **FastText Official**: https://fasttext.cc/docs/en/crawl-vectors.html
2. **RusVectores**: https://rusvectores.org/ru/models/
3. **Navec** (компактные embeddings для русского): https://github.com/natasha/navec

## Что делает скрипт?

1. Загружает предобученные FastText embeddings
2. Читает тексты из JSONL файлов
3. Предобрабатывает тексты (lowercase, удаление пробелов)
4. Для каждого текста:
   - Разбивает на слова
   - Каждое слово преобразует в вектор (300-мерный)
   - Усредняет векторы всех слов → получается вектор текста
5. Нормализует векторы (StandardScaler)
6. Обучает Логистическую регрессию
7. Оценивает качество и сохраняет модель
