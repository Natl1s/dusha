# Baseline Text Models

Модели базовой классификации эмоций на основе текстовых признаков.

## Оглавление

1. [TF-IDF_LogReg.py](#tf-idf_logreg) - TF-IDF + Логистическая регрессия
2. [Embeddings_LogReg.py](#embeddings_logreg) - FastText Embeddings + Логистическая регрессия

---

## TF-IDF_LogReg

Скрипт для обучения модели классификации эмоций на основе TF-IDF признаков и Логистической регрессии.

### Описание

Скрипт выполняет следующие шаги:
1. **Загрузка данных** из JSONL файлов (колонка `speaker_text`)
2. **Предобработка текста**:
   - Приведение к нижнему регистру (lowercase)
   - Удаление лишних пробелов
3. **Построение TF-IDF признаков**:
   - Word-level (1-граммы)
   - N-граммы (биграммы)
   - Максимум 10000 признаков
   - Фильтрация редких и частых слов
4. **Обучение LogisticRegression** с балансировкой классов
5. **Оценка качества** на обучающей и тестовой выборках
6. **Сохранение модели** и векторизатора

### Использование

```bash
# Обучить новую модель
poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode train

# Загрузить существующую модель и оценить
poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode load

# Автоматический режим (загрузить если есть, иначе обучить)
poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode auto

# Обучить без сохранения
poetry run python dusha/my_experiments/baseline_text/TF-IDF_LogReg.py --mode train --no-save
```

### Параметры TF-IDF

- `ngram_range=(1, 2)` - униграммы и биграммы
- `max_features=10000` - максимальное количество признаков
- `min_df=2` - минимальная частота документа (слово должно встречаться минимум в 2 документах)
- `max_df=0.8` - максимальная частота документа (фильтр частых слов)
- `sublinear_tf=True` - сублинейное масштабирование TF

### Параметры модели

- `solver='lbfgs'` - оптимизатор Limited-memory BFGS
- `max_iter=1000` - максимальное количество итераций
- `class_weight='balanced'` - автоматическая балансировка классов

### Выходные файлы

Модели сохраняются в директории `models_params/`:
- `TF-IDF_LogReg_{dataset_name}_model.pkl` - обученная модель
- `TF-IDF_LogReg_{dataset_name}_vectorizer.pkl` - TF-IDF векторизатор
- `TF-IDF_LogReg_{dataset_name}_model_{timestamp}.pkl` - бэкап с временной меткой

---

## Embeddings_LogReg

Скрипт для обучения модели классификации эмоций на основе усредненных FastText embeddings и Логистической регрессии.

### Описание

Скрипт выполняет следующие шаги:
1. **Загрузка предобученных FastText embeddings** (поддержка OOV слов через n-граммы)
2. **Загрузка данных** из JSONL файлов (колонка `speaker_text`)
3. **Предобработка текста**:
   - Приведение к нижнему регистру (lowercase)
   - Удаление лишних пробелов
4. **Преобразование текстов в векторы**:
   - Каждое слово → вектор (FastText)
   - Текст → усреднение векторов всех слов
5. **Нормализация векторов** (StandardScaler)
6. **Обучение LogisticRegression** с балансировкой классов
7. **Оценка качества** на обучающей и тестовой выборках
8. **Сохранение модели** и scaler

### Требования

```bash
# Установка библиотеки gensim
pip install gensim
# Или
poetry add gensim

# Опционально: прогресс-бар
pip install tqdm
```

**Предобученные embeddings:**

Скачайте FastText embeddings для русского языка:
```bash
# Скачать предобученную модель (2.3 GB)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz

# Распаковать
gunzip cc.ru.300.bin.gz

# Переместить в папку
mkdir -p ~/fasttext_models
mv cc.ru.300.bin ~/fasttext_models/
```

Альтернативные источники:
- [FastText Official](https://fasttext.cc/docs/en/crawl-vectors.html)
- [RusVectores](https://rusvectores.org/ru/models/)

### Использование

```bash
# Обучить новую модель (с путем к embeddings по умолчанию)
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train

# Указать путь к embeddings
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train --embeddings-path /path/to/cc.ru.300.bin

# Загрузить существующую модель и оценить
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode load

# Автоматический режим (загрузить если есть, иначе обучить)
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode auto

# Обучить без сохранения
poetry run python dusha/my_experiments/baseline_text/Embeddings_LogReg.py --mode train --no-save
```

### Параметры FastText

- **Размерность вектора**: 300 (зависит от предобученной модели)
- **OOV обработка**: FastText разбивает неизвестные слова на n-граммы символов
- **Усреднение**: Вектор текста = среднее арифметическое векторов всех слов

### Параметры модели

- `solver='lbfgs'` - оптимизатор Limited-memory BFGS
- `max_iter=1000` - максимальное количество итераций
- `class_weight='balanced'` - автоматическая балансировка классов

### Выходные файлы

Модели сохраняются в директории `models_params/`:
- `Embeddings_LogReg_{dataset_name}_model.pkl` - обученная модель
- `Embeddings_LogReg_{dataset_name}_scaler.pkl` - StandardScaler
- `Embeddings_LogReg_{dataset_name}_model_{timestamp}.pkl` - бэкап с временной меткой

### Преимущества FastText

1. **Работа с OOV словами**: FastText может создать вектор для слова, которого нет в словаре
2. **Морфология**: Учитывает части слов (префиксы, суффиксы, корни)
3. **Опечатки**: Лучше справляется с опечатками благодаря n-граммам

---

## Сравнение моделей

| Модель | Признаки | Размерность | Преимущества |
|--------|----------|-------------|--------------|
| TF-IDF_LogReg | Частотные | ~10000 | Быстрая, интерпретируемая, не требует предобученных моделей |
| Embeddings_LogReg | Семантические | 300 | Понимает значения слов, работает с OOV, учитывает морфологию |
