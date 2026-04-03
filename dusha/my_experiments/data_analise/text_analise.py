"""
Анализ текстовых признаков: TF-IDF важные слова, биграммы, визуализация

Скрипт выполняет анализ текстов по эмоциям:
- Загрузка текстов из JSONL файлов (колонка speaker_text)
- Предобработка: lowercase, удаление лишних пробелов
- Построение TF-IDF векторизатора
- Анализ важных слов для каждой эмоции (топ-10)
- Анализ частых биграмм
- Визуализация результатов

Использование:
    poetry run python dusha/my_experiments/data_analise/text_analise.py
    poetry run python dusha/my_experiments/data_analise/text_analise.py --dataset combine_balanced_train_small
    poetry run python dusha/my_experiments/data_analise/text_analise.py --top-k 20 --save-plots
"""

import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import argparse
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Опциональные импорты
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']

# Маппинг эмоций
EMOTION_NAMES = ['angry', 'sad', 'neutral', 'positive']
EMOTION_COLORS = {
    'angry': '#FF6B6B',      # красный
    'sad': '#4ECDC4',        # голубой
    'neutral': '#95A5A6',    # серый
    'positive': '#52C76E'    # зеленый
}


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
    - speaker_text: текст для анализа
    - emotion: метка класса
    
    Args:
        manifest_path: Путь к .jsonl файлу
        
    Returns:
        texts: Список текстов
        labels: Массив меток эмоций
    """
    texts = []
    labels = []

    print(f"Загрузка текстов из {manifest_path}")
    
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

    print(f"✓ Загружено {len(texts)} текстов")
    
    # Статистика по эмоциям
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nРаспределение по эмоциям:")
    for emotion, count in zip(unique, counts):
        print(f"  {emotion:8s}: {count} ({count/len(labels)*100:.1f}%)")
    
    return texts, np.array(labels)


def analyze_tfidf_important_words(texts, labels, top_k=10):
    """
    Анализирует важные слова для каждой эмоции с помощью TF-IDF.
    
    Процесс:
    1. Обучаем TF-IDF векторизатор на всех текстах
    2. Для каждой эмоции вычисляем средние TF-IDF значения
    3. Выбираем топ-K слов с наибольшими значениями
    
    Args:
        texts: Список текстов
        labels: Массив меток эмоций
        top_k: Количество топ-слов для каждой эмоции
        
    Returns:
        top_words: Словарь {emotion: [(word, score), ...]}
        vectorizer: Обученный TF-IDF векторизатор
    """
    print(f"\n{'='*60}")
    print("АНАЛИЗ ВАЖНЫХ СЛОВ (TF-IDF)")
    print(f"{'='*60}")
    
    # Создание TF-IDF векторизатора
    # Используем только униграммы для анализа отдельных слов
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),      # только униграммы
        max_features=5000,       # ограничиваем словарь
        min_df=2,                # слово должно встречаться минимум в 2 документах
        max_df=0.8,              # не более чем в 80% документов
        sublinear_tf=True        # сублинейное масштабирование
    )
    
    # Обучаем векторизатор и преобразуем тексты
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    print(f"✓ TF-IDF матрица построена: {X.shape}")
    print(f"  - Размер словаря: {len(feature_names)}")
    
    # Для каждой эмоции находим важные слова
    top_words = {}
    
    for emotion in EMOTION_NAMES:
        # Выбираем тексты данной эмоции
        mask = labels == emotion
        if not mask.any():
            print(f"⚠ Нет текстов для эмоции: {emotion}")
            continue
        
        # Усредняем TF-IDF значения по всем текстам данной эмоции
        emotion_tfidf = X[mask].mean(axis=0).A1
        
        # Находим топ-K слов
        top_indices = emotion_tfidf.argsort()[-top_k:][::-1]
        top_words_list = [
            (feature_names[idx], emotion_tfidf[idx])
            for idx in top_indices
        ]
        
        top_words[emotion] = top_words_list
        
        # Выводим результаты
        print(f"\n{emotion.upper()} - топ-{top_k} слов:")
        for i, (word, score) in enumerate(top_words_list, 1):
            print(f"  {i:2d}. {word:20s} (TF-IDF: {score:.4f})")
    
    return top_words, vectorizer


def analyze_bigrams(texts, labels, top_k=10):
    """
    Анализирует частые биграммы для каждой эмоции.
    
    Биграмма - это последовательность из двух слов.
    Например: "не хочу", "очень рад", "так себе"
    
    Args:
        texts: Список текстов
        labels: Массив меток эмоций
        top_k: Количество топ-биграмм для каждой эмоции
        
    Returns:
        top_bigrams: Словарь {emotion: [(bigram, count), ...]}
    """
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЧАСТЫХ БИГРАММ")
    print(f"{'='*60}")
    
    # Создание векторизатора для биграмм
    vectorizer = CountVectorizer(
        ngram_range=(2, 2),      # только биграммы
        min_df=2,                # биграмма должна встречаться минимум в 2 документах
        max_df=0.5               # не более чем в 50% документов
    )
    
    # Обучаем векторизатор
    X = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    print(f"✓ Найдено {len(feature_names)} уникальных биграмм")
    
    # Для каждой эмоции находим частые биграммы
    top_bigrams = {}
    
    for emotion in EMOTION_NAMES:
        # Выбираем тексты данной эмоции
        mask = labels == emotion
        if not mask.any():
            continue
        
        # Суммируем частоты биграмм
        emotion_bigram_counts = X[mask].sum(axis=0).A1
        
        # Находим топ-K биграмм
        top_indices = emotion_bigram_counts.argsort()[-top_k:][::-1]
        top_bigrams_list = [
            (feature_names[idx], int(emotion_bigram_counts[idx]))
            for idx in top_indices
            if emotion_bigram_counts[idx] > 0
        ]
        
        top_bigrams[emotion] = top_bigrams_list
        
        # Выводим результаты
        print(f"\n{emotion.upper()} - топ-{min(top_k, len(top_bigrams_list))} биграмм:")
        for i, (bigram, count) in enumerate(top_bigrams_list, 1):
            print(f"  {i:2d}. '{bigram:25s}' (частота: {count})")
    
    return top_bigrams


def plot_top_words_bar(top_words, save_path=None):
    """
    Визуализация топ-слов для каждой эмоции в виде горизонтальных bar chart.
    
    Args:
        top_words: Словарь {emotion: [(word, score), ...]}
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ ТОП-СЛОВ")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTION_NAMES):
        if emotion not in top_words:
            continue
        
        ax = axes[idx]
        
        words, scores = zip(*top_words[emotion])
        words = list(words)
        scores = list(scores)
        
        # Рисуем горизонтальный bar chart
        y_pos = np.arange(len(words))
        ax.barh(y_pos, scores, color=EMOTION_COLORS[emotion], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()  # чтобы самое важное слово было сверху
        ax.set_xlabel('TF-IDF Score', fontsize=12)
        ax.set_title(f'{emotion.capitalize()} - важные слова', 
                    fontsize=14, fontweight='bold', color=EMOTION_COLORS[emotion])
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def plot_top_bigrams_bar(top_bigrams, save_path=None):
    """
    Визуализация топ-биграмм для каждой эмоции.
    
    Args:
        top_bigrams: Словарь {emotion: [(bigram, count), ...]}
        save_path: Путь для сохранения графика (опционально)
    """
    print(f"\n{'='*60}")
    print("ВИЗУАЛИЗАЦИЯ ТОП-БИГРАММ")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTION_NAMES):
        if emotion not in top_bigrams or not top_bigrams[emotion]:
            continue
        
        ax = axes[idx]
        
        bigrams, counts = zip(*top_bigrams[emotion][:10])  # топ-10
        bigrams = list(bigrams)
        counts = list(counts)
        
        # Рисуем горизонтальный bar chart
        y_pos = np.arange(len(bigrams))
        ax.barh(y_pos, counts, color=EMOTION_COLORS[emotion], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bigrams, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Частота', fontsize=12)
        ax.set_title(f'{emotion.capitalize()} - частые биграммы', 
                    fontsize=14, fontweight='bold', color=EMOTION_COLORS[emotion])
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def plot_wordcloud(texts, labels, save_path=None):
    """
    Создание облака слов для каждой эмоции.
    
    Требует библиотеку wordcloud: pip install wordcloud
    
    Args:
        texts: Список текстов
        labels: Массив меток эмоций
        save_path: Путь для сохранения графика (опционально)
    """
    if not WORDCLOUD_AVAILABLE:
        print(f"\n{'='*60}")
        print("ОБЛАКА СЛОВ")
        print(f"{'='*60}")
        print("⚠ Библиотека wordcloud не установлена")
        print("Установите: pip install wordcloud")
        return
    
    print(f"\n{'='*60}")
    print("СОЗДАНИЕ ОБЛАКОВ СЛОВ")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, emotion in enumerate(EMOTION_NAMES):
        # Объединяем все тексты данной эмоции
        emotion_texts = [text for text, label in zip(texts, labels) if label == emotion]
        
        if not emotion_texts:
            continue
        
        combined_text = ' '.join(emotion_texts)
        
        # Создаем облако слов
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(combined_text)
        
        ax = axes[idx]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{emotion.capitalize()}', 
                    fontsize=16, fontweight='bold', color=EMOTION_COLORS[emotion])
        
        print(f"✓ Облако слов для {emotion} создано")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def compare_emotions_vocabulary(top_words):
    """
    Сравнивает словари эмоций и находит уникальные/общие слова.
    
    Args:
        top_words: Словарь {emotion: [(word, score), ...]}
    """
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ СЛОВАРЕЙ ЭМОЦИЙ")
    print(f"{'='*60}")
    
    # Извлекаем слова для каждой эмоции
    emotion_word_sets = {}
    for emotion, words_scores in top_words.items():
        emotion_word_sets[emotion] = set(word for word, score in words_scores)
    
    # Находим общие слова между эмоциями
    print("\nОбщие слова между эмоциями:")
    emotions = list(emotion_word_sets.keys())
    for i in range(len(emotions)):
        for j in range(i+1, len(emotions)):
            emotion1, emotion2 = emotions[i], emotions[j]
            common = emotion_word_sets[emotion1] & emotion_word_sets[emotion2]
            if common:
                print(f"  {emotion1} ∩ {emotion2}: {', '.join(sorted(common))}")
    
    # Находим уникальные слова для каждой эмоции
    print("\nУникальные слова (только в этой эмоции):")
    for emotion, words in emotion_word_sets.items():
        # Слова, которые есть только в этой эмоции
        other_emotions_words = set()
        for other_emotion, other_words in emotion_word_sets.items():
            if other_emotion != emotion:
                other_emotions_words |= other_words
        
        unique = words - other_emotions_words
        if unique:
            print(f"  {emotion:8s}: {', '.join(sorted(unique)[:10])}")


def analyze_text_statistics(texts, labels):
    """
    Анализирует базовые статистики текстов по эмоциям.
    
    Args:
        texts: Список текстов
        labels: Массив меток эмоций
    """
    print(f"\n{'='*60}")
    print("СТАТИСТИКА ТЕКСТОВ ПО ЭМОЦИЯМ")
    print(f"{'='*60}")
    
    for emotion in EMOTION_NAMES:
        emotion_texts = [text for text, label in zip(texts, labels) if label == emotion]
        
        if not emotion_texts:
            continue
        
        # Вычисляем статистики
        word_counts = [len(text.split()) for text in emotion_texts]
        char_counts = [len(text) for text in emotion_texts]
        
        print(f"\n{emotion.upper()}:")
        print(f"  Количество текстов: {len(emotion_texts)}")
        print(f"  Среднее количество слов: {np.mean(word_counts):.2f} (σ={np.std(word_counts):.2f})")
        print(f"  Среднее количество символов: {np.mean(char_counts):.2f} (σ={np.std(char_counts):.2f})")
        print(f"  Мин/Макс слов: {np.min(word_counts)}/{np.max(word_counts)}")
        
        # Показываем примеры
        print(f"  Примеры текстов:")
        for i, text in enumerate(emotion_texts[:3], 1):
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"    {i}. '{preview}'")


def analyze_texts(manifest_path, top_k=10, save_plots=False):
    """
    Основная функция анализа текстов.
    
    Выполняет полный анализ:
    1. Загрузка текстов
    2. Статистика текстов
    3. TF-IDF важные слова
    4. Частые биграммы
    5. Сравнение словарей
    6. Визуализация
    
    Args:
        manifest_path: Путь к манифесту
        top_k: Количество топ-слов/биграмм
        save_plots: Сохранять ли графики на диск
    """
    print(f"\n{'='*60}")
    print("АНАЛИЗ ТЕКСТОВ")
    print(f"{'='*60}")
    print(f"Манифест: {manifest_path}")
    print(f"Топ-K: {top_k}")
    
    # Создаем директорию для сохранения графиков
    output_dir = None
    if save_plots:
        output_dir = Path(__file__).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        print(f"Графики будут сохранены в: {output_dir}")
    
    # 1. Загрузка текстов
    texts, labels = load_texts_from_manifest(manifest_path)
    
    if len(texts) == 0:
        print("❌ Не удалось загрузить тексты!")
        return
    
    # 2. Базовая статистика
    analyze_text_statistics(texts, labels)
    
    # 3. TF-IDF важные слова
    top_words, vectorizer = analyze_tfidf_important_words(texts, labels, top_k=top_k)
    
    # 4. Частые биграммы
    top_bigrams = analyze_bigrams(texts, labels, top_k=top_k)
    
    # 5. Сравнение словарей
    compare_emotions_vocabulary(top_words)
    
    # 6. Визуализация топ-слов
    save_path = output_dir / "top_words.png" if save_plots else None
    plot_top_words_bar(top_words, save_path=save_path)
    
    # 7. Визуализация биграмм
    save_path = output_dir / "top_bigrams.png" if save_plots else None
    plot_top_bigrams_bar(top_bigrams, save_path=save_path)
    
    # 8. Облака слов (опционально)
    if WORDCLOUD_AVAILABLE:
        save_path = output_dir / "wordclouds.png" if save_plots else None
        plot_wordcloud(texts, labels, save_path=save_path)
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*60}")
    
    if save_plots:
        print(f"\n✓ Все графики сохранены в: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Анализ текстов: TF-IDF важные слова, биграммы, визуализация'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='combine_balanced_train_small',
        help='Название датасета (имя файла без .jsonl)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Количество топ-слов/биграмм для каждой эмоции (по умолчанию: 10)'
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
    analyze_texts(
        manifest_path,
        top_k=args.top_k,
        save_plots=args.save_plots
    )
