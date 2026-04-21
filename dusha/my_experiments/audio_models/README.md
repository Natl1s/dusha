# Baseline: Логистическая регрессия

## Описание
Скрипт для обучения и оценки логистической регрессии для классификации эмоций.

## Использование

### Автоматический режим (по умолчанию)
Загружает модель если существует, иначе обучает новую:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logictic_regressoin.py
```

### Обучить новую модель
Всегда обучает модель заново (перезаписывает существующую):
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logictic_regressoin.py --mode train
```

### Загрузить существующую модель
Только загружает и оценивает существующую модель:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logictic_regressoin.py --mode load
```

### Обучить без сохранения
Обучает модель, но не сохраняет результат:
```bash
poetry run python dusha/my_experiments/audio_models/baseline/logictic_regressoin.py --mode train --no-save
```

## Сохранение модели

Модели сохраняются в `dusha/my_experiments/audio_models/baseline/models_params/` с указанием датасета:
- `logictic_regressoin_combine_balanced_train_small_model.pkl` - модель
- `logictic_regressoin_combine_balanced_train_small_scaler.pkl` - нормализатор
- `logictic_regressoin_combine_balanced_train_small_model_YYYYMMDD_HHMMSS.pkl` - бэкап с временной меткой

Имя файла формируется как: `{имя_скрипта}_{имя_датасета}_{тип}.pkl`

## Вывод

Скрипт выводит:
- ✅ Количество загруженных примеров
- ✅ Распределение классов
- ✅ Параметры обученной модели
- ✅ Метрики качества (precision, recall, f1-score)
- ✅ Confusion matrix
- ✅ Путь к сохранённым файлам
