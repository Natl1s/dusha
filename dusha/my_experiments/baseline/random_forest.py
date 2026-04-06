import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import joblib
import argparse
from datetime import datetime

# Импорт base_path из data.config
_data_config_path = Path(__file__).parent.parent.parent / "experiments" / "configs" / "data.config"
_data_config_ns = {}
exec(open(_data_config_path).read(), _data_config_ns)
DATASET_PATH = _data_config_ns['base_path']

# Путь для сохранения моделей
MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem  # random_forest


def save_model(model, scaler, dataset_name, model_name=MODEL_NAME):
    """Сохраняет модель и scaler в файл"""
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
    """Загружает модель и scaler из файла"""
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
    """Проверяет существование сохраненной модели"""
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{full_model_name}_scaler.pkl"
    return model_path.exists() and scaler_path.exists()


def get_dataset_name(train_manifest_path):
    """Извлекает имя датасета из пути к манифесту"""
    # Берём имя файла без расширения
    return Path(train_manifest_path).stem



def _to_fixed_vector(feat: np.ndarray) -> np.ndarray:
    """Преобразует feature-тензор в вектор фиксированной длины.

    Исходные признаки имеют форму вроде (1, 64, T), где T может отличаться
    между примерами. Чтобы получить одинаковую длину для всех объектов,
    агрегируем по временной оси (mean + std).
    """
    arr = np.asarray(feat)

    if arr.ndim == 0:
        raise ValueError("Пустой/скалярный feature-тензор")

    if arr.ndim == 1:
        return arr.astype(np.float32)

    mean_part = arr.mean(axis=-1).reshape(-1)
    std_part = arr.std(axis=-1).reshape(-1)
    return np.concatenate([mean_part, std_part]).astype(np.float32)


def load_features_from_manifest(manifest_path, base_path=None):
    """Загрузка features из JSONL манифеста"""
    features = []
    labels = []

    expected_dim = None

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            npy_path = data['hash_id'] + '.npy'
            feature_path = 'features' / Path(npy_path)

            if base_path:
                feature_path = base_path / feature_path

            try:
                feat = np.load(feature_path)
            except Exception as exc:
                raise ValueError(f"Не удалось загрузить feature-файл: {feature_path}") from exc

            feat_flat = _to_fixed_vector(feat)

            if expected_dim is None:
                expected_dim = feat_flat.shape[0]
            elif feat_flat.shape[0] != expected_dim:
                sample_id = data.get('id', '<unknown>')
                raise ValueError(
                    f"Несовпадение размерности признаков для sample id={sample_id}: "
                    f"получено {feat_flat.shape[0]}, ожидалось {expected_dim}. "
                    f"Файл: {feature_path}, исходная форма: {np.asarray(feat).shape}"
                )

            features.append(feat_flat)
            labels.append(data['emotion'])

    return np.stack(features), np.array(labels)


def print_random_forest_parameters(model, feature_dim):
    """Вывод критичных параметров Random Forest модели"""
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ RANDOM FOREST МОДЕЛИ")
    print(f"{'='*60}")
    
    # Основные гиперпараметры
    print(f"Количество деревьев (n_estimators): {model.n_estimators}")
    print(f"Максимальная глубина (max_depth): {model.max_depth if model.max_depth else 'Неограничена'}")
    print(f"Минимум samples для split (min_samples_split): {model.min_samples_split}")
    print(f"Минимум samples в листе (min_samples_leaf): {model.min_samples_leaf}")
    print(f"Максимум признаков для split (max_features): {model.max_features}")
    print(f"Критерий разделения: {model.criterion}")
    
    # Информация о классах
    print(f"\nКоличество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    
    # Статистика по деревьям
    print(f"\nСтатистика по деревьям:")
    tree_depths = [tree.get_depth() for tree in model.estimators_]
    tree_leaves = [tree.get_n_leaves() for tree in model.estimators_]
    
    print(f"  Глубина деревьев - Min: {min(tree_depths)}, Max: {max(tree_depths)}, Mean: {np.mean(tree_depths):.2f}")
    print(f"  Листьев в деревьях - Min: {min(tree_leaves)}, Max: {max(tree_leaves)}, Mean: {np.mean(tree_leaves):.2f}")
    
    # Feature importance
    print(f"\nВажность признаков (Feature Importance):")
    feature_importances = model.feature_importances_
    
    # Топ-10 самых важных признаков
    top_indices = np.argsort(feature_importances)[-10:][::-1]
    print(f"  Топ-10 признаков:")
    for i, idx in enumerate(top_indices, 1):
        print(f"    {i}. Признак {idx}: {feature_importances[idx]:.6f}")
    
    print(f"\nСтатистика важности признаков:")
    print(f"  Min: {feature_importances.min():.6f}")
    print(f"  Max: {feature_importances.max():.6f}")
    print(f"  Mean: {feature_importances.mean():.6f}")
    print(f"  Std: {feature_importances.std():.6f}")
    print(f"  Признаков с нулевой важностью: {np.sum(feature_importances == 0)}/{len(feature_importances)}")
    
    # OOB score если доступен
    if hasattr(model, 'oob_score_'):
        print(f"\nOut-of-Bag Score: {model.oob_score_:.4f}")


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках"""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Параметры модели
    print_random_forest_parameters(model, X_train.shape[1])
    
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


def train_random_forest(save=True, n_estimators=100, max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1, 
                       max_features='sqrt', oob_score=False):
    """
    Обучение Random Forest классификатора
    
    Args:
        save: Сохранить ли модель после обучения
        n_estimators: Количество деревьев в лесу (по умолчанию 100)
        max_depth: Максимальная глубина деревьев (None = неограничена)
        min_samples_split: Минимум samples для разделения узла (по умолчанию 2)
        min_samples_leaf: Минимум samples в листе (по умолчанию 1)
        max_features: Количество признаков для split ('sqrt', 'log2', None или int)
        oob_score: Вычислять ли Out-of-Bag score (по умолчанию False)
    """
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test.jsonl'

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка обучающих данных...")
    X_train, y_train = load_features_from_manifest(train_manifest, base_path.parent)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_manifest(test_manifest, base_path.parent)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    # Нормализация features (Random Forest не требует обязательной нормализации, но используем для единообразия)
    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение Random Forest
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ RANDOM FOREST МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Параметры:")
    print(f"  n_estimators={n_estimators}")
    print(f"  max_depth={max_depth}")
    print(f"  min_samples_split={min_samples_split}")
    print(f"  min_samples_leaf={min_samples_leaf}")
    print(f"  max_features={max_features}")
    print(f"  oob_score={oob_score}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        oob_score=oob_score,
        random_state=42,
        n_jobs=-1,  # Использовать все доступные процессоры
        verbose=1
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    # Оценка модели
    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)
    
    # Сохранение модели
    if save:
        save_model(model, scaler, dataset_name)

    return model, scaler, dataset_name


def load_and_evaluate():
    """Загрузить существующую модель и оценить её"""
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
    
    model, scaler = load_model(dataset_name)
    
    print("\nЗагрузка обучающих данных...")
    X_train, y_train = load_features_from_manifest(train_manifest, base_path.parent)
    print(f"Количество обучающих примеров: {len(y_train)}")
    
    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_manifest(test_manifest, base_path.parent)
    print(f"Количество тестовых примеров: {len(y_test)}")
    
    # Оценка модели
    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Обучение или загрузка Random Forest для классификации эмоций'
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
        '--n-estimators',
        type=int,
        default=100,
        help='Количество деревьев в лесу (по умолчанию: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Максимальная глубина деревьев (по умолчанию: None - неограничена)'
    )
    parser.add_argument(
        '--min-samples-split',
        type=int,
        default=2,
        help='Минимум samples для разделения узла (по умолчанию: 2)'
    )
    parser.add_argument(
        '--min-samples-leaf',
        type=int,
        default=1,
        help='Минимум samples в листе (по умолчанию: 1)'
    )
    parser.add_argument(
        '--max-features',
        type=str,
        default='sqrt',
        help='Количество признаков для split (по умолчанию: sqrt, также: log2, None или число)'
    )
    parser.add_argument(
        '--oob-score',
        action='store_true',
        help='Вычислять Out-of-Bag score'
    )
    
    args = parser.parse_args()
    
    # Преобразование max_features
    if args.max_features not in ['sqrt', 'log2', 'None']:
        try:
            max_features_value = int(args.max_features)
        except ValueError:
            try:
                max_features_value = float(args.max_features)
            except ValueError:
                max_features_value = args.max_features
    else:
        max_features_value = None if args.max_features == 'None' else args.max_features
    
    # Получение имени датасета для проверки существования модели
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train.jsonl'
    dataset_name = get_dataset_name(train_manifest)
    
    # Определение режима работы
    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        model, scaler, _ = train_random_forest(
            save=not args.no_save,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features_value,
            oob_score=args.oob_score
        )
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        model, scaler = load_and_evaluate()
    else:  # auto
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            model, scaler = load_and_evaluate()
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            model, scaler, _ = train_random_forest(
                save=not args.no_save,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                max_features=max_features_value,
                oob_score=args.oob_score
            )
