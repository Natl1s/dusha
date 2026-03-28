import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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
MODEL_NAME = Path(__file__).stem  # logictic_regressoin


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


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках"""
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
    
    print("\nMatрица ошибок:")
    print(confusion_matrix(y_test, test_pred))


def train_logistic_regression(save=True):
    # Пути к данным
    base_path = DATASET_PATH / 'processed_dataset_090'
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test_small.jsonl'

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

    # Нормализация features
    print("Нормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение логистической регрессии
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print(f"{'='*60}")
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42
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
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    test_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_test_small.jsonl'
    
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
        description='Обучение или загрузка логистической регрессии для классификации эмоций'
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
    train_manifest = base_path / 'aggregated_dataset' / 'combine_balanced_train_small.jsonl'
    dataset_name = get_dataset_name(train_manifest)
    
    # Определение режима работы
    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        model, scaler, _ = train_logistic_regression(save=not args.no_save)
    elif args.mode == 'load':
        print("📂 Режим: Загрузка существующей модели\n")
        model, scaler = load_and_evaluate()
    else:  # auto
        if model_exists(dataset_name):
            print("📂 Режим: AUTO - найдена существующая модель, загружаем...\n")
            model, scaler = load_and_evaluate()
        else:
            print("🎯 Режим: AUTO - модель не найдена, начинаем обучение...\n")
            model, scaler, _ = train_logistic_regression(save=not args.no_save)