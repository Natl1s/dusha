import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

_train_data_config_path = Path(__file__).parent.parent / "train_data.config"
_train_data_config_ns = {}
exec(open(_train_data_config_path).read(), _train_data_config_ns)
TRAIN_DATA_PATH = Path(_train_data_config_ns["train_data_path"])

_test_data_config_path = Path(__file__).parent.parent / "test_data.config"
_test_data_config_ns = {}
exec(open(_test_data_config_path).read(), _test_data_config_ns)
TEST_DATA_PATH = Path(_test_data_config_ns["test_data_path"])

# Путь для сохранения моделей
MODELS_DIR = Path(__file__).parent / "models_params"
MODEL_NAME = Path(__file__).stem  # svm


def save_model(
    model,
    scaler,
    dataset_name,
    model_name=MODEL_NAME,
    training_params=None,
    test_metrics=None,
):
    """Сохраняет модель и scaler в файл"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_model_name = f"{model_name}_{dataset_name}"
    model_path = MODELS_DIR / f"{full_model_name}_model.pkl"
    scaler_path = MODELS_DIR / f"{full_model_name}_scaler.pkl"
    model_path_timestamped = MODELS_DIR / f"{full_model_name}_model_{timestamp}.pkl"
    report_path = MODELS_DIR / f"{full_model_name}_training_report.txt"
    
    # Сохранение основных файлов
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Сохранение с временной меткой (для истории)
    joblib.dump({'model': model, 'scaler': scaler}, model_path_timestamped)

    report_lines = [
        f"model_name: {model_name}",
        f"dataset_name: {dataset_name}",
        f"saved_at: {timestamp}",
        "",
        "training_params:",
        json.dumps(training_params or {}, ensure_ascii=False, indent=2),
        "",
        "test_metrics:",
        json.dumps(test_metrics or {}, ensure_ascii=False, indent=2),
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ МОДЕЛИ СОХРАНЕНЫ")
    print(f"{'='*60}")
    print(f"✓ Модель: {model_path.absolute()}")
    print(f"✓ Scaler: {scaler_path.absolute()}")
    print(f"✓ Бэкап:  {model_path_timestamped.absolute()}")
    print(f"✓ Отчёт:  {report_path.absolute()}")
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


def get_features_base_from_manifest(manifest_path):
    return Path(manifest_path).parents[2]



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


def print_svm_parameters(model):
    """Вывод критичных параметров SVM модели"""
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ SVM МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Ядро (kernel): {model.kernel}")
    print(f"Параметр C (регуляризация): {model.C}")
    
    if model.kernel in ['rbf', 'poly', 'sigmoid']:
        print(f"Параметр gamma: {model.gamma}")
    
    if model.kernel == 'poly':
        print(f"Степень полинома (degree): {model.degree}")
    
    print(f"\nКоличество классов: {len(model.classes_)}")
    print(f"Классы: {model.classes_}")
    
    # Информация об опорных векторах
    print(f"\nОбщее количество опорных векторов: {model.n_support_.sum()}")
    print(f"Опорные векторы по классам: {dict(zip(model.classes_, model.n_support_))}")
    print(f"Процент опорных векторов: {model.n_support_.sum() / len(model.support_) * 100:.2f}%")
    
    # Форма матрицы опорных векторов
    print(f"\nФорма матрицы опорных векторов: {model.support_vectors_.shape}")
    print(f"Форма матрицы dual_coef: {model.dual_coef_.shape}")
    
    # Статистика по dual coefficients
    print(f"\nСтатистика dual coefficients:")
    print(f"  Min: {model.dual_coef_.min():.6f}")
    print(f"  Max: {model.dual_coef_.max():.6f}")
    print(f"  Mean: {model.dual_coef_.mean():.6f}")
    print(f"  Std: {model.dual_coef_.std():.6f}")


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Оценка модели на обучающей и тестовой выборках"""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Параметры модели
    print_svm_parameters(model)
    
    # Оценка на обучающей выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ОБУЧАЮЩЕЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    train_pred = model.predict(X_train_scaled)
    train_report_text = classification_report(
        y_train,
        train_pred,
        labels=['angry', 'sad', 'neutral', 'positive'],
        target_names=['angry', 'sad', 'neutral', 'positive'],
        zero_division=0,
    )
    print(train_report_text)
    
    # Оценка на тестовой выборке
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*60}")
    test_pred = model.predict(X_test_scaled)
    test_report_text = classification_report(
        y_test,
        test_pred,
        labels=['angry', 'sad', 'neutral', 'positive'],
        target_names=['angry', 'sad', 'neutral', 'positive'],
        zero_division=0,
    )
    print(test_report_text)
    
    print("\nМатрица ошибок:")
    test_cm = confusion_matrix(y_test, test_pred)
    print(test_cm)

    test_report_dict = classification_report(
        y_test,
        test_pred,
        labels=['angry', 'sad', 'neutral', 'positive'],
        target_names=['angry', 'sad', 'neutral', 'positive'],
        zero_division=0,
        output_dict=True,
    )

    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_classification_report_text": test_report_text,
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm.tolist(),
    }


def train_svm(save=True, kernel='rbf', C=1.0, gamma='scale'):
    """
    Обучение SVM классификатора
    
    Args:
        save: Сохранить ли модель после обучения
        kernel: Тип ядра ('linear', 'rbf', 'poly', 'sigmoid')
        C: Параметр регуляризации (по умолчанию 1.0)
        gamma: Параметр ядра для 'rbf', 'poly', 'sigmoid' (по умолчанию 'scale')
    """
    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    features_base_path = get_features_base_from_manifest(train_manifest)

    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")

    print("Загрузка обучающих данных...")
    X_train, y_train = load_features_from_manifest(train_manifest, features_base_path)
    print(f"Количество обучающих примеров: {len(y_train)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Распределение классов в train: {np.unique(y_train, return_counts=True)}")

    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_manifest(test_manifest, features_base_path)
    print(f"Количество тестовых примеров: {len(y_test)}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Распределение классов в test: {np.unique(y_test, return_counts=True)}")

    # Нормализация features
    print("\nНормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение SVM
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ SVM МОДЕЛИ")
    print(f"{'='*60}")
    print(f"Параметры: kernel={kernel}, C={C}, gamma={gamma}")
    
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=42,
        verbose=True
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Обучение завершено!")

    # Оценка модели
    metrics = evaluate_model(model, scaler, X_train, y_train, X_test, y_test)
    
    # Сохранение модели
    if save:
        training_params = {
            "kernel": model.kernel,
            "C": model.C,
            "gamma": model.gamma,
            "degree": model.degree,
            "coef0": model.coef0,
            "class_weight": model.class_weight,
            "random_state": model.random_state,
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
        }
        save_model(
            model,
            scaler,
            dataset_name,
            training_params=training_params,
            test_metrics=metrics,
        )

    return model, scaler, dataset_name


def load_and_evaluate():
    """Загрузить существующую модель и оценить её"""
    print(f"{'='*60}")
    print("ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print(f"{'='*60}")
    
    train_manifest = TRAIN_DATA_PATH
    test_manifest = TEST_DATA_PATH
    features_base_path = get_features_base_from_manifest(train_manifest)
    
    # Извлечение имени датасета
    dataset_name = get_dataset_name(train_manifest)
    print(f"📊 Датасет: {dataset_name}\n")
    
    model, scaler = load_model(dataset_name)
    
    print("\nЗагрузка обучающих данных...")
    X_train, y_train = load_features_from_manifest(train_manifest, features_base_path)
    print(f"Количество обучающих примеров: {len(y_train)}")
    
    print("\nЗагрузка тестовых данных...")
    X_test, y_test = load_features_from_manifest(test_manifest, features_base_path)
    print(f"Количество тестовых примеров: {len(y_test)}")
    
    # Оценка модели
    evaluate_model(model, scaler, X_train, y_train, X_test, y_test)

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Обучение или загрузка SVM для классификации эмоций'
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
        '--kernel',
        type=str,
        choices=['linear', 'rbf', 'poly', 'sigmoid'],
        default='rbf',
        help='Тип ядра для SVM (по умолчанию: rbf)'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Параметр регуляризации C (по умолчанию: 1.0)'
    )
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Параметр gamma для ядра (по умолчанию: scale, также можно: auto или числовое значение)'
    )
    
    args = parser.parse_args()
    
    # Преобразование gamma в правильный тип
    try:
        gamma_value = float(args.gamma)
    except ValueError:
        gamma_value = args.gamma  # 'scale' или 'auto'
    
    # Получение имени датасета для проверки существования модели
    dataset_name = get_dataset_name(TRAIN_DATA_PATH)
    
    # Определение режима работы
    if args.mode == 'train':
        print("🎯 Режим: Обучение новой модели\n")
        model, scaler, _ = train_svm(
            save=not args.no_save,
            kernel=args.kernel,
            C=args.C,
            gamma=gamma_value
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
            model, scaler, _ = train_svm(
                save=not args.no_save,
                kernel=args.kernel,
                C=args.C,
                gamma=gamma_value
            )
