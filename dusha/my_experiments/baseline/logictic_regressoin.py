import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json



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
        count = 0
        for line in f:
            count = count + 1
            if count > 100:
                break

            data = json.loads(line.strip())

            feature_path = data['tensor']

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
            labels.append(data['label'])

    return np.stack(features), np.array(labels)


def train_logistic_regression():
    # Пути к данным (измените на ваши)
    base_path = Path('/home/natlis/PycharmProjects/dusha_new/dusha/dataset/processed_dataset_090')
    train_manifest = base_path / 'train' / 'crowd_train_my.jsonl'
    test_manifest = base_path / 'test' / 'crowd_test_my.jsonl'


    print("Загрузка обучающих данных...")
    X_train, y_train = load_features_from_manifest(train_manifest, base_path.parent)

    print("Загрузка тестовых данных...")
    X_test, y_test = load_features_from_manifest(test_manifest, base_path.parent)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # Нормализация features
    print("Нормализация features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение логистической регрессии
    print("Обучение модели...")
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Оценка модели
    print("\nОценка на обучающей выборке:")
    train_pred = model.predict(X_train_scaled)
    print(classification_report(y_train, train_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))

    print("\nОценка на тестовой выборке:")
    test_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, test_pred,
                                target_names=['angry', 'sad', 'neutral', 'positive']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    return model, scaler


if __name__ == "__main__":
    model, scaler = train_logistic_regression()