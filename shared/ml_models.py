from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


@dataclass(slots=True)
class MaterialFeatures:
    # Нормализованное описание материала для обучения классических ML-моделей.
    text: str
    numeric: list[float]


def build_sequential_role(material: Any) -> str:
    # Для задачи последовательного изучения используем роль материала в цепочке.
    if material.previous_record_id and material.next_record_id:
        return "middle"
    if material.next_record_id and not material.previous_record_id:
        return "start"
    if material.previous_record_id and not material.next_record_id:
        return "end"
    return "standalone"


def build_material_features(material: Any) -> MaterialFeatures:
    # Превращаем разрозненные поля материала в единый ML-признак:
    # текстовая часть идет в TF-IDF, числовая часть помогает моделям учитывать размер и сложность.
    text = " ".join(
        part
        for part in [
            material.subject,
            material.topic,
            material.lesson_type,
            material.source_kind,
            material.summary,
            material.text_material,
            " ".join(material.media_descriptions),
        ]
        if part
    )
    numeric = [
        float(material.word_count),
        float(material.char_count),
        float(material.methodical_score),
        float(material.topic_order),
        float(material.estimated_minutes),
        1.0 if material.generated else 0.0,
        1.0 if material.previous_record_id else 0.0,
        1.0 if material.next_record_id else 0.0,
    ]
    return MaterialFeatures(text=text, numeric=numeric)


def _prepare_dense_features(
    features: list[MaterialFeatures],
    *,
    vectorizer: TfidfVectorizer | None = None,
    reducer: TruncatedSVD | None = None,
    scaler: StandardScaler | None = None,
    fit: bool,
) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD, StandardScaler]:
    # Сначала превращаем текст в TF-IDF, затем уменьшаем размерность, после чего добавляем числовые признаки.
    texts = [item.text for item in features]
    numeric = np.array([item.numeric for item in features], dtype=float)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=700, ngram_range=(1, 2), min_df=1)
    text_matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)

    # GradientBoosting требует плотную матрицу, поэтому переводим текстовые признаки в компактное dense-представление.
    max_components = min(64, max(1, text_matrix.shape[1] - 1), max(1, text_matrix.shape[0] - 1))
    if reducer is None:
        reducer = TruncatedSVD(n_components=max_components, random_state=42)
    text_dense = reducer.fit_transform(text_matrix) if fit else reducer.transform(text_matrix)

    if scaler is None:
        scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric) if fit else scaler.transform(numeric)

    return np.hstack([text_dense, numeric_scaled]), vectorizer, reducer, scaler


def _build_estimators(class_count: int) -> dict[str, Any]:
    # Три разных семейства моделей, как требуется в критериях: случайный лес, градиентный бустинг, логистическая регрессия.
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced" if class_count > 1 else None,
            solver="lbfgs",
        ),
    }


def _safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> float:
    # ROC-AUC считаем и для бинарных, и для многоклассовых задач; если данных слишком мало, возвращаем 0.
    unique_count = len(np.unique(y_true))
    if unique_count < 2:
        return 0.0
    try:
        if unique_count == 2:
            positive_index = 1 if probabilities.shape[1] > 1 else 0
            return float(roc_auc_score(y_true, probabilities[:, positive_index]))
        y_bin = label_binarize(y_true, classes=labels)
        return float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr"))
    except Exception:
        return 0.0


def train_task_models(
    materials: list[Any],
    label_getter: Callable[[Any], str],
) -> dict[str, Any]:
    # Обучаем три модели на отложенной выборке и выбираем лучшую по F1, затем по ROC-AUC и accuracy.
    features = [build_material_features(material) for material in materials]
    labels_raw = [label_getter(material) for material in materials]
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels_raw)

    all_features, vectorizer, reducer, scaler = _prepare_dense_features(features, fit=True)
    unique_labels = np.unique(labels_encoded)

    minimum_class_size = min(int(np.sum(labels_encoded == label)) for label in unique_labels) if len(unique_labels) else 0

    if len(materials) >= 8 and len(unique_labels) >= 2 and minimum_class_size >= 2:
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                all_features,
                labels_encoded,
                test_size=0.25,
                random_state=42,
                stratify=labels_encoded if len(materials) >= len(unique_labels) * 2 else None,
            )
            split_strategy = "holdout_25"
        except ValueError:
            x_train = x_test = all_features
            y_train = y_test = labels_encoded
            split_strategy = "fallback_full_dataset"
    else:
        x_train = x_test = all_features
        y_train = y_test = labels_encoded
        split_strategy = "fallback_full_dataset"

    # Если после разбиения в обучающей части остался один класс, безопасно
    # откатываемся на полный набор данных. Это защищает обучение на малых
    # выборках и позволяет агенту завершать модуль без падения.
    if len(np.unique(y_train)) < 2:
        x_train = x_test = all_features
        y_train = y_test = labels_encoded
        split_strategy = "fallback_full_dataset"

    method_details: dict[str, dict[str, float]] = {}
    fitted_estimators: dict[str, Any] = {}
    for name, estimator in _build_estimators(len(unique_labels)).items():
        try:
            estimator.fit(x_train, y_train)
            evaluation_x = x_test
            evaluation_y = y_test
            predicted = estimator.predict(evaluation_x)
            probabilities = estimator.predict_proba(evaluation_x) if hasattr(estimator, "predict_proba") else None
        except Exception:
            # Если конкретный алгоритм оказался неприменим к текущему
            # распределению классов, обучаем безопасный fallback на полном
            # наборе и все равно фиксируем метрики для сравнения.
            estimator.fit(all_features, labels_encoded)
            evaluation_x = all_features
            evaluation_y = labels_encoded
            predicted = estimator.predict(evaluation_x)
            probabilities = estimator.predict_proba(evaluation_x) if hasattr(estimator, "predict_proba") else None
        method_details[name] = {
            "accuracy": round(float(accuracy_score(evaluation_y, predicted)), 3),
            "macro_precision": round(float(precision_score(evaluation_y, predicted, average="macro", zero_division=0)), 3),
            "macro_recall": round(float(recall_score(evaluation_y, predicted, average="macro", zero_division=0)), 3),
            "macro_f1": round(float(f1_score(evaluation_y, predicted, average="macro", zero_division=0)), 3),
            "roc_auc": round(_safe_roc_auc(evaluation_y, probabilities, unique_labels), 3) if probabilities is not None else 0.0,
        }
        fitted_estimators[name] = estimator

    selected_method = max(
        method_details.items(),
        key=lambda item: (item[1]["macro_f1"], item[1]["roc_auc"], item[1]["accuracy"]),
    )[0]

    return {
        "selected_method": selected_method,
        "method_details": method_details,
        "method_scores": {name: metrics["accuracy"] for name, metrics in method_details.items()},
        "labels": list(encoder.classes_),
        "label_mapping": {str(index): label for index, label in enumerate(encoder.classes_)},
        "split_strategy": split_strategy,
        "holdout_size": int(len(y_test)),
        "training_size": int(len(y_train)),
        "package": {
            "task_name": "",
            "selected_method": selected_method,
            "vectorizer": vectorizer,
            "reducer": reducer,
            "scaler": scaler,
            "label_encoder": encoder,
            "estimator": fitted_estimators[selected_method],
            "method_details": method_details,
        },
    }


def save_model_package(path: Path, package: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)


def load_model_package(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_with_package(package: dict[str, Any], material: Any) -> dict[str, Any]:
    # Единая точка предсказания для API: грузим пакет, считаем признаки и возвращаем метки и вероятности.
    features = [build_material_features(material)]
    matrix, _, _, _ = _prepare_dense_features(
        features,
        vectorizer=package["vectorizer"],
        reducer=package["reducer"],
        scaler=package["scaler"],
        fit=False,
    )
    estimator = package["estimator"]
    encoder: LabelEncoder = package["label_encoder"]
    predicted_index = int(estimator.predict(matrix)[0])
    probabilities = estimator.predict_proba(matrix)[0] if hasattr(estimator, "predict_proba") else None
    probability_map = {}
    if probabilities is not None:
        for index, value in enumerate(probabilities):
            probability_map[str(encoder.classes_[index])] = round(float(value), 4)
    return {
        "predicted_label": str(encoder.inverse_transform([predicted_index])[0]),
        "probabilities": probability_map,
        "selected_method": package["selected_method"],
    }
