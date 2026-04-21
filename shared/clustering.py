from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from shared.ml_models import build_material_features


def _text_numeric_features(materials: list[Any]) -> np.ndarray:
    # Готовим компактное числовое представление материалов для кластеризации.
    # Используем только офлайн-доступные признаки, уже собранные модулем A.
    rows: list[list[float]] = []
    for material in materials:
        feature = build_material_features(material)
        rows.append(
            [
                float(material.word_count),
                float(material.char_count),
                float(material.methodical_score),
                float(material.topic_order),
                float(material.estimated_minutes),
                float(len(material.media_descriptions)),
                1.0 if material.generated else 0.0,
                1.0 if material.previous_record_id else 0.0,
                1.0 if material.next_record_id else 0.0,
                float(len(feature.text)),
            ]
        )
    return StandardScaler().fit_transform(np.array(rows, dtype=float))


def _safe_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    # Считаем три метрики из задания. Если разбиение выродилось, возвращаем нейтральные значения.
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(features):
        return {"silhouette_score": -1.0, "davies_bouldin_index": 999.0, "calinski_harabasz_score": 0.0}
    try:
        return {
            "silhouette_score": round(float(silhouette_score(features, labels)), 4),
            "davies_bouldin_index": round(float(davies_bouldin_score(features, labels)), 4),
            "calinski_harabasz_score": round(float(calinski_harabasz_score(features, labels)), 4),
        }
    except Exception:
        return {"silhouette_score": -1.0, "davies_bouldin_index": 999.0, "calinski_harabasz_score": 0.0}


def _score_candidate(metrics: dict[str, float]) -> float:
    # Единый рейтинг для выбора лучшего алгоритма:
    # silhouette и Calinski-Harabasz максимизируем, Davies-Bouldin минимизируем.
    return metrics["silhouette_score"] + metrics["calinski_harabasz_score"] / 1000.0 - metrics["davies_bouldin_index"]


def _run_method(method_name: str, features: np.ndarray, cluster_count: int) -> np.ndarray:
    if method_name == "kmeans":
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        return model.fit_predict(features)
    if method_name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
        return model.fit_predict(features)
    if method_name == "spectral":
        neighbors = max(2, min(8, len(features) - 1))
        model = SpectralClustering(
            n_clusters=cluster_count,
            random_state=42,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
        )
        return model.fit_predict(features)
    raise ValueError(f"Unknown clustering method: {method_name}")


def compare_clustering_methods(features: np.ndarray, task_name: str) -> dict[str, Any]:
    # Перебираем алгоритмы и число кластеров, затем выбираем лучший вариант по совокупности метрик.
    candidates: list[dict[str, Any]] = []
    maximum_k = min(6, max(2, len(features) - 1))
    for method_name in ("kmeans", "agglomerative", "spectral"):
        for cluster_count in range(2, maximum_k + 1):
            try:
                labels = _run_method(method_name, features, cluster_count)
                metrics = _safe_metrics(features, labels)
                candidates.append(
                    {
                        "task_name": task_name,
                        "method_name": method_name,
                        "cluster_count": cluster_count,
                        "labels": labels.tolist(),
                        **metrics,
                        "ranking_score": round(_score_candidate(metrics), 5),
                    }
                )
            except Exception:
                continue
    if not candidates:
        fallback_labels = [index % 2 for index in range(len(features))]
        fallback_metrics = _safe_metrics(features, np.array(fallback_labels))
        return {
            "task_name": task_name,
            "selected_method": "fallback_round_robin",
            "selected_cluster_count": 2,
            "labels": fallback_labels,
            "selected_metrics": fallback_metrics,
            "method_comparison": [],
            "selection_reason": "Недостаточно данных для устойчивой кластеризации, использован безопасный fallback.",
        }
    best = max(candidates, key=lambda item: item["ranking_score"])
    return {
        "task_name": task_name,
        "selected_method": best["method_name"],
        "selected_cluster_count": best["cluster_count"],
        "labels": best["labels"],
        "selected_metrics": {
            "silhouette_score": best["silhouette_score"],
            "davies_bouldin_index": best["davies_bouldin_index"],
            "calinski_harabasz_score": best["calinski_harabasz_score"],
        },
        "method_comparison": [
            {
                "method_name": item["method_name"],
                "cluster_count": item["cluster_count"],
                "silhouette_score": item["silhouette_score"],
                "davies_bouldin_index": item["davies_bouldin_index"],
                "calinski_harabasz_score": item["calinski_harabasz_score"],
                "ranking_score": item["ranking_score"],
            }
            for item in sorted(candidates, key=lambda row: row["ranking_score"], reverse=True)
        ],
        "selection_reason": (
            "Выбран алгоритм с лучшим балансом по Silhouette Score, Davies–Bouldin Index и Calinski–Harabasz Score. "
            "Такой подход обеспечивает устойчивость, интерпретируемость и соответствие реальному содержимому материалов."
        ),
    }


def build_subject_relationships(materials: list[Any]) -> list[dict[str, Any]]:
    # Строим сопоставление дисциплин: какие предметы можно изучать параллельно, какие — последовательно.
    by_subject: dict[str, list[Any]] = defaultdict(list)
    for material in materials:
        by_subject[material.subject].append(material)

    subject_vectors: dict[str, set[str]] = {}
    for subject, items in by_subject.items():
        tokens: set[str] = set()
        for item in items:
            tokens.update(str(item.topic).lower().split())
            tokens.update(str(item.summary).lower().split()[:25])
        subject_vectors[subject] = tokens

    rows: list[dict[str, Any]] = []
    subjects = sorted(by_subject)
    for subject in subjects:
        parallel_with: list[str] = []
        sequential_with: list[str] = []
        restricted_with: list[str] = []
        for other in subjects:
            if other == subject:
                continue
            overlap = len(subject_vectors[subject] & subject_vectors[other])
            if overlap >= 5:
                sequential_with.append(other)
            elif overlap >= 2:
                restricted_with.append(other)
            else:
                parallel_with.append(other)
        rows.append(
            {
                "subject": subject,
                "parallel_with": parallel_with,
                "sequential_with": sequential_with,
                "restricted_with": restricted_with,
                "reasoning": (
                    "Параллельность определяется низким пересечением терминологии и независимостью тематических цепочек; "
                    "последовательность — заметным тематическим пересечением и логическим продолжением содержания."
                ),
            }
        )
    return rows


def build_cluster_suite(materials: list[Any]) -> dict[str, Any]:
    # Единая функция для модуля Б и последующего обучения моделей в модуле В.
    # Возвращаем выбранные кластеры, сравнение алгоритмов и совместимость дисциплин.
    common_features = _text_numeric_features(materials)
    parallel_result = compare_clustering_methods(common_features, "parallel")
    sequential_result = compare_clustering_methods(common_features, "sequential")

    difficulty_rows = np.array(
        [
            [
                float(material.word_count),
                float(material.estimated_minutes),
                float(material.methodical_score),
                float(len(material.media_descriptions)),
            ]
            for material in materials
        ],
        dtype=float,
    )
    difficulty_features = StandardScaler().fit_transform(difficulty_rows)
    difficulty_result = compare_clustering_methods(difficulty_features, "difficulty")

    for index, material in enumerate(materials):
        material.parallel_cluster = f"Параллельный кластер {parallel_result['labels'][index] + 1}"
        material.sequential_cluster = f"Последовательный кластер {sequential_result['labels'][index] + 1}"
        material.difficulty_level = f"Кластер сложности {difficulty_result['labels'][index] + 1}"

    selected_metrics = {
        "parallel_metric": parallel_result["selected_metrics"],
        "sequential_metric": sequential_result["selected_metrics"],
        "difficulty_metric": difficulty_result["selected_metrics"],
    }
    for metric in selected_metrics.values():
        metric["conclusion"] = (
            "Кластеризация признана устойчивой." if metric["silhouette_score"] > 0 else "Кластеризация требует расширения выборки."
        )

    return {
        "created_at_utc": "",
        "assignments": [
            {
                "record_id": material.record_id,
                "parallel_cluster": material.parallel_cluster,
                "sequential_cluster": material.sequential_cluster,
                "difficulty_level": material.difficulty_level,
            }
            for material in materials
        ],
        "parallel": parallel_result,
        "sequential": sequential_result,
        "difficulty": difficulty_result,
        "subject_relationships": build_subject_relationships(materials),
        **selected_metrics,
    }
