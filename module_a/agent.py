from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from math import sqrt
from statistics import mean

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import EXPORTS_DIR, MANIFEST_PATH, REPORTS_DIR, ensure_workspace
from shared.core import (
    Material,
    attach_neighbors,
    build_material_record,
    compute_hash,
    cosine_similarity,
    criteria_lines,
    deduplicate_materials,
    generate_missing_material,
    load_sources_for_module_a,
    text_vector,
    tokenize,
    utc_now,
)
from shared.parsers import get_parse_warnings, reset_parse_warnings
from shared.storage import save_csv, save_database


AUTOGENERATE_ENV = "MODULE_A_AUTOGENERATE"
CHARTS_DIR = REPORTS_DIR / "charts"

# Описание каждого признака набора данных для отчета по пункту 1.4.
ATTRIBUTE_CATALOG: dict[str, dict[str, str]] = {
    "record_id": {"type": "string", "description": "Уникальный идентификатор записи.", "purpose": "Связь записи в базе и отчетах."},
    "subject": {"type": "string", "description": "Название дисциплины или общего набора.", "purpose": "Группировка материалов по предмету."},
    "topic": {"type": "string", "description": "Название темы учебного материала.", "purpose": "Построение последовательности изучения."},
    "lesson_type": {"type": "string", "description": "Тип занятия: лекция, практика, автогенерация и т.д.", "purpose": "Анализ формата обучения."},
    "source_path": {"type": "string", "description": "Путь к файлу или набору файлов-источников.", "purpose": "Проверка происхождения материала."},
    "source_kind": {"type": "string", "description": "Тип исходного материала или набора материалов.", "purpose": "Анализ структуры входных данных."},
    "text_material": {"type": "text", "description": "Нормализованный текст, извлеченный из входных файлов.", "purpose": "Основа для анализа, модерации и кластеризации."},
    "summary": {"type": "text", "description": "Краткое автоматическое описание материала.", "purpose": "Быстрый обзор содержимого."},
    "moderation_conclusion": {"type": "string", "description": "Итог автоматической модерации.", "purpose": "Контроль качества учебного материала."},
    "requirement_checks": {"type": "list[dict]", "description": "Результаты проверки методических требований.", "purpose": "Подробное обоснование модерации."},
    "media_descriptions": {"type": "list[string]", "description": "Текстовые описания фото и видео.", "purpose": "Учет мультимедиа в общем материале."},
    "generated": {"type": "boolean", "description": "Признак автоматически сгенерированного материала.", "purpose": "Разделение исходных и дополненных данных."},
    "topic_order": {"type": "integer", "description": "Порядковый номер темы в траектории.", "purpose": "Построение предыдущих и следующих материалов."},
    "previous_record_id": {"type": "string|null", "description": "Идентификатор предыдущего материала по теме.", "purpose": "Формирование связей непрерывного обучения."},
    "next_record_id": {"type": "string|null", "description": "Идентификатор следующего материала по теме.", "purpose": "Формирование связей непрерывного обучения."},
    "word_count": {"type": "integer", "description": "Количество слов в текстовом материале.", "purpose": "Оценка объема и сложности."},
    "char_count": {"type": "integer", "description": "Количество символов в тексте.", "purpose": "Анализ длины материала."},
    "methodical_score": {"type": "integer", "description": "Сумма успешно пройденных методических проверок.", "purpose": "Оценка качества учебного материала."},
    "difficulty_level": {"type": "string", "description": "Группа сложности материала.", "purpose": "Планирование траектории и кластеризация."},
    "parallel_cluster": {"type": "string", "description": "Кластер для параллельного изучения.", "purpose": "Группировка схожих материалов."},
    "sequential_cluster": {"type": "string", "description": "Кластер последовательного изучения.", "purpose": "Построение линейной траектории."},
    "estimated_minutes": {"type": "integer", "description": "Оценка времени на изучение материала.", "purpose": "Планирование учебного расписания."},
    "content_hash": {"type": "string", "description": "Хэш содержимого материала.", "purpose": "Поиск дублей и контроль версии."},
    "updated_at_utc": {"type": "datetime", "description": "Время формирования записи.", "purpose": "Аудит и контроль актуальности."},
}

METHODICAL_GUIDELINES: list[dict[str, str]] = [
    {"code": "MR-3.1", "category": "Методические", "description": "Материал должен быть последовательным, завершенным и логичным."},
    {"code": "MR-3.3", "category": "Методические", "description": "Материал должен иметь практико-ориентированную направленность и профессиональный контекст."},
    {"code": "MR-3.9", "category": "Методические", "description": "При необходимости должны присутствовать междисциплинарные связи и задания."},
    {"code": "MR-4.1", "category": "Методические", "description": "Содержание должно соответствовать теме и дидактическому каркасу."},
    {"code": "MR-4.6", "category": "Методические", "description": "Текст и мультимедийные фрагменты должны соответствовать нормам современного русского языка."},
    {"code": "MR-5.1", "category": "Технические", "description": "В материале не должно быть рекламы, водяных знаков и посторонних надписей."},
    {"code": "MR-5.9", "category": "Технические", "description": "Недопустимы рекламные и нерелевантные сторонние ссылки."},
]


def build_module_a() -> dict:
    """Собираем единый датасет, анализируем структуру и при необходимости дополняем его новыми материалами."""
    ensure_workspace()
    reset_parse_warnings()

    # Шаг 1. Пользователь вручную кладет исходные файлы в Data/Incoming/module_a,
    # а агент автоматически сканирует папки и формирует список источников.
    manifest = load_sources_for_module_a()
    scan_stats = dict(manifest.get("scan_stats", {}))

    # Шаг 2. Из каждого набора файлов темы собираем единую запись материала.
    materials: list[Material] = [build_material_record(source) for source in manifest["sources"]]
    scan_stats["materials_before_dedup"] = len(materials)
    materials = deduplicate_materials(materials)
    scan_stats["materials_after_dedup"] = len(materials)

    # Шаг 3. Анализируем непрерывность предмета и предлагаем добавить недостающие темы.
    generation_candidates = collect_generation_candidates(manifest, materials)
    suggestions: list[dict] = []
    if generation_candidates:
        selected_topics = choose_generation_topics(generation_candidates)
        for candidate in generation_candidates:
            should_generate = candidate["topic_key"] in selected_topics
            suggestions.append(
                {
                    "subject": candidate["subject"],
                    "topic": candidate["topic"],
                    "reason": candidate["reason"],
                    "generated": should_generate,
                }
            )
            if should_generate:
                generated_source, generated_text = generate_missing_material(
                    candidate["subject"],
                    candidate["topic"],
                    candidate["topic_order"],
                )
                generated_material = build_material_record(generated_source, generated=True, generated_text=generated_text)
                materials.append(generated_material)

    materials = deduplicate_materials(materials)
    scan_stats["generated_materials_count"] = sum(1 for item in materials if item.generated)
    scan_stats["materials_saved_to_database"] = len(materials)
    scan_stats["suggestions_count"] = len(suggestions)
    parse_warnings = get_parse_warnings()
    scan_stats["fallback_files_count"] = len(parse_warnings)
    format_stats = build_format_statistics(manifest, materials, parse_warnings)

    # Шаг 4. Автоматически добавляем признаки наличия предыдущего и следующего материала.
    attach_neighbors(materials)

    # Шаг 5. Выполняем анализ атрибутов, влияющих на схожесть, и полное описание структуры данных.
    similarity_analysis = analyze_similarity_factors(materials)
    attribute_analysis = analyze_dataset_attributes(materials)
    analytical_briefs = build_analytical_briefs(manifest, materials)

    database = {
        "sources": manifest["sources"],
        "materials": [material.to_dict() for material in materials],
        "clusters": {},
        "models": {},
        "module_a_stats": scan_stats,
        "module_a_parse_warnings": parse_warnings,
        "module_a_format_stats": format_stats,
        "analytical_briefs": analytical_briefs,
        "methodical_guidelines": METHODICAL_GUIDELINES,
        "suggestions": suggestions,
        "runs": [
            {
                "module": "A",
                "started_at_utc": utc_now(),
                "result": "OK",
                "details": "Сбор, обработка, модерация, анализ атрибутов и дополнение набора данных завершены.",
            }
        ],
    }
    save_database(database)
    save_exports(materials, attribute_analysis, similarity_analysis, scan_stats, parse_warnings, format_stats, analytical_briefs)
    save_report(materials, suggestions, similarity_analysis, attribute_analysis, scan_stats, parse_warnings, format_stats, analytical_briefs)
    save_analytical_brief_files(analytical_briefs)
    return database


def collect_generation_candidates(manifest: dict, materials: list[Material]) -> list[dict]:
    """Ищем отсутствующие темы по программе и возможные разрывы по порядку изучения."""
    existing_topics = {(material.subject, material.topic) for material in materials}
    candidates: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for subject, syllabus in manifest["syllabus"].items():
        for item in syllabus:
            key = (subject, item["topic"])
            if key in existing_topics or key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "subject": subject,
                    "topic": item["topic"],
                    "topic_key": f"{subject}::{item['topic']}",
                    "topic_order": item["topic_order"],
                    "reason": "Тема есть в программе, но отсутствует среди загруженных материалов.",
                }
            )

    orders_by_subject: dict[str, list[int]] = defaultdict(list)
    for material in materials:
        orders_by_subject[material.subject].append(material.topic_order)
    for subject, orders in orders_by_subject.items():
        explicit_orders = sorted({order for order in orders if 1 <= order < 900})
        for left, right in zip(explicit_orders, explicit_orders[1:]):
            if right - left <= 1 or right - left > 5:
                continue
            for missing_order in range(left + 1, right):
                topic = f"Промежуточная тема {missing_order}"
                key = (subject, topic)
                if key in existing_topics or key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(
                    {
                        "subject": subject,
                        "topic": topic,
                        "topic_key": f"{subject}::{topic}",
                        "topic_order": missing_order,
                        "reason": "Между соседними темами найден разрыв в учебной последовательности.",
                    }
                )
    counts_by_subject = Counter(material.subject for material in materials if material.subject)
    if counts_by_subject:
        average_count = mean(counts_by_subject.values())
        for subject, count in counts_by_subject.items():
            if count >= average_count:
                continue
            subject_orders = sorted(
                material.topic_order
                for material in materials
                if material.subject == subject and 1 <= material.topic_order < 900
            )
            next_order = subject_orders[-1] + 1 if subject_orders else 1
            topic = f"Дополнительная тема {next_order}"
            key = (subject, topic)
            if key in existing_topics or key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "subject": subject,
                    "topic": topic,
                    "topic_key": f"{subject}::{topic}",
                    "topic_order": next_order,
                    "reason": "Дисциплина представлена ниже среднего уровня, поэтому агент предлагает дополнительный материал для выравнивания покрытия.",
                }
            )
    return sorted(candidates, key=lambda item: (item["subject"], item["topic_order"], item["topic"]))


def choose_generation_topics(candidates: list[dict]) -> set[str]:
    """Предлагаем пользователю выбрать конкретные темы для генерации, а не только режим all/none."""
    env_value = os.environ.get(AUTOGENERATE_ENV, "").strip().lower()
    all_topics = {candidate["topic_key"] for candidate in candidates}
    if env_value in {"1", "true", "yes", "y", "да", "all"}:
        print("Автогенерация включена через переменную окружения MODULE_A_AUTOGENERATE.")
        return all_topics
    if env_value in {"0", "false", "no", "n", "нет"}:
        print("Автогенерация отключена через переменную окружения MODULE_A_AUTOGENERATE.")
        return set()

    print(f"Найдены темы для дополнения набора данных: {len(candidates)}")
    for index, candidate in enumerate(candidates[:50], start=1):
        print(f"{index}. {candidate['subject']} / {candidate['topic']} -> {candidate['reason']}")
    if len(candidates) > 20:
        print(f"- ... и еще {len(candidates) - 20} тем.")

    if sys.stdin.isatty():
        answer = input("Введите номера тем через запятую, 'all' для всех тем или 'none' для отказа: ").strip().lower()
        if answer in {"", "all", "все"}:
            return all_topics
        if answer in {"none", "no", "n", "нет"}:
            return set()
        selected: set[str] = set()
        for part in answer.split(","):
            cleaned = part.strip()
            if cleaned.isdigit():
                index = int(cleaned)
                if 1 <= index <= len(candidates):
                    selected.add(candidates[index - 1]["topic_key"])
        return selected

    print("Интерактивный ввод недоступен, поэтому недостающие материалы будут сгенерированы автоматически.")
    return all_topics


def analyze_similarity_factors(materials: list[Material]) -> list[dict]:
    """Оцениваем значимость признаков реальными методами: корреляцией Пирсона и permutation importance."""
    if len(materials) < 3:
        return []

    vectors = {material.record_id: text_vector(material.text_material) for material in materials}
    pair_rows: list[dict[str, float]] = []
    for index, left in enumerate(materials):
        for right in materials[index + 1 :]:
            pair_rows.append(
                {
                    "target_similarity": cosine_similarity(vectors[left.record_id], vectors[right.record_id]),
                    "same_subject": 1.0 if left.subject == right.subject else 0.0,
                    "same_lesson_type": 1.0 if left.lesson_type == right.lesson_type else 0.0,
                    "same_source_kind": 1.0 if left.source_kind == right.source_kind else 0.0,
                    "same_difficulty_level": 1.0 if left.difficulty_level == right.difficulty_level else 0.0,
                    "topic_order_distance": float(abs(left.topic_order - right.topic_order)),
                    "word_count_gap": float(abs(left.word_count - right.word_count)),
                    "char_count_gap": float(abs(left.char_count - right.char_count)),
                    "methodical_score_gap": float(abs(left.methodical_score - right.methodical_score)),
                    "media_count_gap": float(abs(len(left.media_descriptions) - len(right.media_descriptions))),
                    "topic_keyword_overlap": float(len(set(tokenize(left.topic)) & set(tokenize(right.topic)))),
                    "neighbor_link": 1.0 if left.next_record_id == right.record_id or right.next_record_id == left.record_id else 0.0,
                }
            )

    if len(pair_rows) < 4:
        return []

    feature_names = [name for name in pair_rows[0].keys() if name != "target_similarity"]
    x = np.array([[row[name] for name in feature_names] for row in pair_rows], dtype=float)
    y = np.array([row["target_similarity"] for row in pair_rows], dtype=float)

    estimator = RandomForestRegressor(n_estimators=240, random_state=42)
    estimator.fit(x, y)
    importance_result = permutation_importance(estimator, x, y, random_state=42, n_repeats=12)
    importance_map = {name: float(importance_result.importances_mean[index]) for index, name in enumerate(feature_names)}

    rows: list[dict] = []
    for index, feature_name in enumerate(feature_names):
        values = x[:, index]
        correlation = pearson_correlation(values.tolist(), y.tolist())
        rows.append(
            {
                "factor_code": feature_name,
                "factor_name": explain_similarity_factor(feature_name),
                "pearson_correlation": round(correlation, 3),
                "permutation_importance": round(importance_map[feature_name], 4),
                "influence_score": round(abs(correlation) + max(0.0, importance_map[feature_name]), 4),
                "analysis_method": "pearson_correlation + permutation_importance",
                "conclusion": build_similarity_conclusion(correlation, importance_map[feature_name]),
            }
        )
    return sorted(rows, key=lambda item: item["influence_score"], reverse=True)


def pearson_correlation(left: list[float], right: list[float]) -> float:
    # Реализуем коэффициент Пирсона вручную, чтобы отчет опирался на реальный корреляционный анализ.
    if not left or not right or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((l_value - left_mean) * (r_value - right_mean) for l_value, r_value in zip(left, right))
    left_denominator = sqrt(sum((value - left_mean) ** 2 for value in left))
    right_denominator = sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_denominator == 0 or right_denominator == 0:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def explain_similarity_factor(feature_name: str) -> str:
    mapping = {
        "same_subject": "Совпадение дисциплины",
        "same_lesson_type": "Совпадение типа занятия",
        "same_source_kind": "Совпадение формата источника",
        "same_difficulty_level": "Совпадение уровня сложности",
        "topic_order_distance": "Расстояние между темами по порядку изучения",
        "word_count_gap": "Разница по количеству слов",
        "char_count_gap": "Разница по длине текста",
        "methodical_score_gap": "Разница по методическому баллу",
        "media_count_gap": "Разница по числу медиаобъектов",
        "topic_keyword_overlap": "Пересечение ключевых слов в теме",
        "neighbor_link": "Прямая связь предыдущий/следующий материал",
    }
    return mapping.get(feature_name, feature_name)


def build_similarity_conclusion(correlation: float, importance: float) -> str:
    if abs(correlation) >= 0.35 or importance >= 0.03:
        return "Признак значим для группировки и должен использоваться при кластеризации."
    if abs(correlation) >= 0.15 or importance >= 0.01:
        return "Признак умеренно влияет на схожесть и используется как дополнительный."
    return "Признак слабый и играет вспомогательную роль."


def analyze_dataset_attributes(materials: list[Material]) -> dict[str, list[dict]]:
    """Готовим полную статистику по каждому атрибуту: тип, описание, уникальные значения, частоты и текстовые метрики."""
    rows = [material.to_dict() for material in materials]
    attribute_rows: list[dict] = []
    frequency_rows: list[dict] = []
    text_rows: list[dict] = []

    for field_name, meta in ATTRIBUTE_CATALOG.items():
        values = [row[field_name] for row in rows]
        prepared_values = [serialize_value(value) for value in values]
        counter = Counter(prepared_values)
        unique_count = len(counter)
        top_value, top_count = counter.most_common(1)[0] if counter else ("", 0)
        attribute_rows.append(
            {
                "attribute": field_name,
                "data_type": meta["type"],
                "description": meta["description"],
                "purpose": meta["purpose"],
                "unique_values": unique_count,
                "most_common_value": top_value,
                "most_common_count": top_count,
            }
        )

        for value, count in counter.most_common(10):
            frequency_rows.append(
                {
                    "attribute": field_name,
                    "value": value,
                    "count": count,
                    "share": round(count / max(1, len(values)), 3),
                }
            )

        if meta["type"] == "text":
            lengths = [len(str(value)) for value in values]
            word_counts = [len(tokenize(str(value))) for value in values]
            text_rows.append(
                {
                    "attribute": field_name,
                    "average_chars": round(mean(lengths), 1) if lengths else 0,
                    "min_chars": min(lengths) if lengths else 0,
                    "max_chars": max(lengths) if lengths else 0,
                    "average_words": round(mean(word_counts), 1) if word_counts else 0,
                    "min_words": min(word_counts) if word_counts else 0,
                    "max_words": max(word_counts) if word_counts else 0,
                }
            )

    return {
        "attribute_rows": attribute_rows,
        "frequency_rows": frequency_rows,
        "text_rows": text_rows,
    }


def build_format_statistics(
    manifest: dict,
    materials: list[Material],
    parse_warnings: list[dict[str, str]],
) -> dict[str, object]:
    # Считаем, сколько файлов каждого формата реально было загружено и сколько медиаобъектов
    # обнаружено внутри документов и презентаций.
    format_counter: Counter[str] = Counter()
    embedded_media_count = 0
    standalone_media_count = 0
    for source in manifest.get("sources", []):
        for path in source.get("content_paths", []):
            suffix = Path(path).suffix.lower().lstrip(".") or "unknown"
            format_counter[suffix] += 1
        for path in source.get("media_paths", []):
            suffix = Path(path).suffix.lower().lstrip(".") or "unknown"
            format_counter[suffix] += 1
            standalone_media_count += 1
    for material in materials:
        embedded_media_count += sum(1 for item in material.media_descriptions if "Встроенное изображение" in item)
    return {
        "by_format": dict(sorted(format_counter.items())),
        "embedded_media_count": embedded_media_count,
        "standalone_media_count": standalone_media_count,
        "fallback_files_count": len(parse_warnings),
    }


def build_analytical_briefs(manifest: dict, materials: list[Material]) -> list[dict[str, object]]:
    # Готовим отдельную аналитическую справку по каждому загруженному документу/источнику.
    material_by_source = {material.source_path: material for material in materials}
    briefs: list[dict[str, object]] = []
    for source in manifest.get("sources", []):
        content_paths = source.get("content_paths") or ([source["file_path"]] if source.get("file_path") else [])
        source_path = " | ".join(content_paths) if content_paths else " | ".join(source.get("media_paths", []))
        material = material_by_source.get(source_path)
        if material is None:
            continue
        briefs.append(
            {
                "record_id": material.record_id,
                "subject": material.subject,
                "topic": material.topic,
                "source_path": material.source_path,
                "source_kind": material.source_kind,
                "summary": material.summary,
                "moderation_conclusion": material.moderation_conclusion,
                "methodical_score": material.methodical_score,
                "checked_rules_count": len(material.requirement_checks),
                "passed_rules_count": sum(1 for item in material.requirement_checks if item["passed"]),
                "media_objects_count": len(material.media_descriptions),
                "word_count": material.word_count,
                "char_count": material.char_count,
                "detailed_checks": material.requirement_checks,
            }
        )
    return briefs


def serialize_value(value: object) -> str:
    """Приводим любые значения атрибутов к строке для сводной статистики."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"list[{len(value)}]_dict"
        return " | ".join(str(item) for item in value) if value else "EMPTY_LIST"
    if isinstance(value, dict):
        return compute_hash(str(sorted(value.items())))
    return str(value)


def save_exports(
    materials: list[Material],
    attribute_analysis: dict[str, list[dict]],
    similarity_analysis: list[dict],
    scan_stats: dict[str, object],
    parse_warnings: list[dict[str, str]],
    format_stats: dict[str, object],
    analytical_briefs: list[dict[str, object]],
) -> None:
    """Выгружаем основной датасет и отдельные таблицы анализа для проверки критериев."""
    dataset_rows = [
        {
            "record_id": item.record_id,
            "subject": item.subject,
            "topic": item.topic,
            "lesson_type": item.lesson_type,
            "source_path": item.source_path,
            "source_kind": item.source_kind,
            "summary": item.summary,
            "moderation_conclusion": item.moderation_conclusion,
            "generated": item.generated,
            "topic_order": item.topic_order,
            "previous_record_id": item.previous_record_id or "",
            "next_record_id": item.next_record_id or "",
            "word_count": item.word_count,
            "char_count": item.char_count,
            "methodical_score": item.methodical_score,
            "difficulty_level": item.difficulty_level,
            "estimated_minutes": item.estimated_minutes,
        }
        for item in materials
    ]
    save_csv(EXPORTS_DIR / "dataset_module_a.csv", dataset_rows)
    save_csv(EXPORTS_DIR / "module_a_attribute_catalog.csv", attribute_analysis["attribute_rows"])
    save_csv(EXPORTS_DIR / "module_a_attribute_frequency.csv", attribute_analysis["frequency_rows"])
    save_csv(EXPORTS_DIR / "module_a_text_stats.csv", attribute_analysis["text_rows"])
    save_csv(EXPORTS_DIR / "module_a_similarity_factors.csv", similarity_analysis)
    save_csv(
        EXPORTS_DIR / "module_a_ingestion_stats.csv",
        [{"metric": key, "value": value} for key, value in scan_stats.items()],
    )
    save_csv(EXPORTS_DIR / "module_a_parse_warnings.csv", parse_warnings)
    save_csv(
        EXPORTS_DIR / "module_a_format_stats.csv",
        [
            {"metric": "embedded_media_count", "value": format_stats.get("embedded_media_count", 0)},
            {"metric": "standalone_media_count", "value": format_stats.get("standalone_media_count", 0)},
            {"metric": "fallback_files_count", "value": format_stats.get("fallback_files_count", 0)},
            *[
                {"metric": f"format_{format_name}", "value": count}
                for format_name, count in format_stats.get("by_format", {}).items()
            ],
        ],
    )
    save_csv(EXPORTS_DIR / "module_a_analytical_briefs.csv", analytical_briefs)
    save_charts(materials, attribute_analysis, similarity_analysis)


def save_charts(materials: list[Material], attribute_analysis: dict[str, list[dict]], similarity_analysis: list[dict]) -> None:
    """Строим простые SVG-графики без внешних библиотек, чтобы они работали офлайн."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows = [material.to_dict() for material in materials]
    for attribute in ("subject", "lesson_type", "source_kind", "difficulty_level", "moderation_conclusion", "generated"):
        counter = Counter(serialize_value(row[attribute]) for row in dataset_rows)
        write_bar_chart_svg(CHARTS_DIR / f"{attribute}_distribution.svg", f"Распределение: {attribute}", list(counter.items())[:10])
    text_pairs = [(row["attribute"], row["average_words"]) for row in attribute_analysis["text_rows"]]
    write_bar_chart_svg(CHARTS_DIR / "text_attribute_words.svg", "Среднее число слов в текстовых полях", text_pairs)
    similarity_pairs = [(row["factor_name"], row["influence_score"]) for row in similarity_analysis]
    write_bar_chart_svg(CHARTS_DIR / "similarity_factors.svg", "Влияние признаков на схожесть", similarity_pairs)


def write_bar_chart_svg(path: Path, title: str, pairs: list[tuple[str, float]]) -> None:
    """Создаем автономный SVG-график, чтобы его можно было открыть без дополнительных пакетов."""
    if not pairs:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'></svg>", encoding="utf-8")
        return

    width = 900
    row_height = 36
    top = 60
    left = 280
    bar_area = 560
    max_value = max(abs(float(value)) for _, value in pairs) or 1.0
    height = top + row_height * len(pairs) + 30
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Segoe UI,sans-serif;fill:#21313c}.label{font-size:13px}.title{font-size:20px;font-weight:700}.value{font-size:12px}</style>",
        f"<rect width='{width}' height='{height}' fill='#fcfaf4'/>",
        f"<text x='24' y='34' class='title'>{escape_xml(title)}</text>",
    ]
    for index, (label, raw_value) in enumerate(pairs):
        y = top + index * row_height
        value = float(raw_value)
        bar_width = 0 if max_value == 0 else abs(value) / max_value * bar_area
        fill = "#1f8f6a" if value >= 0 else "#c85d48"
        lines.append(f"<text x='24' y='{y + 18}' class='label'>{escape_xml(str(label)[:38])}</text>")
        lines.append(f"<rect x='{left}' y='{y + 4}' width='{bar_area}' height='18' rx='9' fill='#e5ece6'/>")
        lines.append(f"<rect x='{left}' y='{y + 4}' width='{bar_width:.1f}' height='18' rx='9' fill='{fill}'/>")
        lines.append(f"<text x='{left + bar_area + 8}' y='{y + 18}' class='value'>{value:.3f}</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def save_report(
    materials: list[Material],
    suggestions: list[dict],
    similarity_analysis: list[dict],
    attribute_analysis: dict[str, list[dict]],
    scan_stats: dict[str, object],
    parse_warnings: list[dict[str, str]],
    format_stats: dict[str, object],
    analytical_briefs: list[dict[str, object]],
) -> None:
    """Собираем итоговый отчет по модулю A и отдельные аналитические документы."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter(material.subject for material in materials)

    report_lines = [
        "# Отчет по модулю A",
        "",
        "## Что делает агент",
        "- Пользователь вручную помещает исходные файлы в `Data/Incoming/module_a/`, а агент автоматически сканирует папки и читает материалы.",
        "- Поддерживается загрузка текстов, документов, PDF, Excel, фотографий и видео.",
        "- Методические рекомендации, полученные из задания, зашиты в агент как набор проверяемых правил модерации и применяются ко всем материалам.",
        "- В одной папке темы можно хранить сразу несколько файлов разных типов, а агент объединяет их в один учебный материал.",
        "- Агент автоматически добавляет признаки `previous_record_id` и `next_record_id` по тематической последовательности.",
        "- Агент предлагает дополнить набор недостающими материалами и при подтверждении генерирует их, проводит модерацию и записывает в базу.",
        "- По каждому документу формируется отдельная аналитическая справка, которая сохраняется в SQLite и выгружается в отдельный CSV.",
        "",
        "## Как добавляются данные",
        "- Исходные файлы добавляются вручную в `Data/Incoming/module_a/`.",
        "- Обработка, извлечение текста, модерация, построение признаков и формирование SQLite-базы выполняются автоматически.",
        "",
        "## Статистика загрузки файлов",
        f"- Всего файлов найдено во входной папке: **{scan_stats.get('total_files_found', 0)}**.",
        f"- Поддерживаемых файлов найдено: **{scan_stats.get('supported_files_found', 0)}**.",
        f"- Игнорируемых служебных файлов: **{scan_stats.get('ignored_files_count', 0)}**.",
        f"- Сформировано групп-источников: **{scan_stats.get('source_groups_created', 0)}**.",
        f"- Материалов до удаления дублей: **{scan_stats.get('materials_before_dedup', 0)}**.",
        f"- Материалов после удаления дублей: **{scan_stats.get('materials_after_dedup', 0)}**.",
        f"- Материалов сохранено в SQLite: **{scan_stats.get('materials_saved_to_database', 0)}**.",
        f"- Файлов, принятых через резервный fallback-режим: **{scan_stats.get('fallback_files_count', 0)}**.",
        f"- Встроенных изображений и медиа внутри офисных файлов обнаружено: **{format_stats.get('embedded_media_count', 0)}**.",
        f"- Отдельных внешних медиафайлов обнаружено: **{format_stats.get('standalone_media_count', 0)}**.",
        "",
        "## Количество обработанных форматов",
        *[
            f"- Формат `{format_name}`: **{count}** файлов."
            for format_name, count in format_stats.get("by_format", {}).items()
        ],
        "",
        "## Методические рекомендации, используемые агентом",
        *[f"- {item['code']} ({item['category']}): {item['description']}" for item in METHODICAL_GUIDELINES],
        "",
        "## Отказоустойчивая обработка некорректных Office-файлов",
        "- Файлы `.docx`, `.xlsx`, `.xlsm` и `.pptx`, которые не являются корректными OpenXML-архивами, не останавливают работу агента.",
        "- В таких случаях модуль автоматически переходит на резервную обработку: сохраняет предупреждение, формирует техническое описание источника и продолжает сбор датасета.",
        "- Это позволяет принять поврежденные файлы или старые бинарные документы, ошибочно переименованные в современные расширения.",
        "- Полный список файлов, обработанных через fallback, сохраняется в `Data/Exports/module_a_parse_warnings.csv`.",
        "",
        "## Проверка 1.3: анализ атрибутов, влияющих на схожесть",
        "- Результаты сохранены в `Data/Exports/module_a_similarity_factors.csv` и `Docs/Reports/module_a_similarity_analysis.md`.",
        "- Используется реальный корреляционный анализ Пирсона и permutation importance на базе `RandomForestRegressor`.",
        "- Самыми сильными факторами считаются признаки с наибольшим `influence_score`.",
        "",
        "## Проверка 1.4: описание атрибутов и статистика",
        "- Каталог атрибутов сохранен в `Data/Exports/module_a_attribute_catalog.csv`.",
        "- Частоты значений сохранены в `Data/Exports/module_a_attribute_frequency.csv`.",
        "- Метрики текстовых полей сохранены в `Data/Exports/module_a_text_stats.csv`.",
        "- Графики сохранены в `Docs/Reports/charts/` в формате SVG.",
        "",
        "## Проверка 1.5: генерация недостающих материалов",
    ]
    if suggestions:
        for suggestion in suggestions:
            status = "сгенерировано и добавлено в базу" if suggestion["generated"] else "предложено пользователю, но не сгенерировано"
            report_lines.append(f"- {suggestion['subject']} / {suggestion['topic']}: {suggestion['reason']} -> {status}.")
    else:
        report_lines.append("- Пробелов в последовательности тем не обнаружено.")

    report_lines.extend(
        [
            "",
            "## Аналитические справки по документам",
            f"- Всего аналитических справок сформировано: **{len(analytical_briefs)}**.",
            "- Выгрузка справок находится в `Data/Exports/module_a_analytical_briefs.csv`.",
            "- Отдельные markdown-справки лежат в `Docs/Reports/module_a_briefs/`.",
            "- Каждая справка содержит `record_id`, дисциплину, тему, путь к документу, итог модерации, методический балл, число пройденных проверок и список детальных проверок.",
            "",
            "## Частоты материалов по дисциплинам",
            *[f"- {subject}: {count} материалов." for subject, count in sorted(counts.items())],
            "",
            "## Файлы, обработанные через fallback",
        ]
    )
    if parse_warnings:
        for item in parse_warnings[:20]:
            report_lines.append(f"- {item['file_path']} -> {item['warning_code']}: {item['message']}")
        if len(parse_warnings) > 20:
            report_lines.append(f"- Дополнительно случаев fallback: {len(parse_warnings) - 20}. Полный список находится в `Data/Exports/module_a_parse_warnings.csv`.")
    else:
        report_lines.append("- Некорректных Office-файлов в текущем запуске не обнаружено.")

    report_lines.extend(
        [
            "",
            "## Краткие текстовые характеристики",
            f"- Средняя длина текста: {sum(item.char_count for item in materials) / max(1, len(materials)):.1f} символов.",
            f"- Минимальная длина текста: {min(item.char_count for item in materials)} символов.",
            f"- Максимальная длина текста: {max(item.char_count for item in materials)} символов.",
            f"- Среднее число слов: {sum(item.word_count for item in materials) / max(1, len(materials)):.1f}.",
            "",
            "## Где выполнены критерии",
            *[f"- {line}" for line in criteria_lines("A")],
        ]
    )
    (REPORTS_DIR / "module_a_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    save_similarity_report(similarity_analysis)
    save_attribute_report(attribute_analysis)


def save_similarity_report(similarity_analysis: list[dict]) -> None:
    lines = [
        "# Анализ факторов схожести материалов",
        "",
        "Анализ выполнен реальными методами: коэффициент Пирсона и permutation importance на модели `RandomForestRegressor`.",
        "",
        "| Фактор | Корреляция Пирсона | Permutation importance | Интегральное влияние | Метод | Вывод |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in similarity_analysis:
        lines.append(
            f"| {item['factor_name']} | {item['pearson_correlation']} | {item['permutation_importance']} | {item['influence_score']} | {item['analysis_method']} | {item['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "График: `Docs/Reports/charts/similarity_factors.svg`.",
        ]
    )
    (REPORTS_DIR / "module_a_similarity_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def save_attribute_report(attribute_analysis: dict[str, list[dict]]) -> None:
    lines = [
        "# Анализ структуры набора данных",
        "",
        "## Каталог атрибутов",
        "| Атрибут | Тип | Описание | Назначение | Уникальных значений | Самое частое значение | Частота |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in attribute_analysis["attribute_rows"]:
        lines.append(
            f"| {row['attribute']} | {row['data_type']} | {row['description']} | {row['purpose']} | {row['unique_values']} | {row['most_common_value']} | {row['most_common_count']} |"
        )

    lines.extend(
        [
            "",
            "## Текстовые атрибуты",
            "| Атрибут | Ср. символов | Мин. символов | Макс. символов | Ср. слов | Мин. слов | Макс. слов |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in attribute_analysis["text_rows"]:
        lines.append(
            f"| {row['attribute']} | {row['average_chars']} | {row['min_chars']} | {row['max_chars']} | {row['average_words']} | {row['min_words']} | {row['max_words']} |"
        )

    lines.extend(
        [
            "",
            "## Файлы со сводными таблицами и графиками",
            "- `Data/Exports/module_a_attribute_catalog.csv`",
            "- `Data/Exports/module_a_attribute_frequency.csv`",
            "- `Data/Exports/module_a_text_stats.csv`",
            "- `Docs/Reports/charts/subject_distribution.svg`",
            "- `Docs/Reports/charts/lesson_type_distribution.svg`",
            "- `Docs/Reports/charts/source_kind_distribution.svg`",
            "- `Docs/Reports/charts/difficulty_level_distribution.svg`",
            "- `Docs/Reports/charts/moderation_conclusion_distribution.svg`",
            "- `Docs/Reports/charts/generated_distribution.svg`",
            "- `Docs/Reports/charts/text_attribute_words.svg`",
            "- `Data/Exports/module_a_ingestion_stats.csv`",
            "- `Data/Exports/module_a_format_stats.csv`",
            "- `Data/Exports/module_a_parse_warnings.csv`",
            "- `Data/Exports/module_a_analytical_briefs.csv`",
        ]
    )
    (REPORTS_DIR / "module_a_attribute_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def save_analytical_brief_files(analytical_briefs: list[dict[str, object]]) -> None:
    # Для защиты дополнительно сохраняем по одной markdown-справке на каждый материал.
    briefs_dir = REPORTS_DIR / "module_a_briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    for brief in analytical_briefs:
        safe_name = str(brief["record_id"]).replace("/", "_")
        lines = [
            f"# Аналитическая справка {brief['record_id']}",
            "",
            f"- Дисциплина: **{brief['subject']}**",
            f"- Тема: **{brief['topic']}**",
            f"- Источник: `{brief['source_path']}`",
            f"- Тип источника: `{brief['source_kind']}`",
            f"- Итог модерации: **{brief['moderation_conclusion']}**",
            f"- Методический балл: **{brief['methodical_score']}**",
            f"- Пройдено требований: **{brief['passed_rules_count']} из {brief['checked_rules_count']}**",
            f"- Количество медиаобъектов: **{brief['media_objects_count']}**",
            f"- Размер текста: **{brief['word_count']} слов / {brief['char_count']} символов**",
            "",
            "## Краткая аннотация",
            str(brief["summary"]),
            "",
            "## Детальные результаты проверки",
        ]
        for check in brief["detailed_checks"]:
            status = "Пройдено" if check["passed"] else "Не пройдено"
            lines.append(f"- {check['code']} [{check['category']}] {check['requirement']} -> {status}. {check['comment']}")
        (briefs_dir / f"{safe_name}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_module_a()
    print("Модуль A завершен.")
    print(f"Манифест источников: {MANIFEST_PATH}")
    print(f"Экспорт набора данных: {EXPORTS_DIR / 'dataset_module_a.csv'}")
    print(f"Отчет: {REPORTS_DIR / 'module_a_report.md'}")
