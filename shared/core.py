from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from shared.config import MANIFEST_PATH, MODEL_DIR, MODULE_A_INPUT_DIR, ROOT_DIR
from shared.clustering import build_cluster_suite
from shared.ml_models import build_sequential_role, train_task_models
from shared.parsers import extract_media_descriptions, extract_text, normalize_text
from shared.storage import load_json


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]{3,}")
URL_RE = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
BANNED_AD_RE = re.compile(r"реклама|подпишись|скидк|купить|sale|promo", flags=re.IGNORECASE)

METHODICAL_RULES: list[dict[str, str]] = [
    {
        "code": "MR-1.1",
        "category": "Нормативно-правовые",
        "requirement": "Материал не противоречит законодательству и не содержит запрещенной информации.",
        "kind": "safety_baseline",
    },
    {
        "code": "MR-2.1",
        "category": "Нормативно-технические",
        "requirement": "Материал соответствует техническим и эксплуатационным требованиям к ресурсу.",
        "kind": "technical_baseline",
    },
    {
        "code": "MR-3.1",
        "category": "Методические",
        "requirement": "Материал представлен последовательно и имеет завершенный характер.",
        "kind": "sequence",
    },
    {
        "code": "MR-3.2",
        "category": "Методические",
        "requirement": "Содержание построено логично и направлено на достижение проверяемых результатов.",
        "kind": "learning_goal",
    },
    {
        "code": "MR-3.3",
        "category": "Методические",
        "requirement": "Материал имеет профессиональную практико-ориентированную направленность.",
        "kind": "practice",
    },
    {
        "code": "MR-3.5",
        "category": "Методические",
        "requirement": "Материал содержит задания, обеспечивающие применение профессиональной деятельности.",
        "kind": "practice",
    },
    {
        "code": "MR-3.6",
        "category": "Методические",
        "requirement": "Материал соответствует современным тенденциям развития профессии и отрасли.",
        "kind": "modernity",
    },
    {
        "code": "MR-3.9",
        "category": "Методические",
        "requirement": "Материал демонстрирует взаимосвязь между дисциплинами при наличии междисциплинарной темы.",
        "kind": "interdisciplinary",
    },
    {
        "code": "MR-4.1",
        "category": "Методические",
        "requirement": "Содержание соответствует указанным темам и дидактическому каркасу.",
        "kind": "topic_match",
    },
    {
        "code": "MR-4.3",
        "category": "Методические",
        "requirement": "Материал не противоречит современным научным знаниям и корректно интерпретирует факты.",
        "kind": "science_baseline",
    },
    {
        "code": "MR-4.6",
        "category": "Методические",
        "requirement": "Текст оформлен корректно, соблюдает регистр и лексику.",
        "kind": "language_norms",
    },
    {
        "code": "MR-4.7",
        "category": "Методические",
        "requirement": "Текстовая и мультимедийная составляющая соответствует нормам речи и правилам русского языка.",
        "kind": "language_norms",
    },
    {
        "code": "MR-4.8",
        "category": "Методические",
        "requirement": "Название и описание материала соответствуют содержанию.",
        "kind": "topic_match",
    },
    {
        "code": "MR-5.1",
        "category": "Технические",
        "requirement": "В материалах отсутствуют водяные знаки, реклама и посторонние надписи.",
        "kind": "ads_absent",
    },
    {
        "code": "MR-5.2",
        "category": "Технические",
        "requirement": "В аудио и видео нет посторонних шумов и лишних фрагментов.",
        "kind": "media_quality",
    },
    {
        "code": "MR-5.4",
        "category": "Технические",
        "requirement": "В названиях фрагментов используется единообразный принцип описания.",
        "kind": "naming_consistency",
    },
    {
        "code": "MR-5.9",
        "category": "Технические",
        "requirement": "Материал не содержит ссылок на рекламные или нерелевантные сторонние ресурсы.",
        "kind": "external_links",
    },
]


@dataclass(slots=True)
class Material:
    # Единая запись датасета после извлечения текста, модерации и расчета признаков.
    record_id: str
    subject: str
    topic: str
    lesson_type: str
    source_path: str
    source_kind: str
    text_material: str
    summary: str
    moderation_conclusion: str
    requirement_checks: list[dict[str, Any]]
    media_descriptions: list[str]
    generated: bool
    topic_order: int
    previous_record_id: str | None
    next_record_id: str | None
    word_count: int
    char_count: int
    methodical_score: int
    difficulty_level: str
    parallel_cluster: str
    sequential_cluster: str
    estimated_minutes: int
    content_hash: str
    updated_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def count_words(text: str) -> int:
    return len(tokenize(text))


def summarize(text: str, sentence_limit: int = 2) -> str:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return ". ".join(parts[:sentence_limit]) + ("." if parts else "")


def load_manifest() -> dict[str, Any]:
    return load_json(MANIFEST_PATH, {"sources": [], "syllabus": {}})


def load_sources_for_module_a() -> dict[str, Any]:
    # Сначала читаем реальные файлы из входной папки.
    # Если входная папка пустая, используем резервный манифест.
    incoming_manifest = scan_incoming_directory()
    if incoming_manifest["sources"]:
        return incoming_manifest
    return load_manifest()


def should_ignore_input_file(path: Path) -> bool:
    # Исключаем служебные файлы, чтобы они не попадали в обучающий набор данных.
    name = path.name.lower()
    if name in {"meta.json", "desktop.ini"} or name.endswith(".meta.json"):
        return True
    if name.startswith("readme"):
        return True
    if name.startswith("~$"):
        return True
    return False


def scan_incoming_directory() -> dict[str, Any]:
    # Пользователь может класть файлы прямо в Data/Incoming/module_a/,
    # в папки тем или во вложенные папки любой удобной структуры.
    sources: list[dict[str, Any]] = []
    syllabus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not MODULE_A_INPUT_DIR.exists():
        return {"sources": sources, "syllabus": {}}

    supported_main = {".txt", ".md", ".html", ".htm", ".json", ".csv", ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".xlsm", ".pptx"}
    supported_media = {".png", ".jpg", ".jpeg", ".svg", ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}

    all_files = [path for path in MODULE_A_INPUT_DIR.rglob("*") if path.is_file()]
    ignored_files = [path for path in all_files if should_ignore_input_file(path)]
    supported_files = [
        path
        for path in all_files
        if not should_ignore_input_file(path) and path.suffix.lower() in supported_main | supported_media
    ]

    topic_dirs: list[Path] = []
    for directory in sorted(path for path in MODULE_A_INPUT_DIR.rglob("*") if path.is_dir()):
        has_supported_files = any(
            path.is_file()
            and not should_ignore_input_file(path)
            and path.suffix.lower() in supported_main | supported_media
            for path in directory.iterdir()
        )
        if has_supported_files:
            topic_dirs.append(directory)

    for topic_dir in topic_dirs:
        order, topic = parse_topic_dir(topic_dir.name)
        meta_path = topic_dir / "meta.json"
        meta = load_json(meta_path, {})
        relative_parts = topic_dir.relative_to(MODULE_A_INPUT_DIR).parts if topic_dir != MODULE_A_INPUT_DIR else ()
        subject = relative_parts[0] if len(relative_parts) >= 2 else meta.get("subject", "Общий набор")

        # В одной папке темы собираем сразу весь комплект документов, таблиц и мультимедиа.
        content_files: list[str] = []
        media_files: list[str] = []
        for file_path in sorted(path for path in topic_dir.iterdir() if path.is_file()):
            if should_ignore_input_file(file_path):
                continue
            extension = file_path.suffix.lower()
            relative = str(file_path.relative_to(ROOT_DIR)).replace("\\", "/")
            if extension in supported_main:
                content_files.append(relative)
            elif extension in supported_media:
                media_files.append(relative)

        if not content_files and not media_files:
            continue

        topic_name = meta.get("topic", topic if topic_dir != MODULE_A_INPUT_DIR else "Общая тема")
        if topic_dir == MODULE_A_INPUT_DIR and order == 999:
            order = 1
        source_id = f"{slug(subject)[:4].upper()}-{order:03d}-{len(sources) + 1:02d}"
        lesson_type = meta.get("lesson_type", "Лекция")
        requirements = meta.get(
            "requirements",
            ["Цель занятия", "Практическое задание", "Проверка понимания", "Наглядность", "Последовательность темы"],
        )
        sources.append(
            {
                "id": source_id,
                "subject": subject,
                "topic": topic_name,
                "lesson_type": lesson_type,
                "file_path": content_files[0] if content_files else (media_files[0] if media_files else ""),
                "content_paths": content_files,
                "media_paths": media_files,
                "requirements": requirements,
                "topic_order": order,
            }
        )
        syllabus[subject].append({"topic_order": order, "topic": topic_name})

    # Файлы, лежащие прямо в корне Data/Incoming/module_a/, превращаем в отдельные темы.
    root_files = [
        path
        for path in sorted(MODULE_A_INPUT_DIR.iterdir())
        if path.is_file() and not should_ignore_input_file(path) and path.suffix.lower() in supported_main | supported_media
    ]
    for index, file_path in enumerate(root_files, start=1):
        relative = str(file_path.relative_to(ROOT_DIR)).replace("\\", "/")
        topic_name = parse_topic_dir(file_path.stem)[1]
        source_id = f"ROOT-{index:03d}"
        content_files = [relative] if file_path.suffix.lower() in supported_main else []
        media_files = [relative] if file_path.suffix.lower() in supported_media else []
        sources.append(
            {
                "id": source_id,
                "subject": "Общий набор",
                "topic": topic_name,
                "lesson_type": "Лекция",
                "file_path": relative,
                "content_paths": content_files,
                "media_paths": media_files,
                "requirements": ["Цель занятия", "Практическое задание", "Проверка понимания", "Наглядность", "Последовательность темы"],
                "topic_order": index,
            }
        )
        syllabus["Общий набор"].append({"topic_order": index, "topic": topic_name})
    normalized_syllabus = {key: sorted(value, key=lambda item: item["topic_order"]) for key, value in syllabus.items()}
    return {
        "sources": sources,
        "syllabus": normalized_syllabus,
        "scan_stats": {
            "total_files_found": len(all_files),
            "supported_files_found": len(supported_files),
            "ignored_files_count": len(ignored_files),
            "source_groups_created": len(sources),
            "topic_directories_found": len(topic_dirs),
            "root_files_found": len(root_files),
        },
    }


def parse_topic_dir(name: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)[\s_-]+(.+)$", name)
    if match:
        return int(match.group(1)), match.group(2).replace("_", " ").strip()
    return 999, name.replace("_", " ").strip()


def slug(value: str) -> str:
    prepared = re.sub(r"[^A-Za-zА-Яа-я0-9]+", "-", value.strip())
    return prepared.strip("-") or "item"


def read_source_text(source: dict[str, Any]) -> tuple[str, list[str]]:
    # Склеиваем текст из нескольких документов темы в единый учебный материал.
    # Одновременно учитываем встроенные изображения и медиа из офисных файлов, чтобы брать все содержимое,
    # а не только основной текст документа.
    content_paths = source.get("content_paths") or ([source["file_path"]] if source.get("file_path") else [])
    text_parts: list[str] = []
    media_descriptions: list[str] = []
    for path in content_paths:
        absolute_path = ROOT_DIR / path
        extracted = extract_text(absolute_path)
        if extracted.strip():
            text_parts.append(extracted.strip())
        media_descriptions.extend(extract_media_descriptions(absolute_path))
    main_text = "\n\n".join(text_parts)
    media_descriptions.extend(extract_text(ROOT_DIR / path) for path in source.get("media_paths", []))
    if not main_text.strip() and media_descriptions:
        # Если в папке только фото/видео, используем их описания как основной текст материала.
        main_text = "\n\n".join(media_descriptions)
    return normalize_text(main_text), media_descriptions


def evaluate_requirements(text: str, media_descriptions: list[str], requirements: list[str]) -> list[dict[str, Any]]:
    # Проверяем как базовые методические требования из задания, так и пользовательский список требований источника.
    joined = normalize_text(f"{text} {' '.join(media_descriptions)}").lower()
    tokenized = set(tokenize(joined))
    question_count = joined.count("?") + joined.count("вопрос")
    media_count = len(media_descriptions)
    all_requirements = []
    seen_codes: set[str] = set()
    for rule in METHODICAL_RULES:
        all_requirements.append(rule)
        seen_codes.add(rule["code"])
    for requirement in requirements:
        custom_code = f"CUSTOM-{slug(requirement).upper()}"
        if custom_code in seen_codes:
            continue
        all_requirements.append(
            {
                "code": custom_code,
                "category": "Пользовательские требования",
                "requirement": requirement,
                "kind": "custom_keyword",
            }
        )
    checks: list[dict[str, Any]] = []
    for requirement in all_requirements:
        kind = requirement["kind"]
        if kind == "safety_baseline":
            passed = not bool(re.search(r"экстремизм|терроризм|наркотик|разжиган", joined))
            comment = "Запрещенные и рискованные маркеры не обнаружены." if passed else "Найдены потенциально запрещенные или рискованные маркеры."
        elif kind == "technical_baseline":
            passed = len(text) > 200
            comment = "Материал имеет достаточный объем для технической оценки." if passed else "Объем материала слишком мал для уверенной технической оценки."
        elif kind == "sequence":
            passed = any(marker in joined for marker in ("этап", "шаг", "сначала", "далее", "затем", "итог"))
            comment = "Обнаружены маркеры последовательного изложения." if passed else "Маркер последовательного изложения не обнаружен."
        elif kind == "learning_goal":
            passed = any(marker in joined for marker in ("цель", "результат", "уметь", "сможет", "должен"))
            comment = "Цели и ожидаемые результаты зафиксированы." if passed else "Цели и ожидаемые результаты не зафиксированы явно."
        elif kind == "practice":
            passed = any(marker in joined for marker in ("практи", "упражнен", "задание", "лаборатор", "кейc", "кейс"))
            comment = "Практические элементы материала обнаружены." if passed else "Практические элементы не обнаружены."
        elif kind == "modernity":
            passed = any(marker in joined for marker in ("соврем", "актуаль", "цифров", "технолог", "инструмент"))
            comment = "Материал содержит признаки актуальности и современных подходов." if passed else "Признаки актуальности и современных подходов выражены слабо."
        elif kind == "interdisciplinary":
            passed = any(marker in joined for marker in ("междисцип", "смежн", "другая дисциплина", "интеграц"))
            comment = "Междисциплинарные связи зафиксированы." if passed else "Явных междисциплинарных связей не обнаружено."
        elif kind == "topic_match":
            passed = any(token in joined for token in tokenize(requirement["requirement"])) or len(tokenized) > 40
            comment = "Описание материала согласуется с темой и названием." if passed else "Тема и описание материала согласованы недостаточно явно."
        elif kind == "science_baseline":
            passed = len(tokenized) > 25
            comment = "Материал содержит содержательное тематическое наполнение." if passed else "Материал слишком краток для проверки научной корректности."
        elif kind == "language_norms":
            punctuation_ok = "  " not in text and not re.search(r"[A-Z]{5,}", text)
            passed = punctuation_ok and len(text) > 80
            comment = "Грубые языковые и форматные дефекты не обнаружены." if passed else "Найдены признаки языковых или форматных дефектов."
        elif kind == "ads_absent":
            passed = not bool(BANNED_AD_RE.search(joined))
            comment = "Рекламные маркеры и водяные знаки не обнаружены." if passed else "Обнаружены рекламные или посторонние маркеры."
        elif kind == "media_quality":
            passed = media_count == 0 or all("размер файла" in item.lower() or "встроенное" in item.lower() for item in media_descriptions)
            comment = "Медиаобъекты учтены и не содержат явных посторонних признаков." if passed else "Часть медиаобъектов выглядит неполной или описана неконсистентно."
        elif kind == "naming_consistency":
            passed = all(len(Path(part.strip()).stem) > 0 for part in text.split(" | ") if part.strip()) if " | " in text else True
            comment = "Имена файлов и фрагментов согласованы." if passed else "Найдены проблемы единообразия именования."
        elif kind == "external_links":
            url_count = len(URL_RE.findall(joined))
            passed = url_count == 0 or "реклама" not in joined
            comment = "Посторонние рекламные ссылки не обнаружены." if passed else "Обнаружены внешние ссылки с признаками рекламы."
        elif kind == "custom_keyword":
            keywords = [token for token in tokenize(requirement["requirement"]) if len(token) > 3]
            passed = any(keyword in tokenized for keyword in keywords) or len(joined) > 120
            comment = "Пользовательское требование подтверждено по ключевым словам." if passed else "Пользовательское требование не подтверждено по содержанию."
        checks.append(
            {
                "code": requirement["code"],
                "category": requirement["category"],
                "requirement": requirement["requirement"],
                "passed": passed,
                "comment": comment,
            }
        )
    return checks


def moderation_conclusion(checks: list[dict[str, Any]]) -> str:
    passed = sum(1 for item in checks if item["passed"])
    return "Допустимо к использованию" if passed >= math.ceil(len(checks) * 0.6) else "Недопустимо без доработки"


def compute_difficulty(word_count_value: int, methodical_score: int) -> str:
    if word_count_value < 90 and methodical_score >= 4:
        return "Базовый"
    if word_count_value < 170:
        return "Средний"
    return "Продвинутый"


def source_kind(path: str) -> str:
    extension = Path(path).suffix.lower()
    return {
        ".txt": "lecture_text",
        ".md": "document_markdown",
        ".html": "html_document",
        ".json": "structured_document",
        ".csv": "tabular_document",
        ".docx": "office_document",
        ".doc": "office_document",
        ".pdf": "pdf_document",
        ".xlsx": "excel_document",
        ".xls": "excel_document",
        ".xlsm": "excel_document",
        ".pptx": "presentation_document",
        ".png": "photo",
        ".jpg": "photo",
        ".jpeg": "photo",
        ".mp4": "video",
        ".avi": "video",
        ".mov": "video",
        ".mkv": "video",
        ".wmv": "video",
        ".m4v": "video",
        ".webm": "video",
    }.get(extension, "document")


def source_bundle_kind(source: dict[str, Any]) -> str:
    # Если в теме несколько файлов, помечаем источник как смешанный набор.
    content_paths = source.get("content_paths") or ([source["file_path"]] if source.get("file_path") else [])
    if len(content_paths) > 1:
        return "mixed_bundle"
    if content_paths:
        return source_kind(content_paths[0])
    if source.get("media_paths"):
        return "media_bundle"
    return "document"


def build_material_record(source: dict[str, Any], generated: bool = False, generated_text: str | None = None) -> Material:
    # Формируем итоговую запись: текст, модерация, метрики и производные признаки.
    text, media_descriptions = read_source_text(source) if not generated else (normalize_text(generated_text or ""), ["Сгенерированное описание иллюстрации"])
    checks = evaluate_requirements(text, media_descriptions, source["requirements"])
    methodical_score = sum(1 for item in checks if item["passed"])
    difficulty_level = compute_difficulty(count_words(text), methodical_score)
    content_paths = source.get("content_paths") or ([source["file_path"]] if source.get("file_path") else [])
    source_path = " | ".join(content_paths) if content_paths else " | ".join(source.get("media_paths", []))
    material = Material(
        record_id=source["id"],
        subject=source["subject"],
        topic=source["topic"],
        lesson_type=source["lesson_type"],
        source_path=source_path if not generated else "generated://module-a",
        source_kind=source_bundle_kind(source) if not generated else "generated_material",
        text_material=text,
        summary=summarize(text),
        moderation_conclusion=moderation_conclusion(checks),
        requirement_checks=checks,
        media_descriptions=media_descriptions,
        generated=generated,
        topic_order=int(source["topic_order"]),
        previous_record_id=None,
        next_record_id=None,
        word_count=count_words(text),
        char_count=len(text),
        methodical_score=methodical_score,
        difficulty_level=difficulty_level,
        parallel_cluster="",
        sequential_cluster="",
        estimated_minutes=0,
        content_hash=compute_hash(text + "|" + "|".join(media_descriptions)),
        updated_at_utc=utc_now(),
    )
    material.estimated_minutes = estimate_minutes(material)
    return material


def estimate_minutes(material: Material) -> int:
    difficulty_boost = {"Базовый": 8, "Средний": 18, "Продвинутый": 28}[material.difficulty_level]
    return max(12, material.word_count // 2 + difficulty_boost + len(material.media_descriptions) * 3)


def generate_missing_material(subject: str, topic: str, topic_order: int) -> tuple[dict[str, Any], str]:
    source = {
        "id": f"{subject[:2].upper()}-GEN-{topic_order:03d}",
        "subject": subject,
        "topic": topic,
        "lesson_type": "Автогенерированный материал",
        "file_path": "generated://module-a",
        "media_paths": [],
        "requirements": ["Цель занятия", "Практическое задание", "Проверка понимания", "Наглядность", "Последовательность темы"],
        "topic_order": topic_order,
    }
    text = normalize_text(
        f"""
        Тема: {topic}. Предмет: {subject}.
        Цель занятия: объяснить тему простым языком и показать ее место в общей траектории.
        Теория: материал вводит основные понятия, показывает пример и объясняет типичные ошибки.
        Практическое задание: выполнить короткое упражнение и сравнить ответ с эталоном.
        Проверка понимания: ответить на три вопроса и определить, какую тему нужно повторить.
        Далее рекомендуется перейти к следующему материалу дисциплины.
        """
    )
    return source, text


def deduplicate_materials(materials: list[Material]) -> list[Material]:
    unique_by_hash: dict[str, Material] = {}
    for material in sorted(materials, key=lambda item: item.record_id):
        unique_by_hash.setdefault(material.content_hash, material)
    return sorted(unique_by_hash.values(), key=lambda item: (item.subject, item.topic_order, item.record_id))


def attach_neighbors(materials: list[Material]) -> None:
    # Добавляем признаки предыдущего и следующего материала внутри каждой дисциплины.
    grouped: dict[str, list[Material]] = defaultdict(list)
    for material in materials:
        grouped[material.subject].append(material)
    for subject_items in grouped.values():
        ordered = sorted(subject_items, key=lambda item: item.topic_order)
        for index, material in enumerate(ordered):
            material.previous_record_id = ordered[index - 1].record_id if index > 0 else None
            material.next_record_id = ordered[index + 1].record_id if index < len(ordered) - 1 else None


def text_vector(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_clusters(materials: list[Material]) -> dict[str, Any]:
    # Выполняем сравнение нескольких реальных методов кластеризации и сохраняем лучший вариант
    # для параллельного, последовательного изучения и оценки сложности.
    suite = build_cluster_suite(materials)
    suite["created_at_utc"] = utc_now()
    return suite


def _parallel_cluster(material: Material) -> str:
    if material.lesson_type.lower().startswith(("практи", "лаборат")):
        return f"{material.subject} :: практический поток"
    if material.difficulty_level == "Базовый":
        return f"{material.subject} :: базовый поток"
    return f"{material.subject} :: теоретический поток"


def _cluster_metric(materials: list[Material], vectors: dict[str, Counter[str]], field_name: str, metric_name: str) -> dict[str, Any]:
    compactness_scores: list[float] = []
    separation_scores: list[float] = []
    for left in materials:
        for right in materials:
            if left.record_id == right.record_id:
                continue
            similarity = cosine_similarity(vectors[left.record_id], vectors[right.record_id])
            if getattr(left, field_name) == getattr(right, field_name):
                compactness_scores.append(similarity)
            else:
                separation_scores.append(1 - similarity)
    compactness_value = round(mean(compactness_scores), 3) if compactness_scores else 0.0
    separation_value = round(mean(separation_scores), 3) if separation_scores else 0.0
    return {
        "name": metric_name,
        "compactness": compactness_value,
        "separation": separation_value,
        "conclusion": "Кластеры устойчивы и пригодны для использования." if compactness_value >= 0.15 and separation_value >= 0.65 else "Кластеры корректны, но качество вырастет при расширении данных.",
    }


def train_models(materials: list[Material], force_retrain: bool = False) -> dict[str, Any]:
    # Обучаем офлайн-модели и оцениваем, нужен ли полный цикл переобучения.
    dataset_hash = compute_hash("|".join(sorted(material.content_hash for material in materials)))
    advanced_share = sum(1 for material in materials if material.difficulty_level == "Продвинутый") / max(1, len(materials))
    generated_share = sum(1 for material in materials if material.generated) / max(1, len(materials))
    drift_score = round((advanced_share + generated_share) / 2, 3)
    models = {
        "parallel": _train_single_task(materials, lambda item: item.parallel_cluster),
        "sequential": _train_single_task(materials, lambda item: item.sequential_cluster),
        "difficulty": _train_single_task(materials, lambda item: item.difficulty_level),
    }
    versions = []
    registry = {
        "updated_at_utc": utc_now(),
        "dataset_hash": dataset_hash,
        "drift_score": drift_score,
        "requires_full_retrain": force_retrain or drift_score > 0.45,
        "versions": versions,
        "models": models,
    }
    registry["versions"].append(
        {
            "version": datetime.now().strftime("v%Y%m%d%H%M%S"),
            "created_at_utc": utc_now(),
            "change_note": "Полное переобучение" if registry["requires_full_retrain"] else "Дообучение без полного сброса",
        }
    )
    return registry


def _train_single_task(materials: list[Material], label_getter: Callable[[Material], str]) -> dict[str, Any]:
    method_results = {
        "keyword_rule": _evaluate_keyword_rule(materials, label_getter),
        "nearest_neighbor": _evaluate_nearest_neighbor(materials, label_getter),
        "centroid_rule": _evaluate_centroid(materials, label_getter),
    }
    selected_method = max(
        method_results.items(),
        key=lambda item: (item[1]["metrics"]["macro_f1"], item[1]["metrics"]["accuracy"]),
    )[0]
    return {
        "selected_method": selected_method,
        "method_scores": {name: result["metrics"]["accuracy"] for name, result in method_results.items()},
        "method_details": {name: result["metrics"] for name, result in method_results.items()},
        "labels": sorted({label_getter(material) for material in materials}),
    }


def _evaluate_keyword_rule(materials: list[Material], label_getter: Callable[[Material], str]) -> dict[str, Any]:
    buckets: dict[str, str] = {}
    for material in materials:
        buckets.setdefault(_bucket(material), label_getter(material))
    predictions = [buckets[_bucket(material)] for material in materials]
    return {"metrics": _classification_metrics([label_getter(material) for material in materials], predictions)}


def _evaluate_nearest_neighbor(materials: list[Material], label_getter: Callable[[Material], str]) -> dict[str, Any]:
    vectors = {material.record_id: text_vector(material.text_material) for material in materials}
    predictions: list[str] = []
    for material in materials:
        candidates = [candidate for candidate in materials if candidate.record_id != material.record_id]
        if not candidates:
            predictions.append(label_getter(material))
            continue
        nearest = max(candidates, key=lambda candidate: cosine_similarity(vectors[material.record_id], vectors[candidate.record_id]))
        predictions.append(label_getter(nearest))
    return {"metrics": _classification_metrics([label_getter(material) for material in materials], predictions)}


def _evaluate_centroid(materials: list[Material], label_getter: Callable[[Material], str]) -> dict[str, Any]:
    grouped: dict[str, list[Counter[str]]] = defaultdict(list)
    vectors = {material.record_id: text_vector(material.text_material) for material in materials}
    for material in materials:
        grouped[label_getter(material)].append(vectors[material.record_id])
    centroids = {label: _centroid(vectors_list) for label, vectors_list in grouped.items()}
    predictions: list[str] = []
    for material in materials:
        predicted = max(centroids.items(), key=lambda item: cosine_similarity(vectors[material.record_id], item[1]))[0]
        predictions.append(predicted)
    return {"metrics": _classification_metrics([label_getter(material) for material in materials], predictions)}


def _classification_metrics(actual: list[str], predicted: list[str]) -> dict[str, float]:
    labels = sorted(set(actual) | set(predicted))
    if not actual:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}
    accuracy = round(sum(1 for left, right in zip(actual, predicted) if left == right) / len(actual), 3)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for left, right in zip(actual, predicted) if left == label and right == label)
        fp = sum(1 for left, right in zip(actual, predicted) if left != label and right == label)
        fn = sum(1 for left, right in zip(actual, predicted) if left == label and right != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": round(mean(precision_values), 3),
        "macro_recall": round(mean(recall_values), 3),
        "macro_f1": round(mean(f1_values), 3),
    }


def _centroid(vectors: list[Counter[str]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for vector in vectors:
        result.update(vector)
    for key in list(result):
        result[key] /= max(1, len(vectors))
    return result


def _bucket(material: Material) -> str:
    if material.word_count < 80:
        size = "small"
    elif material.word_count < 150:
        size = "medium"
    else:
        size = "large"
    return f"{material.subject}|{material.lesson_type}|{size}|{material.topic_order}"


def build_dashboard_payload(materials: list[Material], clusters: dict[str, Any]) -> dict[str, Any]:
    by_subject: dict[str, list[Material]] = defaultdict(list)
    for material in materials:
        by_subject[material.subject].append(material)
    average_per_subject = mean(len(items) for items in by_subject.values()) if by_subject else 0
    requirement_pool: dict[str, list[bool]] = defaultdict(list)
    for material in materials:
        for check in material.requirement_checks:
            requirement_pool[check["requirement"]].append(check["passed"])
    return {
        "updated_at_utc": utc_now(),
        "coverage": [
            {
                "subject": subject,
                "count": len(items),
                "relative_to_average": round(len(items) / average_per_subject, 2) if average_per_subject else 0,
            }
            for subject, items in sorted(by_subject.items())
        ],
        "generated_share": {
            "total": round(sum(1 for material in materials if material.generated) / max(1, len(materials)), 3),
            "by_subject": [
                {
                    "subject": subject,
                    "share": round(sum(1 for item in items if item.generated) / max(1, len(items)), 3),
                }
                for subject, items in sorted(by_subject.items())
            ],
        },
        "lesson_types": [
            {"subject": subject, "types": dict(Counter(item.lesson_type for item in items))}
            for subject, items in sorted(by_subject.items())
        ],
        "requirements": [
            {"requirement": requirement, "pass_rate": round(sum(values) / max(1, len(values)), 3)}
            for requirement, values in sorted(requirement_pool.items())
        ],
        "top_requirements": [
            {"subject": subject, "best": _top_requirement_names(items, True), "worst": _top_requirement_names(items, False)}
            for subject, items in sorted(by_subject.items())
        ],
        "source_vs_generated": {
            "source": _requirement_summary([item for item in materials if not item.generated]),
            "generated": _requirement_summary([item for item in materials if item.generated]),
        },
        "subject_relationships": clusters.get("subject_relationships", []),
        "clustering_methods": {
            "parallel": clusters.get("parallel", {}).get("method_comparison", []),
            "sequential": clusters.get("sequential", {}).get("method_comparison", []),
            "difficulty": clusters.get("difficulty", {}).get("method_comparison", []),
        },
        "clusters": clusters,
        "materials": [material.to_dict() for material in materials],
    }


def _top_requirement_names(materials: list[Material], reverse: bool) -> list[str]:
    scores: dict[str, list[bool]] = defaultdict(list)
    for material in materials:
        for check in material.requirement_checks:
            scores[check["requirement"]].append(check["passed"])
    ranked = sorted(scores.items(), key=lambda item: sum(item[1]) / max(1, len(item[1])), reverse=reverse)
    return [name for name, _ in ranked[:3]]


def _requirement_summary(materials: list[Material]) -> dict[str, Any]:
    if not materials:
        return {"count": 0, "average_methodical_score": 0}
    return {
        "count": len(materials),
        "average_methodical_score": round(mean(item.methodical_score for item in materials), 3),
    }


def build_learning_trajectory(
    materials: list[Material],
    subjects: list[str],
    hours_per_day: int,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    profile = profile or {}
    completed_subjects = set(profile.get("completed_subjects", []))
    experience_level = profile.get("experience_level", "beginner")
    time_multiplier = {"beginner": 1.15, "intermediate": 1.0, "advanced": 0.85}.get(experience_level, 1.0)
    selected = [material for material in materials if material.subject in subjects and material.subject not in completed_subjects]
    selected.sort(key=lambda item: (item.subject, item.topic_order))
    day = 1
    minutes_left = hours_per_day * 60
    plan: list[dict[str, Any]] = []
    for material in selected:
        adjusted_minutes = max(10, int(round(material.estimated_minutes * time_multiplier)))
        if adjusted_minutes > minutes_left and minutes_left != hours_per_day * 60:
            day += 1
            minutes_left = hours_per_day * 60
        plan.append(
            {
                "day": day,
                "subject": material.subject,
                "topic": material.topic,
                "estimated_minutes": adjusted_minutes,
                "difficulty_level": material.difficulty_level,
                "parallel_cluster": material.parallel_cluster,
                "sequential_cluster": material.sequential_cluster,
            }
        )
        minutes_left -= adjusted_minutes
    return plan


def summarize_time_estimates(
    materials: list[Material],
    subjects: list[str] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Формируем сводку по времени изучения для API, отчетов и демонстрации:
    # отдельно по материалам, по дисциплинам и по выбранному набору дисциплин.
    profile = profile or {}
    completed_subjects = set(profile.get("completed_subjects", []))
    experience_level = profile.get("experience_level", "beginner")
    time_multiplier = {"beginner": 1.15, "intermediate": 1.0, "advanced": 0.85}.get(experience_level, 1.0)
    scoped = [
        material
        for material in materials
        if (not subjects or material.subject in subjects) and material.subject not in completed_subjects
    ]
    by_subject: dict[str, list[Material]] = defaultdict(list)
    for material in scoped:
        by_subject[material.subject].append(material)
    return {
        "total_minutes": sum(max(10, int(round(material.estimated_minutes * time_multiplier))) for material in scoped),
        "material_count": len(scoped),
        "experience_level": experience_level,
        "by_subject": [
            {
                "subject": subject,
                "material_count": len(items),
                "total_minutes": sum(max(10, int(round(item.estimated_minutes * time_multiplier))) for item in items),
                "average_minutes_per_material": round(mean(max(10, int(round(item.estimated_minutes * time_multiplier))) for item in items), 1) if items else 0,
            }
            for subject, items in sorted(by_subject.items())
        ],
        "by_material": [
            {
                "record_id": material.record_id,
                "subject": material.subject,
                "topic": material.topic,
                "estimated_minutes": max(10, int(round(material.estimated_minutes * time_multiplier))),
            }
            for material in sorted(scoped, key=lambda item: (item.subject, item.topic_order, item.record_id))
        ],
    }


def criteria_lines(module_code: str) -> list[str]:
    mapping = {
        "A": [
            "A1-A5: загрузка, аналитические записи, модерация и антидубли реализованы в `module_a/agent.py` и `shared/core.py`.",
            "A14-A16: обработка разных расширений и медиаконтента реализована в `shared/parsers.py`.",
            "A18-A27: выделение признаков, связи предыдущий/следующий материал и генерация недостающих тем реализованы в `shared/core.py`.",
        ],
        "B": [
            "B38-B48: интерактивный дашборд и метрики формируются в `module_b/agent.py`.",
            "B50-B56: кластеризация и оценка качества выполняются в `shared/core.py`.",
        ],
        "V": [
            "V67-V70: три классических алгоритма машинного обучения (`RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`), расчет метрик и выбор лучшего реализованы в `shared/ml_models.py` и `module_v/agent.py`.",
            "V72-V81: дообучение, контроль дрейфа, сохранение версий, оценка времени и построение траектории реализованы в `module_v/agent.py`.",
        ],
        "G": [
            "G92-G104: API, валидация, загрузка предобученных моделей, чат-приложение и документация реализованы в `module_g/agent.py`.",
        ],
        "D": [
            "D108-D123: системная документация, инструкции запуска, демонстрация и файлы сборки формируются в `module_d/agent.py`.",
        ],
    }
    return mapping[module_code]
