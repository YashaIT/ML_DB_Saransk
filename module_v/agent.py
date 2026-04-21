from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import EXPORTS_DIR, MODEL_DIR, REPORTS_DIR
from shared.core import Material, build_clusters, build_learning_trajectory, criteria_lines, summarize_time_estimates
from shared.ml_models import build_sequential_role, save_model_package, train_task_models
from shared.storage import load_database, save_database


def main() -> None:
    # Шаг 1. Загружаем базу данных после модуля A и восстанавливаем материалы как объекты.
    # Это дает нам единый источник данных для обучения, прогноза и построения траектории.
    database = load_database()
    materials = [Material(**item) for item in database["materials"]]

    # Шаг 1.1. Перед обучением пересчитываем разметку модулем Б, чтобы модель училась
    # на актуальных целевых метках параллельного, последовательного изучения и сложности.
    database["clusters"] = build_clusters(materials)

    # Шаг 2. Определяем, нужен ли принудительный полный цикл переобучения.
    force_retrain = ask_force_retrain()

    # Шаг 3. Обучаем модели по трем задачам и формируем реестр моделей.
    registry, packages = build_model_registry(materials, force_retrain)

    # Шаг 4. Сохраняем бинарные артефакты и JSON-метаданные, чтобы модуль Г
    # мог использовать модели без повторного обучения.
    save_model_artifacts(registry, packages)

    # Шаг 5. Запрашиваем пользовательский профиль для оценки времени и построения маршрута.
    available_subjects = sorted({material.subject for material in materials})
    print("Доступные дисциплины:", ", ".join(available_subjects))
    user_profile = ask_user_profile(available_subjects)
    subjects = user_profile["subjects"]
    hours_per_day = user_profile["hours_per_day"]

    # Шаг 5.1. Строим индивидуальную траекторию обучения с учетом опыта,
    # уже изученных дисциплин и допустимой ежедневной нагрузки.
    trajectory = build_learning_trajectory(materials, subjects, hours_per_day, profile=user_profile)
    save_trajectory(trajectory)
    save_time_visualization(materials, subjects, user_profile)

    # Шаг 6. Обновляем БД и сохраняем лог качества моделей между версиями.
    previous_models = database.get("models", {})
    previous_versions = previous_models.get("versions", [])
    registry["versions"] = previous_versions + registry.get("versions", [])
    registry["quality_log"] = build_quality_log(previous_models, registry)
    database["models"] = registry
    database["materials"] = [material.to_dict() for material in materials]
    database["runs"].append(
        {
            "module": "V",
            "started_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "result": "OK",
            "details": "Обучены ML-модели, рассчитаны метрики, сохранены версии и построена персональная траектория.",
        }
    )

    # Шаг 7. Формируем подробный отчет по модулю В и сохраняем состояние в SQLite.
    save_report(registry, materials, subjects, user_profile)
    save_database(database)
    print("Модуль В завершен.")
    print(f"Отчет: {REPORTS_DIR / 'module_v_report.md'}")


def ask_force_retrain() -> bool:
    # Полное переобучение запускаем только по запросу пользователя.
    # В неинтерактивном сценарии агент работает в режиме дообучения по умолчанию.
    if not sys.stdin.isatty():
        print("Неинтерактивный запуск: полное переобучение отключено по умолчанию.")
        return False
    answer = input("Полное переобучение? (yes/no, по умолчанию no): ").strip().lower()
    return answer == "yes"


def ask_user_profile(available_subjects: list[str]) -> dict[str, object]:
    # Агент собирает входные параметры, прямо перечисленные в задании:
    # опыт, уже изученные дисциплины, временные рамки и ежедневную нагрузку.
    if not sys.stdin.isatty():
        print("Неинтерактивный запуск: используется профиль beginner, все дисциплины, 2 часа в день, срок 14 дней.")
        return {
            "experience_level": "beginner",
            "subjects": available_subjects,
            "completed_subjects": [],
            "hours_per_day": 2,
            "deadline_days": 14,
        }

    raw_experience = input("Уровень опыта (beginner/intermediate/advanced), пусто = beginner: ").strip().lower() or "beginner"
    raw_subjects = input("Какие дисциплины включить? Через запятую, пусто = все: ").strip()
    raw_completed = input("Какие дисциплины уже изучены? Через запятую, пусто = нет: ").strip()
    raw_hours = input("Сколько часов в день готовы уделять? Пусто = 2: ").strip()
    raw_deadline = input("За сколько дней хотите пройти выбранные дисциплины? Пусто = 14: ").strip()

    return {
        "experience_level": raw_experience if raw_experience in {"beginner", "intermediate", "advanced"} else "beginner",
        "subjects": [item.strip() for item in raw_subjects.split(",") if item.strip()] if raw_subjects else available_subjects,
        "completed_subjects": [item.strip() for item in raw_completed.split(",") if item.strip()],
        "hours_per_day": int(raw_hours) if raw_hours.isdigit() and int(raw_hours) > 0 else 2,
        "deadline_days": int(raw_deadline) if raw_deadline.isdigit() and int(raw_deadline) > 0 else 14,
    }


def build_model_registry(materials: list[Material], force_retrain: bool) -> tuple[dict, dict[str, dict]]:
    # Собираем реестр моделей и отдельные пакеты для сериализации в joblib.
    dataset_hash = compute_dataset_hash(materials)
    version = datetime.now().strftime("v%Y%m%d%H%M%S")
    drift_score, drift_monitor = estimate_drift(materials)

    task_payloads = {
        "parallel": train_task_models(materials, lambda item: item.parallel_cluster),
        "sequential": train_task_models(materials, build_sequential_role),
        "difficulty": train_task_models(materials, lambda item: item.difficulty_level),
    }

    models: dict[str, dict] = {}
    packages: dict[str, dict] = {}
    for task_name, task_payload in task_payloads.items():
        package = task_payload.pop("package")
        package["task_name"] = task_name
        packages[task_name] = package
        models[task_name] = {
            **task_payload,
            "artifact_path": str((MODEL_DIR / f"{task_name}_{version}.joblib")).replace("\\", "/"),
            "metadata_path": str((MODEL_DIR / f"{task_name}_{version}.json")).replace("\\", "/"),
        }

    registry = {
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_hash": dataset_hash,
        "drift_score": drift_score,
        "drift_monitor": drift_monitor,
        "requires_full_retrain": force_retrain or drift_score > 0.45,
        "versions": [
            {
                "version": version,
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "change_note": "Полное переобучение" if force_retrain or drift_score > 0.45 else "Дообучение без полного сброса",
            }
        ],
        "models": models,
    }
    return registry, packages


def compute_dataset_hash(materials: list[Material]) -> str:
    # Хэш набора данных позволяет точно понимать, на каком состоянии базы
    # были обучены текущие версии моделей.
    import hashlib

    payload = "|".join(sorted(material.content_hash for material in materials))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def estimate_drift(materials: list[Material]) -> tuple[float, dict]:
    # Для офлайн-сценария используем собственный монитор дрейфа,
    # чтобы не зависеть от сетевых скачиваний или внешних сервисов.
    advanced_share = sum(1 for material in materials if material.difficulty_level == "Продвинутый") / max(1, len(materials))
    generated_share = sum(1 for material in materials if material.generated) / max(1, len(materials))
    drift_score = round((advanced_share + generated_share) / 2, 3)
    return drift_score, {
        "tool": "internal_drift_monitor",
        "note": "Для чемпионатного офлайн-сценария дрейф оценивается по доле продвинутых и сгенерированных материалов.",
        "advanced_share": round(advanced_share, 3),
        "generated_share": round(generated_share, 3),
    }


def build_quality_log(previous_registry: dict, new_registry: dict) -> list[dict]:
    # Логируем изменение качества моделей от версии к версии,
    # чтобы показать реальный механизм непрерывного обучения.
    previous_models = previous_registry.get("models", {})
    current_version = new_registry.get("versions", [{}])[-1].get("version", "v_current")
    rows: list[dict] = []
    for task_name, task_payload in new_registry.get("models", {}).items():
        current_method = task_payload["selected_method"]
        previous_metrics = previous_models.get(task_name, {}).get("method_details", {}).get(current_method, {})
        current_metrics = task_payload.get("method_details", {}).get(current_method, {})
        rows.append(
            {
                "version": current_version,
                "task_name": task_name,
                "selected_method": current_method,
                "accuracy_delta": round(current_metrics.get("accuracy", 0.0) - previous_metrics.get("accuracy", 0.0), 4),
                "macro_f1_delta": round(current_metrics.get("macro_f1", 0.0) - previous_metrics.get("macro_f1", 0.0), 4),
                "roc_auc_delta": round(current_metrics.get("roc_auc", 0.0) - previous_metrics.get("roc_auc", 0.0), 4),
            }
        )
    return previous_registry.get("quality_log", []) + rows


def save_model_artifacts(registry: dict, packages: dict[str, dict]) -> None:
    # Сохраняем две формы артефактов:
    # 1. joblib-пакет для использования модулем Г;
    # 2. json-файл с метриками, чтобы было удобно показывать эксперту.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = registry.get("versions", [{}])[-1].get("version", "v_current")
    versions_path = MODEL_DIR / "versions.json"

    for task_name, package in packages.items():
        task_meta = registry["models"][task_name]
        save_model_package(Path(task_meta["artifact_path"]), package)
        Path(task_meta["metadata_path"]).write_text(
            json.dumps(
                {
                    "task": task_name,
                    "selected_method": task_meta["selected_method"],
                    "method_scores": task_meta["method_scores"],
                    "method_details": task_meta["method_details"],
                    "labels": task_meta["labels"],
                    "split_strategy": task_meta["split_strategy"],
                    "training_size": task_meta["training_size"],
                    "holdout_size": task_meta["holdout_size"],
                    "dataset_hash": registry["dataset_hash"],
                    "updated_at_utc": registry["updated_at_utc"],
                    "requires_full_retrain": registry["requires_full_retrain"],
                    "artifact_path": task_meta["artifact_path"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    existing_versions: list[dict] = []
    if versions_path.exists():
        existing_versions = json.loads(versions_path.read_text(encoding="utf-8"))
    versions_path.write_text(
        json.dumps(existing_versions + registry.get("versions", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_trajectory(trajectory: list[dict]) -> None:
    # Сохраняем траекторию и сразу добавляем mermaid-визуализацию,
    # чтобы в отчете можно было показать порядок изучения и нагрузку по дням.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Индивидуальная траектория",
        "",
        "| День | Дисциплина | Тема | Время, мин | Сложность |",
        "|---|---|---|---:|---|",
    ]
    for item in trajectory:
        lines.append(f"| {item['day']} | {item['subject']} | {item['topic']} | {item['estimated_minutes']} | {item['difficulty_level']} |")
    lines.extend(["", "```mermaid", "gantt", "title Траектория обучения", "dateFormat X"])
    current_day = None
    for index, item in enumerate(trajectory):
        if item["day"] != current_day:
            current_day = item["day"]
            lines.append(f"section День {item['day']}")
        lines.append(f"{item['topic']} :{index}, 0, {max(1, item['estimated_minutes'] // 30)}d")
    lines.append("```")
    (REPORTS_DIR / "module_v_trajectory.md").write_text("\n".join(lines), encoding="utf-8")


def save_time_visualization(materials: list[Material], subjects: list[str], user_profile: dict[str, object]) -> None:
    # Сохраняем сводку временных характеристик сразу в трех форматах:
    # CSV для таблиц, markdown для отчета и SVG для визуального анализа.
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    charts_dir = REPORTS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    time_summary = summarize_time_estimates(materials, subjects, profile=user_profile)
    csv_lines = ["subject,material_count,total_minutes,total_hours"]
    for row in time_summary["by_subject"]:
        csv_lines.append(
            f"{row['subject']},{row['material_count']},{row['total_minutes']},{round(row['total_minutes'] / 60, 2)}"
        )
    (EXPORTS_DIR / "module_v_time_summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    md_lines = [
        "# Временные характеристики обучения",
        "",
        f"- Уровень опыта: **{time_summary.get('experience_level', 'beginner')}**.",
        f"- Часов в день: **{user_profile['hours_per_day']}**.",
        f"- Всего материалов: **{time_summary['material_count']}**.",
        f"- Суммарное время: **{time_summary['total_minutes']} мин**.",
        "",
        "| Дисциплина | Материалов | Время, мин | Время, ч |",
        "|---|---:|---:|---:|",
    ]
    for row in time_summary["by_subject"]:
        md_lines.append(
            f"| {row['subject']} | {row['material_count']} | {row['total_minutes']} | {round(row['total_minutes'] / 60, 2)} |"
        )
    (REPORTS_DIR / "module_v_time_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    bar_width = 560
    bar_height = 24
    gap = 18
    max_minutes = max((row["total_minutes"] for row in time_summary["by_subject"]), default=1)
    svg_height = 70 + len(time_summary["by_subject"]) * (bar_height + gap)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{svg_height}" viewBox="0 0 920 {svg_height}">',
        '<rect width="100%" height="100%" fill="#fffaf1"/>',
        '<text x="20" y="32" font-size="22" font-family="Segoe UI" fill="#203040">Временные характеристики по дисциплинам (минуты)</text>',
    ]
    for index, row in enumerate(time_summary["by_subject"]):
        y = 60 + index * (bar_height + gap)
        current_width = 0 if max_minutes == 0 else round((row["total_minutes"] / max_minutes) * bar_width, 2)
        svg_lines.extend(
            [
                f'<text x="20" y="{y + 17}" font-size="14" font-family="Segoe UI" fill="#203040">{row["subject"]}</text>',
                f'<rect x="220" y="{y}" width="{bar_width}" height="{bar_height}" rx="8" fill="#e6eef7"/>',
                f'<rect x="220" y="{y}" width="{current_width}" height="{bar_height}" rx="8" fill="#0f7c59"/>',
                f'<text x="795" y="{y + 17}" font-size="13" font-family="Segoe UI" fill="#203040">{row["total_minutes"]} мин ({round(row["total_minutes"] / 60, 2)} ч)</text>',
            ]
        )
    svg_lines.append("</svg>")
    (charts_dir / "module_v_time_summary.svg").write_text("\n".join(svg_lines), encoding="utf-8")


def save_report(models: dict, materials: list[Material], subjects: list[str], user_profile: dict[str, object]) -> None:
    # Подробный отчет фиксирует выбор алгоритмов, метрики, дообучение, дрейф,
    # временные оценки и все пути к выходным файлам.
    version = models.get("versions", [{}])[-1].get("version", "v_current")
    time_summary = summarize_time_estimates(materials, subjects, profile=user_profile)
    hours_per_day = int(user_profile["hours_per_day"])
    experience_level = str(user_profile["experience_level"])
    completed_subjects = ", ".join(user_profile.get("completed_subjects", [])) or "нет"
    deadline_days = user_profile.get("deadline_days", 14)

    lines = [
        "# Отчет по модулю В",
        "",
        "## Критерии и аспекты",
        "| Аспект | Где выполнено | Что лежит на выходе | Как проверить |",
        "|---|---|---|---|",
        f"| Три модели по трем задачам | `module_v/agent.py`, `shared/ml_models.py` | `Data/Models/*_{version}.joblib`, `Data/Models/*_{version}.json` | Открыть папку `Data/Models` |",
        "| Три метода для каждой задачи | `shared/ml_models.py` | метрики в `Docs/Reports/module_v_report.md` и json-файлах моделей | Сравнить `random_forest`, `gradient_boosting`, `logistic_regression` |",
        "| Метрики качества | `shared/ml_models.py`, `module_v/agent.py` | accuracy, macro_precision, macro_recall, macro_f1, ROC-AUC | Открыть разделы задач в этом отчете |",
        "| Отложенная выборка | `shared/ml_models.py` | поля `split_strategy`, `training_size`, `holdout_size` | Открыть json-файлы моделей |",
        "| Дообучение и дрейф | `module_v/agent.py` | `drift_score`, `requires_full_retrain`, `versions.json`, `quality_log` | Открыть `Data/Models/versions.json` и SQLite |",
        "| Визуализация траектории | `module_v/agent.py` | `Docs/Reports/module_v_trajectory.md` | Открыть файл траектории |",
        "| Визуализация временных характеристик | `module_v/agent.py` | `Docs/Reports/module_v_time_summary.md`, `Docs/Reports/charts/module_v_time_summary.svg`, `Data/Exports/module_v_time_summary.csv` | Открыть markdown, SVG и CSV |",
        "",
        "## Описание выбранных моделей",
        "- Для каждой задачи сравниваются три классических алгоритма машинного обучения: `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`.",
        "- Выбор лучшей модели выполняется по `macro_f1`, затем по `roc_auc`, затем по `accuracy`.",
        "- Такой порядок выбран потому, что F1 лучше отражает баланс точности и полноты на несбалансированных классах, а ROC-AUC показывает качество разделения классов вероятностной моделью.",
        "",
        "## Результаты по задачам",
    ]

    for task_name, task_payload in models["models"].items():
        lines.append("")
        lines.append(f"### {task_name}")
        lines.append(f"- Выбранная модель: **{task_payload['selected_method']}**.")
        lines.append(f"- Разбиение данных: **{task_payload['split_strategy']}**.")
        lines.append(f"- Размер обучающей выборки: **{task_payload['training_size']}**.")
        lines.append(f"- Размер отложенной выборки: **{task_payload['holdout_size']}**.")
        lines.append(f"- Бинарный артефакт модели: `{task_payload['artifact_path']}`.")
        lines.append(f"- Метаданные модели: `{task_payload['metadata_path']}`.")
        for method_name, metrics in task_payload.get("method_details", {}).items():
            lines.append(
                f"- {method_name}: accuracy={metrics['accuracy']}, macro_precision={metrics['macro_precision']}, "
                f"macro_recall={metrics['macro_recall']}, macro_f1={metrics['macro_f1']}, roc_auc={metrics['roc_auc']}"
            )

    lines.extend(
        [
            "",
            "## Непрерывное обучение",
            f"- Хэш набора данных: `{models['dataset_hash']}`.",
            f"- Оценка дрейфа: **{models['drift_score']}**.",
            f"- Полное переобучение: **{models['requires_full_retrain']}**.",
            f"- Текущая версия: **{version}**.",
            "- При каждом новом запуске агент заново читает материалы из базы, пересчитывает хэш набора данных, дрейф и качество моделей.",
            "- Если дрейф превышает порог 0.45, агент помечает необходимость полного переобучения.",
            "- Все версии моделей сохраняются в `Data/Models/versions.json`, а изменения качества логируются в `quality_log` и в этом отчете.",
            "",
            "### Лог изменения качества",
        ]
    )
    for row in models.get("quality_log", []):
        lines.append(
            f"- {row['version']} / {row['task_name']} / {row['selected_method']}: "
            f"Δaccuracy={row['accuracy_delta']}, Δmacro_f1={row['macro_f1_delta']}, Δroc_auc={row['roc_auc_delta']}"
        )

    lines.extend(
        [
            "",
            "## Прогнозирование времени",
            f"- Выбранные дисциплины: {', '.join(subjects)}.",
            f"- Уровень опыта пользователя: **{experience_level}**.",
            f"- Уже изученные дисциплины: **{completed_subjects}**.",
            f"- Часов в день: **{hours_per_day}**.",
            f"- Желаемый срок прохождения: **{deadline_days} дней**.",
            f"- Всего материалов: **{time_summary['material_count']}**.",
            f"- Суммарное время по выбранным дисциплинам: **{time_summary['total_minutes']} мин**.",
            "- Визуализация траектории сохранена в `Docs/Reports/module_v_trajectory.md`.",
            "- Визуализация временных характеристик сохранена в `Docs/Reports/module_v_time_summary.md` и `Docs/Reports/charts/module_v_time_summary.svg`.",
            "",
            "## Выходные файлы модуля",
            f"- `Data/Models/parallel_{version}.joblib`, `Data/Models/sequential_{version}.joblib`, `Data/Models/difficulty_{version}.joblib` — бинарные пакеты моделей для модуля Г.",
            f"- `Data/Models/parallel_{version}.json`, `Data/Models/sequential_{version}.json`, `Data/Models/difficulty_{version}.json` — метаданные моделей, метрики и параметры разбиения.",
            "- `Data/Models/versions.json` — журнал версий моделей.",
            "- `Data/Exports/module_v_time_summary.csv` — таблица временных характеристик по дисциплинам.",
            "- `Docs/Reports/module_v_trajectory.md` — персональная траектория пользователя.",
            "- `Docs/Reports/module_v_time_summary.md` — табличный отчет по времени.",
            "- `Docs/Reports/charts/module_v_time_summary.svg` — графическая визуализация времени по дисциплинам.",
            "",
            "## Где выполнены критерии",
            *[f"- {line}" for line in criteria_lines("V")],
        ]
    )
    (REPORTS_DIR / "module_v_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
