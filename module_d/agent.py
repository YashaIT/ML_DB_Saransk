from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import REPORTS_DIR, ROOT_DIR, ensure_workspace
from shared.storage import load_database


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def main() -> None:
    ensure_workspace()
    build_readme()
    build_system_docs()
    build_detailed_module_reports()
    build_launch_files()
    build_build_scripts()
    print("Модуль Д завершен.")
    print(f"README: {ROOT_DIR / 'README.md'}")


def build_readme() -> None:
    write(
        ROOT_DIR / "README.md",
        """# ML_DB_Saransk

Оффлайн-система на Python для конкурсного задания по компетенции «Машинное обучение и большие данные».

## Структура
- `module_a` — сбор, предобработка, модерация и расширение набора данных
- `module_b` — аналитический дашборд, роли доступа и кластеризация
- `module_v` — обучение моделей, контроль дрейфа и построение траекторий
- `module_g` — локальный API и web-интерфейс
- `module_d` — документация, карта критериев, инструкции запуска и сборка
- `shared` — общее ядро проекта

## Что нужно установить
- Python 3.12+
- PyInstaller для сборки `exe`

## База данных
- `Data/ml_db.sqlite3`

## Куда класть исходные данные
- Все материалы клади в `Data/Incoming/module_a/`
- Допустимы структуры:
  - `Data/Incoming/module_a/<Дисциплина>/<NN_Тема>/`
  - `Data/Incoming/module_a/<NN_Тема>/`
  - файлы сразу в `Data/Incoming/module_a/`
- Можно класть папки внутри папок, агент сам соберет материалы

## Поддерживаемые форматы
- текст: `txt`, `md`, `html`, `json`, `csv`
- документы: `doc`, `docx`, `pdf`
- таблицы: `xlsx`, `xls`, `xlsm`
- изображения: `png`, `jpg`, `jpeg`, `svg`
- видео: `mp4`, `avi`, `mov`, `mkv`, `wmv`, `m4v`, `webm`

## Пароли для модуля Б
- `viewer / viewer123`
- `admin / admin123`

## Порядок запуска
1. Положить данные в `Data/Incoming/module_a/`
2. Запустить `run_module_a.bat`
3. Запустить `run_module_b.bat`
4. Запустить `run_module_v.bat`
5. Запустить `run_module_g.bat`
6. Запустить `run_module_d.bat`

## Сборка исполняемых файлов
- `build_all.bat`
- готовые файлы будут в `Distribution`
""",
    )


def build_system_docs() -> None:
    write(
        REPORTS_DIR / "system_documentation.md",
        """# Системная документация

## Архитектура
- `module_a` собирает, нормализует и модерирует материалы, затем сохраняет их в SQLite и экспортирует датасет
- `module_b` читает SQLite, строит аналитику и кластеры, отображает их в локальном web-dashboard
- `module_v` обучает модели по трем задачам, сохраняет версии и строит траектории
- `module_g` предоставляет API и web-интерфейс для использования готовых моделей и материалов
- `module_d` генерирует README, отчеты, карту критериев, bat-файлы и build-скрипты

## Роли
- `viewer` видит обзорный дашборд
- `admin` видит обзор, служебную информацию и таблицу материалов
- пароли лежат в `Data/security.json`

## Схема работы
1. `module_a` формирует `Data/ml_db.sqlite3`
2. `module_b` использует SQLite для аналитики и кластеров
3. `module_v` сохраняет модели в `Data/Models`
4. `module_g` загружает уже готовые модели и отвечает через API
5. `module_d` готовит документацию и материалы защиты
""",
    )
    write(
        REPORTS_DIR / "demo_script.md",
        """# Сценарий демонстрации до 10 минут

1. Показать папку `Data/Incoming/module_a/`
2. Запустить `run_module_a.bat`, показать `Data/ml_db.sqlite3` и `dataset_module_a.csv`
3. Показать `module_a_report.md`, `module_a_attribute_analysis.md`, `module_a_similarity_analysis.md`
4. Запустить `run_module_b.bat`, показать роли `viewer` и `admin`
5. Показать `module_b_report.md` и SVG-графики
6. Запустить `run_module_v.bat`, показать `Data/Models` и `module_v_trajectory.md`
7. Запустить `run_module_g.bat`, показать интерфейс и API
8. Запустить `run_module_d.bat`, открыть `criteria_map.md`
""",
    )
    write(
        REPORTS_DIR / "championship_steps.md",
        """# Работа по модулям на чемпионате

## Модуль A
- Папки: `module_a`, `shared`, `Data`
- Данные: `Data/Incoming/module_a/`
- Запуск: `run_module_a.bat`
- Результаты: `Data/ml_db.sqlite3`, `Data/Exports/dataset_module_a.csv`, отчеты

## Модуль Б
- Папки: `module_b`, `shared`, `Data`
- Запуск: `run_module_b.bat`
- Адрес: `http://127.0.0.1:5081`
- Вход: `viewer/viewer123`, `admin/admin123`

## Модуль В
- Папки: `module_v`, `shared`, `Data`
- Запуск: `run_module_v.bat`
- Результаты: `Data/Models/*.json`, `Docs/Reports/module_v_report.md`

## Модуль Г
- Папки: `module_g`, `shared`, `Data`
- Запуск: `run_module_g.bat`
- Адрес: `http://127.0.0.1:5082`

## Модуль Д
- Папки: `module_d`, `Docs`
- Запуск: `run_module_d.bat`
- Результаты: документация, карта критериев, README
""",
    )


def build_detailed_module_reports() -> None:
    # Защитные отчеты по модулям подготовлены вручную и уже лежат в Docs/Reports.
    # Здесь специально не перезаписываем их шаблонным содержимым, чтобы не
    # потерять детальную трассировку по критериям, файлам и результатам.
    load_database()
    return


def build_module_a_report(generated_count: int, suggestions: list[dict], counts_by_subject: dict[str, int]) -> str:
    aspect_rows = [
        ["Прием исходных данных", "Чтение файлов из корня, подпапок тем и вложенных папок.", "`shared/core.py`, `shared/parsers.py`, `module_a/agent.py`", "`Data/ml_db.sqlite3`, `Data/Exports/dataset_module_a.csv`", "В базе и в итоговом датасете видны все найденные материалы."],
        ["Поддержка форматов", "Поддержаны документы, таблицы, изображения, видео и текст.", "`shared/parsers.py`", "Поля `source_kind`, `text_material`, `source_path`", "По типу источника и извлеченному тексту в базе и экспортах."],
        ["Аналитическая запись", "Формируются `record_id`, предмет, тема, текст, summary, признаки и результат модерации.", "`shared/core.py`", "`Data/Exports/dataset_module_a.csv`", "В строках датасета и таблице `materials`."],
        ["Модерация", "Для каждого материала рассчитываются `requirement_checks` и `moderation_conclusion`.", "`shared/core.py`", "Таблица `materials`", "В полях модерации в SQLite."],
        ["Удаление дублей", "Повторы устраняются по `content_hash`.", "`shared/core.py`", "Таблица `materials`", "При повторном запуске число дублей не растет."],
        ["Prev/next", "Формируются `previous_record_id` и `next_record_id`.", "`shared/core.py`", "`Data/Exports/dataset_module_a.csv`", "В CSV-экспорте набора данных."],
        ["Факторы схожести", "Рассчитываются факторы, влияющие на схожесть, и `influence_score`.", "`module_a/agent.py`", "`Data/Exports/module_a_similarity_factors.csv`, `Docs/Reports/module_a_similarity_analysis.md`", "В отчете и CSV факторов схожести."],
        ["Описание атрибутов", "Для каждого атрибута есть расшифровка, назначение и тип данных.", "`module_a/agent.py`", "`Data/Exports/module_a_attribute_catalog.csv`, `Docs/Reports/module_a_attribute_analysis.md`", "В каталоге атрибутов и markdown-отчете."],
        ["Уникальные значения и частоты", "Подсчитываются уникальные значения, частоты и самые частые значения.", "`module_a/agent.py`", "`Data/Exports/module_a_attribute_frequency.csv`", "В CSV с частотами и отчете по атрибутам."],
        ["Анализ текстовых полей", "Считаются средняя, минимальная и максимальная длина, а также количество слов.", "`module_a/agent.py`", "`Data/Exports/module_a_text_stats.csv`, `Docs/Reports/charts/text_attribute_words.svg`", "В CSV статистики и на SVG-графике."],
        ["Визуализации", "Строятся столбчатые диаграммы распределений и текстовых характеристик.", "`module_a/agent.py`", "`Docs/Reports/charts/*.svg`", "В папке графиков модуля A."],
        ["Генерация новых материалов", "Агент ищет отсутствующие и недопредставленные темы, предлагает генерацию и добавляет новые записи.", "`module_a/agent.py`, `shared/core.py`", "SQLite: `suggestions`, `materials`, поле `generated=1`", "В таблице предложений, в базе и в отчете."],
    ]
    suggestion_lines = [
        f"{item.get('subject', 'Не указан')} / {item.get('topic', 'Без темы')} -> {item.get('reason', 'Причина не указана')} (generated={item.get('generated')})"
        for item in suggestions
    ] or ["Дополнительные предложения на момент генерации отчета не зафиксированы."]
    subject_lines = [f"{subject}: {count} материалов" for subject, count in sorted(counts_by_subject.items(), key=lambda pair: pair[0].lower())]
    return f"""# Отчет по модулю A

## Назначение
Модуль A отвечает за загрузку исходных учебных материалов, преобразование их в аналитический набор данных, автоматическую модерацию, анализ структуры атрибутов, анализ факторов схожести и генерацию недостающих материалов.

## Входные данные
{bullet_list([
    "Основная входная папка: `Data/Incoming/module_a/`.",
    "Допускаются одиночные файлы прямо в корне входной папки.",
    "Допускаются папки тем, папки дисциплин и вложенные папки.",
    "Поддерживаются документы, таблицы, изображения, видео и текст.",
])}

## Структура работы агента
1. Сканирование входной папки.
2. Определение типа каждого найденного источника.
3. Формирование аналитической записи учебного материала.
4. Автоматическая модерация материала.
5. Построение тематических связей и признаков `previous/next`.
6. Анализ атрибутов и факторов схожести.
7. Генерация недостающих материалов.
8. Сохранение результатов в SQLite, CSV и Markdown.

## Аспекты из критериев
{table(["Аспект критерия", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где это видно в результатах"], aspect_rows)}

## Что добавлено автоматически
{bullet_list([f"Количество сгенерированных материалов: **{generated_count}**.", f"Количество предложений на дополнение набора: **{len(suggestions)}**."])}

## Список предложений и сгенерированных тем
{bullet_list(suggestion_lines)}

## Материалы по дисциплинам
{bullet_list(subject_lines)}

## Основные артефакты модуля
{bullet_list([
    "`Data/ml_db.sqlite3` — основная SQLite-база.",
    "`Data/Exports/dataset_module_a.csv` — основной экспорт набора данных.",
    "`Data/Exports/module_a_attribute_catalog.csv` — каталог атрибутов.",
    "`Data/Exports/module_a_attribute_frequency.csv` — частоты и уникальные значения.",
    "`Data/Exports/module_a_text_stats.csv` — статистика текстовых полей.",
    "`Data/Exports/module_a_similarity_factors.csv` — факторы схожести.",
    "`Docs/Reports/module_a_attribute_analysis.md` — отчет по атрибутам.",
    "`Docs/Reports/module_a_similarity_analysis.md` — отчет по схожести.",
    "`Docs/Reports/charts/` — SVG-визуализации.",
])}

## Как показать модуль эксперту
{bullet_list([
    "Показать структуру `Data/Incoming/module_a/`.",
    "Запустить `run_module_a.bat`.",
    "Показать `Data/ml_db.sqlite3` и `Data/Exports/dataset_module_a.csv`.",
    "Открыть `Docs/Reports/module_a_attribute_analysis.md`.",
    "Открыть `Docs/Reports/module_a_similarity_analysis.md`.",
    "Показать SVG-графики из `Docs/Reports/charts/`.",
])}
"""


def build_module_b_report() -> str:
    aspect_rows = [
        ["Интерактивный дашборд", "Реализован локальный web-dashboard на `127.0.0.1:5081`.", "`module_b/agent.py`", "Локальный интерфейс в браузере", "На странице дашборда после запуска модуля Б."],
        ["Работа от SQLite", "Дашборд загружает данные из `Data/ml_db.sqlite3`.", "`module_b/agent.py`, `shared/storage.py`", "Сводные данные, карточки, таблицы", "В интерфейсе и в admin-режиме."],
        ["Разграничение прав", "Есть роли `viewer` и `admin`, вход осуществляется по паролю.", "`module_b/agent.py`, `Data/security.json`", "Форма логина и токен сессии", "Разная доступность блоков после входа разными ролями."],
        ["Покрытие тем", "Считается покрытие тем по дисциплинам относительно среднего уровня.", "`shared/core.py`", "Блок coverage", "В web-интерфейсе и в отчете модуля Б."],
        ["Доля сгенерированных материалов", "Считается общая доля и доля по предметам.", "`shared/core.py`", "Блок generated share", "В карточках dashboard и в итоговом отчете."],
        ["Распределение типов занятий", "Строится аналитика по `lesson_type`.", "`shared/core.py`", "Блок lesson types", "В дашборде по типам учебных материалов."],
        ["Выполнение методических требований", "Собирается pass rate по каждому требованию и по предметам.", "`shared/core.py`", "Блок requirements", "В дашборде и в markdown-отчете."],
        ["Лучшие и проблемные требования", "Выделяются TOP/LOW требования по предметам относительно среднего.", "`module_b/agent.py`, `shared/core.py`", "`Docs/Reports/module_b_report.md`", "В отчете и на экране аналитики."],
        ["Кластеры для параллельного изучения", "Материалы объединяются в группы для параллельного изучения.", "`shared/core.py`", "Кластерные метки в данных и графики", "В отчете и на графике `module_b_parallel_clusters.svg`."],
        ["Кластеры для последовательного изучения", "Материалы группируются в последовательные цепочки изучения.", "`shared/core.py`", "Кластерные метки и графики", "В отчете и на графике `module_b_sequential_clusters.svg`."],
        ["Группировка по сложности", "Материалы группируются по уровню сложности.", "`shared/core.py`", "Кластерные метки и графики", "В отчете и на графике `module_b_difficulty_clusters.svg`."],
        ["Оценка качества кластеризации", "Рассчитываются метрики `compactness` и `separation`.", "`shared/core.py`", "`Docs/Reports/module_b_report.md`, `Docs/Reports/charts/module_b_cluster_quality.svg`", "В отчетной таблице и на графике качества кластеров."],
        ["Визуальный анализ кластеров", "Сгенерированы отдельные SVG-визуализации по типам кластеров и качеству.", "`module_b/agent.py`", "`Docs/Reports/charts/module_b_*.svg`", "В папке графиков модуля Б."],
    ]
    return f"""# Отчет по модулю Б

## Назначение
Модуль Б отвечает за аналитический дашборд, разграничение прав доступа, расчет сводных показателей по набору данных и кластеризацию учебных материалов.

## Что получает модуль Б на вход
{bullet_list([
    "SQLite-базу `Data/ml_db.sqlite3`, сформированную модулем A.",
    "Настройки ролей и паролей из `Data/security.json`.",
    "Сводные признаки материалов, которые уже записаны в базу данных.",
])}

## Структура работы агента
1. Загружается база SQLite.
2. Рассчитываются аналитические показатели и кластерные признаки.
3. Поднимается локальный web-сервер дашборда.
4. Пользователь проходит вход с ролью `viewer` или `admin`.
5. Интерфейс показывает доступные блоки в зависимости от роли.

## Аспекты из критериев
{table(["Аспект критерия", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где это видно в результатах"], aspect_rows)}

## Основные артефакты модуля
{bullet_list([
    "`Docs/Reports/module_b_report.md` — подробный текстовый отчет по аналитике и кластеризации.",
    "`Docs/Reports/charts/module_b_parallel_clusters.svg` — визуализация параллельных кластеров.",
    "`Docs/Reports/charts/module_b_sequential_clusters.svg` — визуализация последовательных кластеров.",
    "`Docs/Reports/charts/module_b_difficulty_clusters.svg` — визуализация групп сложности.",
    "`Docs/Reports/charts/module_b_cluster_quality.svg` — график метрик качества кластеризации.",
])}

## Как показать модуль эксперту
{bullet_list([
    "Запустить `run_module_b.bat`.",
    "Открыть `http://127.0.0.1:5081`.",
    "Войти под `viewer` и показать обзорную аналитику.",
    "Затем войти под `admin` и показать таблицу материалов, служебные данные и путь к базе.",
    "Открыть `Docs/Reports/module_b_report.md` и SVG-графики из `Docs/Reports/charts/`.",
])}
"""


def build_module_v_report() -> str:
    aspect_rows = [
        ["Три задачи моделирования", "Обучаются модели для параллельного изучения, последовательного изучения и уровня сложности.", "`shared/core.py`, `module_v/agent.py`", "`Data/Models/*.json`, `Docs/Reports/module_v_report.md`", "В отчете по моделям и в каталоге `Data/Models`."],
        ["Три метода на каждую задачу", "Для каждой задачи используются `keyword_rule`, `nearest_neighbor`, `centroid_rule`.", "`shared/core.py`", "`Docs/Reports/module_v_report.md`", "В таблице методов внутри отчета."],
        ["Выбор лучшего метода", "Лучший метод определяется по `macro_f1` и `accuracy`.", "`shared/core.py`", "`Docs/Reports/module_v_report.md`", "В поле `selected_method` и в описании результатов."],
        ["Метрики качества", "Фиксируются `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`.", "`shared/core.py`, `module_v/agent.py`", "`Docs/Reports/module_v_report.md`", "В таблицах сравнения методов."],
        ["Дообучение моделей", "После обновления данных агент умеет обновить состояние моделей и их версии.", "`module_v/agent.py`, `shared/core.py`", "`Data/Models/*.json`", "В новых версиях артефактов моделей."],
        ["Контроль дрейфа", "Считается `drift_score` и формируется признак необходимости полного переобучения.", "`shared/core.py`, `module_v/agent.py`", "`Docs/Reports/module_v_report.md`", "В блоке про drift и retrain."],
        ["Полное переобучение", "Предусмотрен сценарий полного переобучения при превышении порога дрейфа.", "`module_v/agent.py`", "`Docs/Reports/module_v_report.md`, `Data/Models/*.json`", "По признаку `requires_full_retrain` и новой версии модели."],
        ["Версионирование моделей", "Каждая версия модели сохраняется как отдельный JSON-артефакт.", "`module_v/agent.py`", "`Data/Models/*.json`", "В папке моделей по именам файлов."],
        ["Оценка времени изучения", "Агент оценивает время освоения материалов, дисциплин и выбранного набора тем.", "`shared/core.py`, `module_v/agent.py`", "`Docs/Reports/module_v_report.md`", "В отчете модуля В и при построении траектории."],
        ["Индивидуальная траектория", "Формируется учебная траектория с порядком прохождения и визуализацией.", "`module_v/agent.py`", "`Docs/Reports/module_v_trajectory.md`", "В таблице траектории и `mermaid`-диаграмме."],
    ]
    return f"""# Отчет по модулю В

## Назначение
Модуль В отвечает за обучение и выбор моделей, контроль изменения данных, сохранение версий моделей и построение индивидуальных траекторий обучения.

## Что получает модуль В на вход
{bullet_list([
    "SQLite-базу `Data/ml_db.sqlite3` после выполнения модуля A.",
    "Признаки материалов и тематические связи, уже рассчитанные в базе.",
    "Необходимость обучить модели и подготовить траекторию для пользователя.",
])}

## Структура работы агента
1. Загружаются материалы и признаки из базы.
2. Для каждой задачи запускаются три метода.
3. Рассчитываются метрики качества и выбирается лучший метод.
4. Оценивается дрейф и при необходимости выполняется полное переобучение.
5. Версии моделей сохраняются в `Data/Models`.
6. Строится индивидуальная траектория обучения.

## Аспекты из критериев
{table(["Аспект критерия", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где это видно в результатах"], aspect_rows)}

## Основные артефакты модуля
{bullet_list([
    "`Docs/Reports/module_v_report.md` — отчет по обучению моделей и метрикам.",
    "`Docs/Reports/module_v_trajectory.md` — готовая траектория обучения.",
    "`Data/Models/*.json` — версии моделей и их артефакты.",
])}

## Как показать модуль эксперту
{bullet_list([
    "Запустить `run_module_v.bat`.",
    "Открыть `Docs/Reports/module_v_report.md` и показать три задачи, три метода и метрики качества.",
    "Открыть папку `Data/Models` и показать сохраненные версии моделей.",
    "Открыть `Docs/Reports/module_v_trajectory.md` и показать траекторию и оценку времени.",
])}
"""


def build_module_g_report() -> str:
    aspect_rows = [
        ["API с валидацией", "Все входные параметры API проходят базовую валидацию.", "`module_g/agent.py`", "HTTP-ответы локального API", "На реальных запросах к API и в web-интерфейсе."],
        ["Загрузка готовых моделей", "Модуль использует уже подготовленные модели и не переобучает их при старте.", "`module_g/agent.py`", "Эндпоинт `/api/health`", "В поле `models_loaded=true`."],
        ["Карточка материала", "Реализован просмотр аналитической карточки материала.", "`module_g/agent.py`", "`GET /api/material`", "В JSON-ответе API и в web-форме интерфейса."],
        ["Оценка по методическим критериям", "Реализована отдельная выдача результатов модерации материала.", "`module_g/agent.py`", "`GET /api/moderation`", "В JSON-ответе и в пользовательском интерфейсе."],
        ["Параллельное изучение", "Реализована выдача данных о возможности параллельного изучения.", "`module_g/agent.py`", "`GET /api/parallel`", "В API и в web-интерфейсе."],
        ["Последовательное изучение", "Реализована выдача данных о последовательности изучения.", "`module_g/agent.py`", "`GET /api/sequential`", "В API и в web-интерфейсе."],
        ["Оценка сложности", "Реализован отдельный маршрут для оценки уровня сложности.", "`module_g/agent.py`", "`GET /api/difficulty`", "В API и в web-интерфейсе."],
        ["Оценка времени", "Реализован отдельный маршрут для оценки времени освоения.", "`module_g/agent.py`", "`POST /api/time-estimate`", "В JSON-ответе и в интерфейсе."],
        ["Индивидуальная траектория", "Реализован маршрут генерации траектории обучения.", "`module_g/agent.py`", "`POST /api/trajectory`", "В API и в интерфейсе модуля Г."],
        ["Локальное приложение", "Поверх API работает web-страница для пользователя.", "`module_g/agent.py`", "`http://127.0.0.1:5082`", "В браузере после запуска модуля."],
        ["Интеграционный bridge", "Подготовлен bridge-файл для подключения Telegram-сценария.", "`module_g/telegram_bridge.py`", "Отдельный python-файл в проекте", "В структуре проекта и документации."],
        ["Документация API", "Собрана отдельная документация с маршрутами и примерами использования.", "`module_g/agent.py`, `module_d/agent.py`", "`Docs/Reports/api_reference.md`", "В отдельном markdown-файле документации API."],
    ]
    return f"""# Отчет по модулю Г

## Назначение
Модуль Г предоставляет локальный API и web-интерфейс, через которые можно использовать уже подготовленные материалы и модели без повторного обучения.

## Что получает модуль Г на вход
{bullet_list([
    "SQLite-базу `Data/ml_db.sqlite3` с материалами и признаками.",
    "Готовые версии моделей из `Data/Models`.",
    "Пользовательские запросы на просмотр материала, модерацию, сложность, время и траекторию.",
])}

## Структура работы агента
1. Модуль загружает базу и готовые модели.
2. Поднимается локальный API.
3. Поверх API открывается web-интерфейс.
4. Пользователь выполняет запросы к маршрутам API или через форму на странице.

## Аспекты из критериев
{table(["Аспект критерия", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где это видно в результатах"], aspect_rows)}

## Основные артефакты модуля
{bullet_list([
    "`Docs/Reports/module_g_report.md` — подробный отчет по модулю Г.",
    "`Docs/Reports/api_reference.md` — документация по маршрутам API.",
    "`module_g/telegram_bridge.py` — bridge для сценария внешней интеграции.",
])}

## Как показать модуль эксперту
{bullet_list([
    "Запустить `run_module_g.bat`.",
    "Открыть `http://127.0.0.1:5082`.",
    "Показать проверку здоровья API через `/api/health`.",
    "Показать запрос карточки материала, модерации, сложности, времени и траектории.",
    "Открыть `Docs/Reports/api_reference.md` и показать список маршрутов.",
])}
"""


def build_module_d_report() -> str:
    aspect_rows = [
        ["Главный README", "Собрана единая инструкция по проекту, входным данным, запуску и сборке.", "`module_d/agent.py`", "`README.md`", "В корне проекта."],
        ["Системная документация", "Описаны архитектура, роли, схема работы и взаимодействие модулей.", "`module_d/agent.py`", "`Docs/Reports/system_documentation.md`", "В системной документации."],
        ["Сценарий демонстрации", "Подготовлен пошаговый сценарий показа решения до 10 минут.", "`module_d/agent.py`", "`Docs/Reports/demo_script.md`", "В файле демонстрации."],
        ["Шаги по модулям на чемпионате", "Подготовлена инструкция, что запускать и что показывать по каждому модулю.", "`module_d/agent.py`", "`Docs/Reports/championship_steps.md`", "В отдельном файле шагов чемпионата."],
        ["Карта критериев", "Подготовлена сводная карта соответствия аспектов критериям оценивания.", "`module_d/agent.py`", "`Docs/Reports/criteria_map.md`", "В отдельном markdown-файле."],
        ["Bat-файлы запуска", "Для каждого модуля есть запуск одной командой.", "`module_d/agent.py`", "`run_module_a.bat`, `run_module_b.bat`, `run_module_v.bat`, `run_module_g.bat`, `run_module_d.bat`", "В корне проекта."],
        ["Сборка exe", "Подготовлен единый build-скрипт для упаковки модулей в исполняемые файлы.", "`module_d/agent.py`", "`build_all.bat`", "В корне проекта и в папке `Distribution` после сборки."],
    ]
    return f"""# Отчет по модулю Д

## Назначение
Модуль Д отвечает за документацию проекта, материалы для защиты, единые инструкции запуска и подготовку сборки поставки.

## Что делает модуль Д
1. Пересобирает README и системную документацию.
2. Формирует отчеты по модулям в единой структуре.
3. Формирует карту критериев оценивания.
4. Генерирует bat-файлы запуска.
5. Генерирует build-скрипт для упаковки `exe`.

## Аспекты из критериев
{table(["Аспект критерия", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где это видно в результатах"], aspect_rows)}

## Основные артефакты модуля
{bullet_list([
    "`README.md`",
    "`Docs/Reports/system_documentation.md`",
    "`Docs/Reports/demo_script.md`",
    "`Docs/Reports/championship_steps.md`",
    "`Docs/Reports/criteria_map.md`",
    "`run_module_*.bat`",
    "`build_all.bat`",
])}

## Как показать модуль эксперту
{bullet_list([
    "Запустить `run_module_d.bat`.",
    "Открыть `README.md` и показать общую инструкцию.",
    "Открыть `Docs/Reports/system_documentation.md` и показать архитектуру.",
    "Открыть `Docs/Reports/championship_steps.md` и показать порядок работы на чемпионате.",
    "Открыть `Docs/Reports/criteria_map.md` и показать соответствие аспектов критериям.",
])}
"""


def build_criteria_map() -> str:
    rows = [
        ["A", "Загрузка материалов", "Прием файлов из корня, подпапок и вложенных папок", "`shared/core.py`, `shared/parsers.py`, `module_a/agent.py`", "`Data/ml_db.sqlite3`, `Data/Exports/dataset_module_a.csv`", "В SQLite и CSV после запуска модуля A"],
        ["A", "Поддержка форматов", "Обработка `txt/md/html/json/csv/doc/docx/pdf/xlsx/xls/xlsm/png/jpg/jpeg/svg/mp4/avi/mov/mkv/wmv/m4v/webm`", "`shared/parsers.py`", "Поля `source_kind`, `text_material`", "В базе и экспорте набора данных"],
        ["A", "Аналитическая запись", "Формирование карточки материала с идентификатором, темой, предметом и текстом", "`shared/core.py`", "`Data/Exports/dataset_module_a.csv`", "В строках датасета"],
        ["A", "Модерация", "Автоматическая проверка требований и итоговый вывод", "`shared/core.py`", "Поля `requirement_checks`, `moderation_conclusion`", "В SQLite и модульном отчете"],
        ["A", "Антидубли", "Дедупликация по `content_hash`", "`shared/core.py`", "Таблица `materials`", "При повторном запуске модуль не размножает записи"],
        ["A", "Prev/next", "Признаки предыдущего и следующего материала", "`shared/core.py`", "`dataset_module_a.csv`", "В экспортном датасете"],
        ["A", "Анализ атрибутов", "Описание признаков, типы данных, частоты, статистика текстовых полей", "`module_a/agent.py`", "`module_a_attribute_analysis.md`, `module_a_attribute_catalog.csv`, `module_a_attribute_frequency.csv`, `module_a_text_stats.csv`", "В отчете по атрибутам и CSV-файлах"],
        ["A", "Факторы схожести", "Анализ признаков, влияющих на схожесть учебных материалов", "`module_a/agent.py`", "`module_a_similarity_analysis.md`, `module_a_similarity_factors.csv`", "В отчете по схожести"],
        ["A", "Визуализации", "Графики распределений и текстовых характеристик", "`module_a/agent.py`", "`Docs/Reports/charts/*.svg`", "В папке графиков"],
        ["A", "Генерация новых материалов", "Поиск отсутствующих и недопредставленных тем, генерация и добавление в БД", "`module_a/agent.py`, `shared/core.py`", "SQLite `suggestions`, `materials`, поле `generated=1`", "В базе, датасете и отчете модуля A"],
        ["Б", "Дашборд", "Локальный интерактивный web-dashboard", "`module_b/agent.py`", "`http://127.0.0.1:5081`", "В браузере"],
        ["Б", "SQLite", "Дашборд строится на базе `Data/ml_db.sqlite3`", "`module_b/agent.py`, `shared/storage.py`", "Данные dashboard", "В admin-режиме и аналитических блоках"],
        ["Б", "Права доступа", "Роли `viewer/admin` и вход по паролю", "`module_b/agent.py`, `Data/security.json`", "Форма входа и токены", "На странице логина и после авторизации"],
        ["Б", "Покрытие тем", "Сравнение наполнения предметов относительно среднего", "`shared/core.py`", "Блок coverage", "В dashboard и отчете"],
        ["Б", "Доля генерации", "Доля сгенерированных материалов общая и по предметам", "`shared/core.py`", "Блок generated share", "В dashboard и отчете"],
        ["Б", "Типы занятий", "Распределение по `lesson_type`", "`shared/core.py`", "Блок lesson types", "В dashboard"],
        ["Б", "Методические требования", "Сводка pass rate по требованиям", "`shared/core.py`", "Блок requirements", "В dashboard и отчете"],
        ["Б", "TOP/LOW требования", "Лидирующие и проблемные требования по предметам", "`module_b/agent.py`, `shared/core.py`", "`Docs/Reports/module_b_report.md`", "В отчете модуля Б"],
        ["Б", "Кластеризация", "Параллельные, последовательные и сложностные кластеры", "`shared/core.py`", "Кластерные метки и SVG-графики", "В отчете и графиках"],
        ["Б", "Метрики кластеризации", "Расчет `compactness` и `separation`", "`shared/core.py`", "`module_b_report.md`, `module_b_cluster_quality.svg`", "В отчете и графике качества"],
        ["В", "Три задачи", "Модели параллельности, последовательности и сложности", "`shared/core.py`, `module_v/agent.py`", "`Data/Models/*.json`, `module_v_report.md`", "В отчете модуля В"],
        ["В", "Три метода", "Три метода на каждую задачу", "`shared/core.py`", "`module_v_report.md`", "В сравнительной таблице методов"],
        ["В", "Выбор лучшего метода", "Выбор по `macro_f1` и `accuracy`", "`shared/core.py`", "`module_v_report.md`", "В строке selected method"],
        ["В", "Метрики качества", "Accuracy, precision, recall, F1", "`shared/core.py`, `module_v/agent.py`", "`module_v_report.md`", "В отчете модуля В"],
        ["В", "Дообучение и дрейф", "Контроль `drift_score` и сценарий полного переобучения", "`shared/core.py`, `module_v/agent.py`", "`module_v_report.md`", "В блоке drift/retrain"],
        ["В", "Версии моделей", "Сохранение артефактов моделей в отдельных файлах", "`module_v/agent.py`", "`Data/Models/*.json`", "В каталоге моделей"],
        ["В", "Оценка времени", "Оценка времени на материалы, дисциплины и наборы", "`shared/core.py`", "`module_v_report.md`", "В отчете и траектории"],
        ["В", "Траектория", "Построение и визуализация индивидуальной траектории", "`module_v/agent.py`", "`module_v_trajectory.md`", "В markdown и mermaid"],
        ["Г", "API", "Локальный API с валидацией входных данных", "`module_g/agent.py`", "HTTP-ответы", "На запросах к API"],
        ["Г", "Готовые модели", "Использование уже обученных моделей без переобучения", "`module_g/agent.py`", "`/api/health`", "В поле `models_loaded=true`"],
        ["Г", "Карточка материала", "Маршрут просмотра материала", "`module_g/agent.py`", "`GET /api/material`", "В API и UI"],
        ["Г", "Модерация", "Маршрут просмотра методической оценки", "`module_g/agent.py`", "`GET /api/moderation`", "В API и UI"],
        ["Г", "Параллельное изучение", "Маршрут оценки параллельности", "`module_g/agent.py`", "`GET /api/parallel`", "В API и UI"],
        ["Г", "Последовательное изучение", "Маршрут оценки последовательности", "`module_g/agent.py`", "`GET /api/sequential`", "В API и UI"],
        ["Г", "Сложность", "Маршрут оценки сложности", "`module_g/agent.py`", "`GET /api/difficulty`", "В API и UI"],
        ["Г", "Время", "Маршрут оценки времени освоения", "`module_g/agent.py`", "`POST /api/time-estimate`", "В API и UI"],
        ["Г", "Траектория", "Маршрут построения траектории", "`module_g/agent.py`", "`POST /api/trajectory`", "В API и UI"],
        ["Г", "Локальный интерфейс", "Web-страница поверх API", "`module_g/agent.py`", "`http://127.0.0.1:5082`", "В браузере"],
        ["Г", "Bridge", "Подготовлен bridge-файл под Telegram-интеграцию", "`module_g/telegram_bridge.py`", "Python-файл bridge", "В структуре проекта"],
        ["Г", "Документация API", "Описание маршрутов и примеров", "`module_g/agent.py`, `module_d/agent.py`", "`Docs/Reports/api_reference.md`", "В отдельном markdown-файле"],
        ["Д", "README", "Единая инструкция по запуску и использованию", "`module_d/agent.py`", "`README.md`", "В корне проекта"],
        ["Д", "Системная документация", "Архитектура, роли и схема работы", "`module_d/agent.py`", "`Docs/Reports/system_documentation.md`", "В отчете по документации"],
        ["Д", "Сценарий демонстрации", "Пошаговый demo script", "`module_d/agent.py`", "`Docs/Reports/demo_script.md`", "В файле сценария"],
        ["Д", "Шаги чемпионата", "Инструкция по модульному показу", "`module_d/agent.py`", "`Docs/Reports/championship_steps.md`", "В файле шагов чемпионата"],
        ["Д", "Карта критериев", "Сводка аспектов, файлов и результатов", "`module_d/agent.py`", "`Docs/Reports/criteria_map.md`", "В текущем файле"],
        ["Д", "Bat-файлы", "Запуск каждого модуля одной командой", "`module_d/agent.py`", "`run_module_*.bat`", "В корне проекта"],
        ["Д", "Build-script", "Сборка исполняемых файлов в `Distribution`", "`module_d/agent.py`", "`build_all.bat`", "В корне проекта и после сборки"],
    ]
    return f"""# Карта критериев

Этот файл нужен для защиты: по каждой группе аспектов здесь указано, что выполнено, где находится реализация, где лежит результат и где его показывать эксперту.

{table(["Модуль", "Аспект", "Что выполнено", "В каком файле выполняется", "Где лежит результат", "Где видно в результатах"], rows)}
"""


def build_launch_files() -> None:
    scripts = {
        "run_module_a.bat": "@echo off\r\npython module_a\\agent.py\r\npause\r\n",
        "run_module_b.bat": "@echo off\r\npython module_b\\agent.py\r\npause\r\n",
        "run_module_v.bat": "@echo off\r\npython module_v\\agent.py\r\npause\r\n",
        "run_module_g.bat": "@echo off\r\npython module_g\\agent.py\r\npause\r\n",
        "run_module_d.bat": "@echo off\r\npython module_d\\agent.py\r\npause\r\n",
    }
    for name, content in scripts.items():
        write(ROOT_DIR / name, content)


def build_build_scripts() -> None:
    write(
        ROOT_DIR / "build_all.bat",
        """@echo off
set PYTHONUTF8=1
pyinstaller --noconfirm --onefile --name module_a_agent module_a\\agent.py
pyinstaller --noconfirm --onefile --name module_b_agent module_b\\agent.py
pyinstaller --noconfirm --onefile --name module_v_agent module_v\\agent.py
pyinstaller --noconfirm --onefile --name module_g_agent module_g\\agent.py
pyinstaller --noconfirm --onefile --name module_d_agent module_d\\agent.py
if not exist Distribution mkdir Distribution
copy /Y dist\\module_a_agent.exe Distribution\\module_a_agent.exe
copy /Y dist\\module_b_agent.exe Distribution\\module_b_agent.exe
copy /Y dist\\module_v_agent.exe Distribution\\module_v_agent.exe
copy /Y dist\\module_g_agent.exe Distribution\\module_g_agent.exe
copy /Y dist\\module_d_agent.exe Distribution\\module_d_agent.exe
""",
    )
    write(
        REPORTS_DIR / "delivery_notes.md",
        """# Пояснения по поставке

- Итоговая поставка может включать исходники Python и собранные `exe`
- Для оффлайн-использования без Python нужно выполнить `build_all.bat`
- Готовые исполняемые файлы складываются в `Distribution`
- Во время работы агенты не должны скачивать зависимости из сети
""",
    )


if __name__ == "__main__":
    main()
