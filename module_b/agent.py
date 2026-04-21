from __future__ import annotations

import json
import secrets
import sys
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import DATABASE_PATH, REPORTS_DIR
from shared.core import Material, build_clusters, build_dashboard_payload, criteria_lines
from shared.storage import ensure_security_file, load_database, save_database


HOST = "127.0.0.1"
PORT = 5081
SESSIONS: dict[str, str] = {}


def load_materials() -> tuple[list[Material], dict]:
    # Загружаем материалы из SQLite и сразу пересчитываем кластеры.
    # Так дашборд всегда показывает актуальную разметку после модулей А и В.
    database = load_database()
    materials = [Material(**item) for item in database["materials"]]
    clusters = build_clusters(materials)
    database["clusters"] = clusters
    save_database(database)
    return materials, clusters


def build_cluster_distributions(materials: list[Material]) -> dict[str, list[dict[str, object]]]:
    # Готовим сводки распределения материалов по кластерам,
    # чтобы использовать их и в браузере, и в экспортируемых SVG-файлах.
    parallel_counter = Counter(material.parallel_cluster for material in materials)
    sequential_counter = Counter(material.sequential_cluster for material in materials)
    difficulty_counter = Counter(material.difficulty_level for material in materials)
    return {
        "parallel": [{"cluster": name, "count": count} for name, count in parallel_counter.most_common(20)],
        "sequential": [{"cluster": name, "count": count} for name, count in sequential_counter.most_common(20)],
        "difficulty": [{"cluster": name, "count": count} for name, count in difficulty_counter.most_common(20)],
    }


def build_subject_lesson_matrix(payload: dict) -> list[dict[str, object]]:
    # Преобразуем lesson_types из общего payload в явную матрицу,
    # чтобы фронтенду было проще рисовать диаграммы по выбранной дисциплине.
    rows: list[dict[str, object]] = []
    for item in payload.get("lesson_types", []):
        rows.append({"subject": item["subject"], "types": item.get("types", {})})
    return rows


def build_view_payload(role: str) -> dict:
    # Формируем единый JSON для интерактивного дашборда.
    # В admin-режиме дополнительно показываем таблицу материалов и путь к БД.
    materials, clusters = load_materials()
    payload = build_dashboard_payload(materials, clusters)
    payload["role"] = role
    payload["database_path"] = str(DATABASE_PATH)
    payload["cluster_distributions"] = build_cluster_distributions(materials)
    payload["subject_lesson_matrix"] = build_subject_lesson_matrix(payload)
    payload["materials"] = [
        {
            "record_id": item.record_id,
            "subject": item.subject,
            "topic": item.topic,
            "difficulty_level": item.difficulty_level,
            "parallel_cluster": item.parallel_cluster,
            "sequential_cluster": item.sequential_cluster,
            "estimated_minutes": item.estimated_minutes,
            "moderation_conclusion": item.moderation_conclusion,
        }
        for item in materials
    ] if role == "admin" else []
    save_report(materials, clusters)
    return payload


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_bar_chart_svg(path: Path, title: str, pairs: list[tuple[str, float]], unit_label: str) -> None:
    # Экспортируем статический SVG-график для отчета и офлайн-защиты.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pairs:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='120'></svg>", encoding="utf-8")
        return

    width = 980
    row_height = 34
    top = 60
    left = 360
    bar_area = 520
    max_value = max(abs(float(value)) for _, value in pairs) or 1.0
    height = top + row_height * len(pairs) + 28
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Segoe UI,sans-serif;fill:#1f2e38}.label{font-size:13px}.title{font-size:20px;font-weight:700}.value{font-size:12px}</style>",
        f"<rect width='{width}' height='{height}' fill='#fcfaf4'/>",
        f"<text x='24' y='34' class='title'>{escape_xml(title)}</text>",
    ]
    for index, (label, raw_value) in enumerate(pairs):
        y = top + index * row_height
        value = float(raw_value)
        bar_width = abs(value) / max_value * bar_area if max_value else 0
        lines.append(f"<text x='24' y='{y + 18}' class='label'>{escape_xml(str(label)[:48])}</text>")
        lines.append(f"<rect x='{left}' y='{y + 4}' width='{bar_area}' height='18' rx='9' fill='#e4ece8'/>")
        lines.append(f"<rect x='{left}' y='{y + 4}' width='{bar_width:.1f}' height='18' rx='9' fill='#1b8a5a'/>")
        lines.append(f"<text x='{left + bar_area + 10}' y='{y + 18}' class='value'>{value:.3f} {escape_xml(unit_label)}</text>")
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_cluster_visuals(materials: list[Material], clusters: dict) -> None:
    # Создаем SVG-файлы визуального анализа кластеризации,
    # чтобы эти же диаграммы можно было показать без браузерного UI.
    charts_dir = REPORTS_DIR / "charts"
    distributions = build_cluster_distributions(materials)
    write_bar_chart_svg(
        charts_dir / "module_b_parallel_clusters.svg",
        "Распределение материалов по параллельным кластерам",
        [(item["cluster"], item["count"]) for item in distributions["parallel"]],
        "мат.",
    )
    write_bar_chart_svg(
        charts_dir / "module_b_sequential_clusters.svg",
        "Распределение материалов по последовательным кластерам",
        [(item["cluster"], item["count"]) for item in distributions["sequential"]],
        "мат.",
    )
    write_bar_chart_svg(
        charts_dir / "module_b_difficulty_clusters.svg",
        "Распределение материалов по кластерам сложности",
        [(item["cluster"], item["count"]) for item in distributions["difficulty"]],
        "мат.",
    )
    quality_pairs = [
        ("Параллельная silhouette", clusters["parallel_metric"]["silhouette_score"]),
        ("Параллельная DBI", clusters["parallel_metric"]["davies_bouldin_index"]),
        ("Параллельная CHI", clusters["parallel_metric"]["calinski_harabasz_score"]),
        ("Последовательная silhouette", clusters["sequential_metric"]["silhouette_score"]),
        ("Последовательная DBI", clusters["sequential_metric"]["davies_bouldin_index"]),
        ("Последовательная CHI", clusters["sequential_metric"]["calinski_harabasz_score"]),
        ("Сложность silhouette", clusters["difficulty_metric"]["silhouette_score"]),
        ("Сложность DBI", clusters["difficulty_metric"]["davies_bouldin_index"]),
        ("Сложность CHI", clusters["difficulty_metric"]["calinski_harabasz_score"]),
    ]
    write_bar_chart_svg(
        charts_dir / "module_b_cluster_quality.svg",
        "Метрики качества кластеризации",
        quality_pairs,
        "score",
    )


def save_report(materials: list[Material], clusters: dict) -> None:
    # Формируем подробный отчет по модулю Б:
    # критерии, обоснование разметки, метрики, диаграммы и выходные файлы.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_cluster_visuals(materials, clusters)

    subject_requirements: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for material in materials:
        for check in material.requirement_checks:
            subject_requirements[material.subject][check["requirement"]].append(check["passed"])

    requirement_lines: list[str] = []
    for subject, requirement_pool in sorted(subject_requirements.items()):
        ranked = sorted(
            requirement_pool.items(),
            key=lambda item: sum(item[1]) / max(1, len(item[1])),
            reverse=True,
        )
        best = ", ".join(name for name, _ in ranked[:3]) or "нет"
        worst = ", ".join(name for name, _ in ranked[-3:]) or "нет"
        requirement_lines.append(f"- {subject}: лучшие требования — {best}; проблемные требования — {worst}.")

    relationship_lines: list[str] = []
    for item in clusters.get("subject_relationships", []):
        relationship_lines.append(
            f"- {item['subject']}: параллельно -> {', '.join(item['parallel_with']) or 'нет'}; "
            f"последовательно -> {', '.join(item['sequential_with']) or 'нет'}; "
            f"не рекомендуется параллельно -> {', '.join(item['restricted_with']) or 'нет'}. "
            f"Основание: {item['reasoning']}"
        )

    lines = [
        "# Отчет по модулю Б",
        "",
        "## Критерии и аспекты",
        "| Аспект | Где выполнено | Что лежит на выходе | Как проверить |",
        "|---|---|---|---|",
        "| Интерактивный дашборд | `module_b/agent.py` | локальный web-интерфейс `http://127.0.0.1:5081` | запустить модуль Б и открыть страницу |",
        "| Разграничение прав доступа | `module_b/agent.py`, `Data/security.json` | роли `viewer` и `admin` | войти под разными ролями |",
        "| Интерактивные диаграммы | `module_b/agent.py` | SVG-графики в браузере и экспортированные SVG в `Docs/Reports/charts` | открыть дашборд и SVG-файлы |",
        "| Сравнение методов кластеризации | `shared/clustering.py`, `module_b/agent.py` | метрики и сравнение в отчете и дашборде | открыть раздел сравнения методов |",
        "| Совместимость дисциплин | `shared/clustering.py`, `module_b/agent.py` | селектор дисциплины на дашборде и раздел отчета | выбрать дисциплину в блоке `Совместимость дисциплин` |",
        "",
        "## Реализованный функционал",
        "- Дашборд запускается на `http://127.0.0.1:5081` и работает поверх SQLite `Data/ml_db.sqlite3`.",
        "- Вход выполняется по паролю, доступны роли `viewer` и `admin`.",
        "- В браузере отображаются интерактивные bar chart, donut chart, диаграммы сравнения методов и диаграммы распределения кластеров.",
        "- В дашборде можно выбрать дисциплину и увидеть, с какими дисциплинами ее допустимо изучать параллельно или последовательно.",
        "",
        "## Обоснование метода кластеризации",
        "- Для каждой задачи сравниваются `KMeans`, `AgglomerativeClustering` и `SpectralClustering`.",
        "- Параллельное изучение определяется через независимость дисциплин и низкое пересечение терминологии.",
        "- Последовательное изучение определяется через логическое продолжение и тематическую связь.",
        "- Кластеры сложности строятся по числовым признакам: объему текста, времени освоения, методическому баллу и количеству медиаобъектов.",
        "",
        "## Метрики качества кластеризации",
        "- Используются `Silhouette Score`, `Davies–Bouldin Index` и `Calinski–Harabasz Score`.",
        "- `Silhouette Score` удобен для понятного сравнения близости объектов к своему кластеру и удаленности от соседних.",
        "- `Davies–Bouldin Index` нужен для оценки компактности внутри кластеров и их разделимости. Меньше — лучше.",
        "- `Calinski–Harabasz Score` показывает качество разделения кластеров и похожесть объектов внутри них. Больше — лучше.",
        f"- Параллельная кластеризация: silhouette={clusters['parallel_metric']['silhouette_score']}, DBI={clusters['parallel_metric']['davies_bouldin_index']}, CHI={clusters['parallel_metric']['calinski_harabasz_score']}.",
        f"- Последовательная кластеризация: silhouette={clusters['sequential_metric']['silhouette_score']}, DBI={clusters['sequential_metric']['davies_bouldin_index']}, CHI={clusters['sequential_metric']['calinski_harabasz_score']}.",
        f"- Сложность: silhouette={clusters['difficulty_metric']['silhouette_score']}, DBI={clusters['difficulty_metric']['davies_bouldin_index']}, CHI={clusters['difficulty_metric']['calinski_harabasz_score']}.",
        "",
        "## Совместимость дисциплин",
        *relationship_lines,
        "",
        "## Визуальный анализ",
        "- `Docs/Reports/charts/module_b_parallel_clusters.svg` — распределение материалов по параллельным кластерам.",
        "- `Docs/Reports/charts/module_b_sequential_clusters.svg` — распределение материалов по последовательным кластерам.",
        "- `Docs/Reports/charts/module_b_difficulty_clusters.svg` — распределение материалов по кластерам сложности.",
        "- `Docs/Reports/charts/module_b_cluster_quality.svg` — метрики качества кластеризации.",
        "",
        "## Анализ выполнения требований",
        *requirement_lines,
        "",
        "## Где выполнены критерии",
        *[f"- {line}" for line in criteria_lines("B")],
    ]
    (REPORTS_DIR / "module_b_report.md").write_text("\n".join(lines), encoding="utf-8")


HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Модуль Б</title>
  <style>
    :root{
      --bg:#f2eee4; --card:#fffdf8; --line:#d3c8b7; --ink:#1b2a34; --accent:#1b8a5a; --muted:#6f7c86;
      --a1:#1b8a5a; --a2:#d46a4b; --a3:#2969b0; --a4:#c8a12e; --a5:#805ad5;
    }
    *{box-sizing:border-box}
    body{font-family:"Segoe UI",sans-serif;margin:0;background:
      radial-gradient(circle at top left, rgba(27,138,90,.10), transparent 22%),
      radial-gradient(circle at top right, rgba(41,105,176,.12), transparent 20%),
      linear-gradient(145deg,#f2eee4,#eef6f7);color:var(--ink)}
    header{padding:20px 28px;position:sticky;top:0;background:rgba(255,253,248,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(8px);z-index:10}
    h1,h2,h3,h4{margin:0 0 12px}
    p{margin:8px 0}
    main{padding:20px 28px;display:grid;gap:16px}
    .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
    .card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:0 8px 24px rgba(27,42,52,.05)}
    .wide{grid-column:1/-1}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#e7f4ee;margin:4px;font-size:13px}
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
    .metric{padding:14px;border-radius:16px;background:linear-gradient(135deg,#faf7ef,#eef8f4);border:1px solid var(--line)}
    .metric .value{font-size:28px;font-weight:700}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    #dashboard{display:none}
    #admin{display:none}
    input,select{padding:10px;border:1px solid var(--line);border-radius:12px;width:100%}
    button{padding:10px 16px;border:none;border-radius:12px;background:#0f7c59;color:#fff;cursor:pointer}
    .auth-grid{display:grid;gap:12px;max-width:420px}
    .chart-wrap{display:grid;gap:10px}
    .svg-host{width:100%;overflow:auto}
    .legend{display:flex;flex-wrap:wrap;gap:8px}
    .legend span{display:inline-flex;align-items:center;gap:6px;font-size:13px}
    .legend i{display:inline-block;width:12px;height:12px;border-radius:3px}
    .muted{color:var(--muted);font-size:13px}
    .method-table td,.method-table th{font-size:13px}
  </style>
</head>
<body>
  <header>
    <h1>Модуль Б. Аналитический дашборд</h1>
    <div id="updated">Ожидание входа...</div>
  </header>
  <main>
    <section class="card" id="loginBox">
      <h2>Вход</h2>
      <div class="auth-grid">
        <select id="role">
          <option value="viewer">viewer</option>
          <option value="admin">admin</option>
        </select>
        <input id="password" type="password" placeholder="Пароль">
        <button onclick="login()">Войти</button>
        <div id="authError"></div>
      </div>
    </section>

    <section id="dashboard">
      <div class="metric-grid">
        <div class="metric"><div class="muted">Материалов</div><div class="value" id="metricMaterials">0</div></div>
        <div class="metric"><div class="muted">Сгенерировано</div><div class="value" id="metricGenerated">0%</div></div>
        <div class="metric"><div class="muted">Дисциплин</div><div class="value" id="metricSubjects">0</div></div>
        <div class="metric"><div class="muted">Требований</div><div class="value" id="metricRequirements">0</div></div>
      </div>

      <div class="grid">
        <section class="card">
          <h2>Покрытие тем относительно среднего</h2>
          <div class="svg-host" id="coverageChart"></div>
          <div class="muted">Значение показывает количество материалов и коэффициент относительно среднего уровня покрытия.</div>
        </section>

        <section class="card">
          <h2>Доля автоматически сгенерированных материалов</h2>
          <div class="svg-host" id="generatedChart"></div>
          <div class="muted">Все значения подписаны в процентах.</div>
        </section>

        <section class="card">
          <h2>Распределение по типам занятий</h2>
          <label class="muted">Дисциплина для анализа</label>
          <select id="lessonSubjectSelector" onchange="renderLessonTypes()"></select>
          <div class="svg-host" id="lessonTypesChart"></div>
          <div id="lessonTypesLegend" class="legend"></div>
        </section>

        <section class="card">
          <h2>Обеспеченность требований методических рекомендаций</h2>
          <div class="svg-host" id="requirementsChart"></div>
          <div class="muted">Показатели указаны в процентах выполнения.</div>
        </section>

        <section class="card">
          <h2>TOP требований по дисциплинам</h2>
          <div id="tops"></div>
        </section>

        <section class="card">
          <h2>Совместимость дисциплин</h2>
          <label class="muted">Выберите дисциплину</label>
          <select id="subjectSelector" onchange="renderSubjectRelations()"></select>
          <div id="subjectRelations"></div>
        </section>

        <section class="card">
          <h2>Качество кластеризации</h2>
          <div class="svg-host" id="clusterMetricsChart"></div>
          <div id="clusters"></div>
        </section>

        <section class="card wide">
          <h2>Распределение кластеров</h2>
          <div class="svg-host" id="clusterDistributionChart"></div>
        </section>

        <section class="card wide">
          <h2>Сравнение методов кластеризации</h2>
          <div id="methodComparison"></div>
        </section>

        <section class="card wide" id="admin">
          <h2>Материалы и служебные данные</h2>
          <div id="materials"></div>
        </section>
      </div>
    </section>
  </main>

  <script>
    let token = localStorage.getItem('dashboard_token') || '';
    const COLORS = ['#1b8a5a','#d46a4b','#2969b0','#c8a12e','#805ad5','#1f9fb5','#b65f7a','#5b7c2f'];

    async function login() {
      const role = document.getElementById('role').value;
      const password = document.getElementById('password').value;
      const response = await fetch('/api/login', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({role, password})
      });
      const payload = await response.json();
      if (!response.ok) {
        document.getElementById('authError').textContent = payload.error || 'Ошибка входа';
        return;
      }
      token = payload.token;
      localStorage.setItem('dashboard_token', token);
      render();
    }

    function makeBarChart(items, config) {
      const width = config.width || 760;
      const rowHeight = config.rowHeight || 34;
      const top = 42;
      const left = config.left || 220;
      const barArea = config.barArea || 420;
      const maxValue = Math.max(...items.map(item => Math.abs(item.value)), 1);
      const height = top + items.length * rowHeight + 20;
      const rows = items.map((item, index) => {
        const y = top + index * rowHeight;
        const barWidth = (Math.abs(item.value) / maxValue) * barArea;
        return `
          <text x="12" y="${y + 16}" font-size="13">${item.label}</text>
          <rect x="${left}" y="${y}" width="${barArea}" height="18" rx="9" fill="#e4ece8"></rect>
          <rect x="${left}" y="${y}" width="${barWidth}" height="18" rx="9" fill="${item.color || '#1b8a5a'}"></rect>
          <text x="${left + barArea + 10}" y="${y + 16}" font-size="12">${item.display}</text>
        `;
      }).join('');
      return `
        <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
          <rect width="100%" height="100%" fill="#fffdf8"></rect>
          ${rows}
        </svg>
      `;
    }

    function makeDonutChart(percent, label) {
      const radius = 64;
      const circumference = 2 * Math.PI * radius;
      const offset = circumference * (1 - percent / 100);
      return `
        <svg xmlns="http://www.w3.org/2000/svg" width="240" height="220" viewBox="0 0 240 220">
          <circle cx="110" cy="100" r="${radius}" fill="none" stroke="#e4ece8" stroke-width="24"></circle>
          <circle cx="110" cy="100" r="${radius}" fill="none" stroke="#1b8a5a" stroke-width="24"
            stroke-dasharray="${circumference}" stroke-dashoffset="${offset}" transform="rotate(-90 110 100)"></circle>
          <text x="110" y="96" text-anchor="middle" font-size="30" font-weight="700">${Math.round(percent)}%</text>
          <text x="110" y="122" text-anchor="middle" font-size="13">${label}</text>
        </svg>
      `;
    }

    function makeStackedBars(series, width = 760) {
      const max = Math.max(...series.map(item => item.total), 1);
      const rows = series.map((item, index) => {
        const y = 16 + index * 42;
        let x = 220;
        const segments = item.parts.map((part, partIndex) => {
          const partWidth = (part.value / max) * 420;
          const svg = `<rect x="${x}" y="${y}" width="${partWidth}" height="18" rx="8" fill="${COLORS[partIndex % COLORS.length]}"></rect>`;
          x += partWidth;
          return svg;
        }).join('');
        return `
          <text x="12" y="${y + 14}" font-size="13">${item.label}</text>
          <rect x="220" y="${y}" width="420" height="18" rx="8" fill="#eef3ef"></rect>
          ${segments}
          <text x="655" y="${y + 14}" font-size="12">${item.total} шт.</text>
        `;
      }).join('');
      return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${series.length * 42 + 16}">${rows}</svg>`;
    }

    function makeGroupedMetricsChart(groups) {
      const items = [];
      groups.forEach(group => {
        items.push({label:`${group.name} silhouette`, value:group.silhouette, display:String(group.silhouette), color:'#1b8a5a'});
        items.push({label:`${group.name} DBI`, value:group.dbi, display:String(group.dbi), color:'#d46a4b'});
        items.push({label:`${group.name} CHI`, value:group.chi, display:String(group.chi), color:'#2969b0'});
      });
      return makeBarChart(items, {width:820, left:280, barArea:420});
    }

    function renderSubjectRelations() {
      const selector = document.getElementById('subjectSelector');
      const chosen = selector.value;
      const row = (window.dashboardData.subject_relationships || []).find(item => item.subject === chosen);
      if (!row) {
        document.getElementById('subjectRelations').innerHTML = '<p>Нет данных.</p>';
        return;
      }
      document.getElementById('subjectRelations').innerHTML =
        `<p><b>Можно изучать параллельно:</b> ${row.parallel_with.length ? row.parallel_with.join(', ') : 'нет'}</p>` +
        `<p><b>Можно изучать последовательно:</b> ${row.sequential_with.length ? row.sequential_with.join(', ') : 'нет'}</p>` +
        `<p><b>Не рекомендуется параллельно:</b> ${row.restricted_with.length ? row.restricted_with.join(', ') : 'нет'}</p>` +
        `<p>${row.reasoning}</p>`;
    }

    function renderLessonTypes() {
      const subject = document.getElementById('lessonSubjectSelector').value;
      const rows = window.dashboardData.subject_lesson_matrix || [];
      const found = rows.find(item => item.subject === subject);
      const host = document.getElementById('lessonTypesChart');
      const legend = document.getElementById('lessonTypesLegend');
      if (!found) {
        host.innerHTML = '<p>Нет данных.</p>';
        legend.innerHTML = '';
        return;
      }
      const parts = Object.entries(found.types).map(([label, value], index) => ({
        label,
        value,
        color: COLORS[index % COLORS.length]
      }));
      const total = parts.reduce((sum, item) => sum + item.value, 0);
      host.innerHTML = makeStackedBars([{label: subject, parts, total}], 760);
      legend.innerHTML = parts.map(item => `<span><i style="background:${item.color}"></i>${item.label}: ${item.value} шт.</span>`).join('');
    }

    function renderMethodComparison(data) {
      const tasks = ['parallel','sequential','difficulty'];
      return tasks.map(task => {
        const rows = (data.clustering_methods[task] || []).slice(0, 6).map(item => `
          <tr>
            <td>${item.method_name}</td>
            <td>${item.cluster_count}</td>
            <td>${item.silhouette_score}</td>
            <td>${item.davies_bouldin_index}</td>
            <td>${item.calinski_harabasz_score}</td>
            <td>${item.ranking_score}</td>
          </tr>
        `).join('');
        return `
          <h3>${task}</h3>
          <table class="method-table">
            <thead><tr><th>Метод</th><th>k</th><th>Silhouette</th><th>DBI</th><th>CHI</th><th>Rank</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        `;
      }).join('');
    }

    async function render() {
      if (!token) {
        document.getElementById('loginBox').style.display = 'block';
        document.getElementById('dashboard').style.display = 'none';
        return;
      }

      const response = await fetch('/api/dashboard', {headers:{'Authorization':'Bearer ' + token}});
      if (response.status === 401) {
        localStorage.removeItem('dashboard_token');
        token = '';
        document.getElementById('authError').textContent = 'Сессия истекла или пароль неверный.';
        document.getElementById('loginBox').style.display = 'block';
        document.getElementById('dashboard').style.display = 'none';
        return;
      }

      const data = await response.json();
      window.dashboardData = data;
      document.getElementById('loginBox').style.display = 'none';
      document.getElementById('dashboard').style.display = 'block';
      document.getElementById('updated').textContent = 'Роль: ' + data.role + ' | Обновлено: ' + new Date(data.updated_at_utc).toLocaleString();
      document.getElementById('admin').style.display = data.role === 'admin' ? 'block' : 'none';

      document.getElementById('metricMaterials').textContent = data.materials.length || data.coverage.reduce((sum, item) => sum + item.count, 0);
      document.getElementById('metricGenerated').textContent = Math.round(data.generated_share.total * 100) + '%';
      document.getElementById('metricSubjects').textContent = data.coverage.length;
      document.getElementById('metricRequirements').textContent = data.requirements.length;

      document.getElementById('coverageChart').innerHTML = makeBarChart(
        data.coverage.map((item, index) => ({
          label: item.subject,
          value: item.count,
          display: `${item.count} шт. | ${item.relative_to_average}x`,
          color: COLORS[index % COLORS.length]
        })),
        {width:860, left:260, barArea:460}
      );

      const generatedSeries = data.generated_share.by_subject.map((item, index) => ({
        label: item.subject,
        value: item.share * 100,
        display: `${Math.round(item.share * 100)}%`,
        color: COLORS[index % COLORS.length]
      }));
      document.getElementById('generatedChart').innerHTML =
        `<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap">` +
        `<div>${makeDonutChart(data.generated_share.total * 100, 'Общая доля')}</div>` +
        `<div>${makeBarChart(generatedSeries, {width:560, left:180, barArea:280})}</div>` +
        `</div>`;

      const lessonSelector = document.getElementById('lessonSubjectSelector');
      lessonSelector.innerHTML = (data.subject_lesson_matrix || []).map(item => `<option value="${item.subject}">${item.subject}</option>`).join('');
      renderLessonTypes();

      document.getElementById('requirementsChart').innerHTML = makeBarChart(
        data.requirements.map((item, index) => ({
          label: item.requirement,
          value: item.pass_rate * 100,
          display: `${Math.round(item.pass_rate * 100)}%`,
          color: COLORS[index % COLORS.length]
        })).slice(0, 12),
        {width:880, left:380, barArea:360}
      );

      document.getElementById('tops').innerHTML = data.top_requirements.map(item =>
        `<h3>${item.subject}</h3><p><b>Лучшие:</b> ${item.best.join(', ')}</p><p><b>Проблемные:</b> ${item.worst.join(', ')}</p>`
      ).join('');

      const selector = document.getElementById('subjectSelector');
      selector.innerHTML = (data.subject_relationships || []).map(item => `<option value="${item.subject}">${item.subject}</option>`).join('');
      renderSubjectRelations();

      document.getElementById('clusterMetricsChart').innerHTML = makeGroupedMetricsChart([
        {
          name:'Параллельно',
          silhouette:data.clusters.parallel_metric.silhouette_score,
          dbi:data.clusters.parallel_metric.davies_bouldin_index,
          chi:data.clusters.parallel_metric.calinski_harabasz_score
        },
        {
          name:'Последовательно',
          silhouette:data.clusters.sequential_metric.silhouette_score,
          dbi:data.clusters.sequential_metric.davies_bouldin_index,
          chi:data.clusters.sequential_metric.calinski_harabasz_score
        },
        {
          name:'Сложность',
          silhouette:data.clusters.difficulty_metric.silhouette_score,
          dbi:data.clusters.difficulty_metric.davies_bouldin_index,
          chi:data.clusters.difficulty_metric.calinski_harabasz_score
        }
      ]);
      document.getElementById('clusters').innerHTML =
        `<p><b>Параллельные:</b> ${data.clusters.parallel_metric.conclusion}</p>` +
        `<p><b>Последовательные:</b> ${data.clusters.sequential_metric.conclusion}</p>` +
        `<p><b>Сложность:</b> ${data.clusters.difficulty_metric.conclusion}</p>`;

      document.getElementById('clusterDistributionChart').innerHTML = `
        <h3>Параллельные</h3>${makeBarChart((data.cluster_distributions.parallel || []).slice(0,8).map((item, index) => ({label:item.cluster, value:item.count, display:`${item.count} шт.`, color:COLORS[index % COLORS.length]})), {width:860, left:270, barArea:430})}
        <h3>Последовательные</h3>${makeBarChart((data.cluster_distributions.sequential || []).slice(0,8).map((item, index) => ({label:item.cluster, value:item.count, display:`${item.count} шт.`, color:COLORS[index % COLORS.length]})), {width:860, left:270, barArea:430})}
        <h3>Сложность</h3>${makeBarChart((data.cluster_distributions.difficulty || []).slice(0,8).map((item, index) => ({label:item.cluster, value:item.count, display:`${item.count} шт.`, color:COLORS[index % COLORS.length]})), {width:860, left:270, barArea:430})}
      `;

      document.getElementById('methodComparison').innerHTML = renderMethodComparison(data);

      document.getElementById('materials').innerHTML =
        `<p><b>SQLite база:</b> ${data.database_path}</p>` +
        `<table><thead><tr><th>ID</th><th>Дисциплина</th><th>Тема</th><th>Сложность</th><th>Параллельно</th><th>Последовательно</th><th>Минут</th><th>Модерация</th></tr></thead>` +
        `<tbody>${data.materials.map(item => `<tr><td>${item.record_id}</td><td>${item.subject}</td><td>${item.topic}</td><td>${item.difficulty_level}</td><td>${item.parallel_cluster}</td><td>${item.sequential_cluster}</td><td>${item.estimated_minutes}</td><td>${item.moderation_conclusion}</td></tr>`).join('')}</tbody></table>`;
    }

    render();
    setInterval(render, 10000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict | str, status: int = 200) -> None:
        # Универсальная отправка HTML-страниц и JSON-ответов.
        if isinstance(payload, str):
            body = payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        # Читаем JSON из тела POST-запроса.
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        return json.loads(raw.decode("utf-8") or "{}") if raw else {}

    def _authorized_role(self) -> str | None:
        # Определяем роль пользователя по токену после входа.
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return None
        token = header.removeprefix("Bearer ").strip()
        return SESSIONS.get(token)

    def do_GET(self) -> None:  # noqa: N802
        # Главная страница отдает HTML, а защищенный маршрут дашборда — JSON.
        if self.path == "/":
            self._send(HTML)
            return
        if self.path == "/api/dashboard":
            role = self._authorized_role()
            if role is None:
                self._send({"error": "Требуется авторизация."}, status=HTTPStatus.UNAUTHORIZED)
                return
            self._send(build_view_payload(role))
            return
        self._send({"error": "Маршрут не найден."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        # В POST поддерживаем вход по роли и паролю.
        if self.path != "/api/login":
            self._send({"error": "Маршрут не найден."}, status=HTTPStatus.NOT_FOUND)
            return
        payload = self._read_json()
        role = payload.get("role", "")
        password = payload.get("password", "")
        security = ensure_security_file()
        expected = {
            "viewer": security.get("viewer_password", "viewer123"),
            "admin": security.get("admin_password", "admin123"),
        }
        if role not in expected or password != expected[role]:
            self._send({"error": "Неверная роль или пароль."}, status=HTTPStatus.UNAUTHORIZED)
            return
        token = secrets.token_hex(24)
        SESSIONS[token] = role
        self._send({"token": token, "role": role})


if __name__ == "__main__":
    # Перед запуском сервера сразу пересобираем отчет и SVG-графики.
    materials, clusters = load_materials()
    save_report(materials, clusters)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Модуль Б запущен: http://{HOST}:{PORT}")
    server.serve_forever()
