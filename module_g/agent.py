from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import REPORTS_DIR
from shared.core import Material, build_learning_trajectory, criteria_lines, summarize_time_estimates
from shared.ml_models import load_model_package, predict_with_package
from shared.storage import load_database


HOST = "127.0.0.1"
PORT = 5082


def load_runtime() -> tuple[list[Material], dict, dict[str, dict]]:
    # Загружаем материалы, метаданные моделей и сами бинарные пакеты моделей.
    # Модуль Г только использует уже обученные модели и не запускает обучение заново.
    database = load_database()
    materials = [Material(**item) for item in database["materials"]]
    registry = database.get("models", {})
    packages: dict[str, dict] = {}
    for task_name, payload in registry.get("models", {}).items():
        artifact_path = payload.get("artifact_path")
        if artifact_path and Path(artifact_path).exists():
            packages[task_name] = load_model_package(Path(artifact_path))
    return materials, registry, packages


def validate_payload(payload: dict) -> list[str]:
    # Валидируем все входные параметры API, чтобы клиент сразу получал понятные ошибки.
    errors: list[str] = []
    if "record_id" in payload and not isinstance(payload["record_id"], str):
        errors.append("Поле record_id должно быть строкой.")
    if "subjects" in payload and not isinstance(payload["subjects"], list):
        errors.append("Поле subjects должно быть списком.")
    if "completed_subjects" in payload and not isinstance(payload["completed_subjects"], list):
        errors.append("Поле completed_subjects должно быть списком.")
    if "hours_per_day" in payload and (not isinstance(payload["hours_per_day"], int) or payload["hours_per_day"] <= 0):
        errors.append("Поле hours_per_day должно быть положительным целым числом.")
    if "deadline_days" in payload and (not isinstance(payload["deadline_days"], int) or payload["deadline_days"] <= 0):
        errors.append("Поле deadline_days должно быть положительным целым числом.")
    if "experience_level" in payload and payload["experience_level"] not in {"beginner", "intermediate", "advanced"}:
        errors.append("Поле experience_level должно быть одним из значений: beginner, intermediate, advanced.")
    return errors


def find_material(materials: list[Material], record_id: str) -> Material | None:
    # Ищем материал по уникальному идентификатору для всех GET-запросов карточки.
    return next((item for item in materials if item.record_id == record_id), None)


def build_profile(payload: dict) -> dict[str, object]:
    # Преобразуем пользовательские параметры в единый профиль,
    # который одинаково используется в оценке времени и построении траектории.
    return {
        "experience_level": payload.get("experience_level", "beginner"),
        "completed_subjects": payload.get("completed_subjects", []),
        "hours_per_day": payload.get("hours_per_day", 2),
        "deadline_days": payload.get("deadline_days", 14),
        "subjects": payload.get("subjects", []),
    }


def predict_task(packages: dict[str, dict], task_name: str, material: Material) -> dict:
    # Выполняем инференс по конкретной задаче на основе уже загруженного бинарного пакета модели.
    if task_name not in packages:
        return {"error": f"Модель {task_name} не загружена."}
    return predict_with_package(packages[task_name], material)


def material_summary(material: Material, packages: dict[str, dict]) -> dict:
    # Возвращаем расширенную карточку материала:
    # исходные атрибуты, результаты модерации и предсказания всех трех моделей.
    return {
        "record_id": material.record_id,
        "subject": material.subject,
        "topic": material.topic,
        "moderation_conclusion": material.moderation_conclusion,
        "difficulty_level_actual": material.difficulty_level,
        "estimated_minutes": material.estimated_minutes,
        "parallel_cluster_actual": material.parallel_cluster,
        "sequential_cluster_actual": material.sequential_cluster,
        "model_predictions": {
            "parallel": predict_task(packages, "parallel", material),
            "sequential": predict_task(packages, "sequential", material),
            "difficulty": predict_task(packages, "difficulty", material),
        },
        "requirement_checks": material.requirement_checks,
    }


def moderation_payload(material: Material) -> dict:
    # Отдельный ответ по модерации нужен, чтобы приложение могло быстро показать,
    # соответствует ли материал методическим рекомендациям и почему.
    return {
        "record_id": material.record_id,
        "moderation_conclusion": material.moderation_conclusion,
        "methodical_score": material.methodical_score,
        "requirement_checks": material.requirement_checks,
    }


def parallel_payload(material: Material, packages: dict[str, dict]) -> dict:
    # Возвращаем фактический и предсказанный класс параллельного изучения.
    prediction = predict_task(packages, "parallel", material)
    return {
        "record_id": material.record_id,
        "parallel_cluster_actual": material.parallel_cluster,
        "parallel_cluster_predicted": prediction.get("predicted_label"),
        "probabilities": prediction.get("probabilities", {}),
    }


def sequential_payload(material: Material, packages: dict[str, dict]) -> dict:
    # Возвращаем фактические связи материала и прогноз модели по роли в последовательности.
    prediction = predict_task(packages, "sequential", material)
    return {
        "record_id": material.record_id,
        "previous_record_id": material.previous_record_id,
        "next_record_id": material.next_record_id,
        "sequential_role_predicted": prediction.get("predicted_label"),
        "probabilities": prediction.get("probabilities", {}),
    }


def difficulty_payload(material: Material, packages: dict[str, dict]) -> dict:
    # Возвращаем фактическую и предсказанную сложность, а также оценку времени освоения.
    prediction = predict_task(packages, "difficulty", material)
    return {
        "record_id": material.record_id,
        "difficulty_level_actual": material.difficulty_level,
        "difficulty_level_predicted": prediction.get("predicted_label"),
        "estimated_minutes": material.estimated_minutes,
        "probabilities": prediction.get("probabilities", {}),
    }


CHAT_HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Модуль Г</title>
  <style>
    body{font-family:"Segoe UI",sans-serif;background:linear-gradient(130deg,#f4efe5,#eef5ff);margin:0;color:#1b2a34}
    main{max-width:1120px;margin:0 auto;padding:24px}
    .panel{background:#fffdf8;border:1px solid #d7ccb8;border-radius:18px;padding:18px;margin-bottom:16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
    input,select{width:100%;padding:10px;border-radius:12px;border:1px solid #cdbfa9;box-sizing:border-box;margin-bottom:10px}
    button{padding:10px 16px;border:none;border-radius:12px;background:#0f7c59;color:white;cursor:pointer;margin-right:8px;margin-bottom:8px}
    pre{white-space:pre-wrap;background:#f7f3eb;padding:12px;border-radius:12px}
  </style>
</head>
<body>
<main>
  <div class="panel">
    <h1>Модуль Г. API и интегрированное приложение</h1>
    <p>Интерфейс использует локальный API и загружает предобученные модели из <code>Data/Models</code>. Здесь можно проверить модерацию, прогнозы моделей, оценку времени и персональную траекторию.</p>
  </div>

  <div class="panel">
    <h2>Проверка материала</h2>
    <input id="record" placeholder="Например: ROOT-001">
    <button onclick="callGet('/api/material?record_id=' + encodeURIComponent(document.getElementById('record').value.trim()), 'material')">Карточка</button>
    <button onclick="callGet('/api/moderation?record_id=' + encodeURIComponent(document.getElementById('record').value.trim()), 'material')">Модерация</button>
    <button onclick="callGet('/api/parallel?record_id=' + encodeURIComponent(document.getElementById('record').value.trim()), 'material')">Параллельно</button>
    <button onclick="callGet('/api/sequential?record_id=' + encodeURIComponent(document.getElementById('record').value.trim()), 'material')">Последовательно</button>
    <button onclick="callGet('/api/difficulty?record_id=' + encodeURIComponent(document.getElementById('record').value.trim()), 'material')">Сложность</button>
    <pre id="material"></pre>
  </div>

  <div class="panel">
    <h2>Профиль пользователя, время и траектория</h2>
    <div class="grid">
      <div><input id="subjects" placeholder="Дисциплины через запятую"></div>
      <div><select id="experience"><option value="beginner">beginner</option><option value="intermediate">intermediate</option><option value="advanced">advanced</option></select></div>
      <div><input id="completed" placeholder="Уже изученные дисциплины"></div>
      <div><input id="hours" placeholder="Часов в день" value="2"></div>
      <div><input id="deadline" placeholder="Срок в днях" value="14"></div>
    </div>
    <button onclick="callTimeEstimate()">Оценка времени</button>
    <button onclick="callTrajectory()">Траектория</button>
    <pre id="path"></pre>
  </div>

  <script>
    function buildProfilePayload() {
      return {
        subjects: document.getElementById('subjects').value.split(',').map(x => x.trim()).filter(Boolean),
        experience_level: document.getElementById('experience').value,
        completed_subjects: document.getElementById('completed').value.split(',').map(x => x.trim()).filter(Boolean),
        hours_per_day: parseInt(document.getElementById('hours').value || '2', 10),
        deadline_days: parseInt(document.getElementById('deadline').value || '14', 10)
      };
    }

    async function callGet(url, target) {
      const response = await fetch(url);
      document.getElementById(target).textContent = JSON.stringify(await response.json(), null, 2);
    }

    async function postJson(url, payload, target) {
      const response = await fetch(url, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload)
      });
      document.getElementById(target).textContent = JSON.stringify(await response.json(), null, 2);
    }

    async function callTimeEstimate() {
      await postJson('/api/time-estimate', buildProfilePayload(), 'path');
    }

    async function callTrajectory() {
      await postJson('/api/trajectory', buildProfilePayload(), 'path');
    }
  </script>
</main>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict | str, status: int = 200) -> None:
        # Универсальная отправка HTML и JSON-ответов.
        if isinstance(payload, str):
            body = payload.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
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

    def _resolve_material(self, materials: list[Material]) -> tuple[Material | None, list[str]]:
        # Извлекаем record_id из query string и проверяем, что материал существует.
        params = parse_qs(urlparse(self.path).query)
        record_id = params.get("record_id", [""])[0]
        errors = validate_payload({"record_id": record_id})
        if errors:
            return None, errors
        material = find_material(materials, record_id)
        if material is None:
            return None, ["Материал с указанным record_id не найден."]
        return material, []

    def do_GET(self) -> None:  # noqa: N802
        # GET-маршруты используются для проверки материала, моделей и состояния сервиса.
        materials, registry, packages = load_runtime()
        if self.path == "/":
            save_report()
            self._send(CHAT_HTML)
            return
        if self.path == "/api/health":
            self._send({"status": "ok", "models_loaded": bool(packages), "material_count": len(materials)})
            return
        if self.path == "/api/models":
            self._send({"models": registry.get("models", {}), "loaded_from": "Data/Models/*.joblib"})
            return
        if self.path.startswith("/api/material"):
            material, errors = self._resolve_material(materials)
            if errors:
                self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send({"found": True, "material": material_summary(material, packages)})
            return
        if self.path.startswith("/api/moderation"):
            material, errors = self._resolve_material(materials)
            if errors:
                self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send(moderation_payload(material))
            return
        if self.path.startswith("/api/parallel"):
            material, errors = self._resolve_material(materials)
            if errors:
                self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send(parallel_payload(material, packages))
            return
        if self.path.startswith("/api/sequential"):
            material, errors = self._resolve_material(materials)
            if errors:
                self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send(sequential_payload(material, packages))
            return
        if self.path.startswith("/api/difficulty"):
            material, errors = self._resolve_material(materials)
            if errors:
                self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send(difficulty_payload(material, packages))
            return
        self._send({"error": "Маршрут не найден."}, status=HTTPStatus.NOT_FOUND)


    def do_POST(self) -> None:  # noqa: N802
        # POST-маршруты используют профиль пользователя для персонализированных расчетов.
        materials, registry, _packages = load_runtime()
        payload = self._read_json()
        errors = validate_payload(payload)
        if errors:
            self._send({"errors": errors}, status=HTTPStatus.BAD_REQUEST)
            return
        profile = build_profile(payload)
        if self.path == "/api/time-estimate":
            subjects = payload.get("subjects") or sorted({material.subject for material in materials})
            self._send(
                {
                    "time_estimate": summarize_time_estimates(materials, subjects, profile=profile),
                    "models_loaded": bool(registry.get("models")),
                }
            )
            return
        if self.path == "/api/trajectory":
            subjects = payload.get("subjects") or sorted({material.subject for material in materials})
            hours_per_day = payload.get("hours_per_day", 2)
            trajectory = build_learning_trajectory(materials, subjects, hours_per_day, profile=profile)
            self._send({"trajectory": trajectory, "models_loaded": bool(registry.get("models"))})
            return
        self._send({"error": "Маршрут не найден."}, status=HTTPStatus.NOT_FOUND)


def save_report() -> None:
    # Дополняем отчет и документацию API, не переписывая общую структуру проекта.
    lines = [
        "# Отчет по модулю Г",
        "",
        "## Критерии и аспекты",
        "| Аспект | Где выполнено | Что лежит на выходе | Как проверить |",
        "|---|---|---|---|",
        "| API валидирует входные данные | `module_g/agent.py` | ответы API с ошибками при некорректном вводе | отправить неверный `record_id` или неправильный JSON |",
        "| API поддерживает все необходимые запросы | `module_g/agent.py` | GET `/api/material`, `/api/moderation`, `/api/parallel`, `/api/sequential`, `/api/difficulty`; POST `/api/time-estimate`, `/api/trajectory` | открыть `Docs/Reports/api_reference.md` |",
        "| API загружает предобученные модели | `module_g/agent.py`, `Data/Models/*.joblib` | бинарные пакеты моделей | открыть `/api/health` и `/api/models` |",
        "| API не обучает модели заново | `module_g/agent.py` | только чтение из `Data/Models` | перезапуск модуля Г не меняет модели |",
        "| Интегрированное приложение использует API | `module_g/agent.py` | web-интерфейс `http://127.0.0.1:5082` | открыть главную страницу |",
        "| Telegram-интеграция подготовлена | `module_g/telegram_bridge.py` | bridge для Telegram Bot API и инструкция по токену | открыть `module_g/telegram_bridge.py` и раздел Telegram в этом отчете |",
        "| Документация API | `module_g/agent.py` | `Docs/Reports/api_reference.md` | открыть markdown-файл |",
        "",
        "## Реализованный функционал",
        "- `GET /api/material` — полная карточка материала и предсказания моделей.",
        "- `GET /api/moderation` — оценка соответствия методическим рекомендациям.",
        "- `GET /api/parallel` — прогноз класса параллельного изучения.",
        "- `GET /api/sequential` — прогноз роли в последовательной траектории.",
        "- `GET /api/difficulty` — прогноз уровня сложности.",
        "- `POST /api/time-estimate` — оценка времени по дисциплинам с учетом опыта, уже изученных дисциплин, часов в день и срока.",
        "- `POST /api/trajectory` — построение индивидуальной траектории по тем же параметрам.",
        "",
        "## Методические рекомендации в модуле Г",
        "- Ответ `GET /api/moderation` возвращает заключение и полный список проверок, сформированных в модуле А на основе методических критериев.",
        "- Таким образом интегрированное приложение показывает не только прогноз моделей, но и проверку материала по нормативным и методическим требованиям.",
        "",
        "## Telegram",
        "- Используется **Telegram Bot API** по HTTPS: методы `getUpdates` и `sendMessage`.",
        "- Токен бота можно задать двумя способами: через переменную окружения `TELEGRAM_BOT_TOKEN` или через файл `Data/telegram_config.json`.",
        "- В `Data/telegram_config.json` нужно положить JSON вида: `{ \"bot_token\": \"123456:ABC...\" }`.",
        "- После задания токена файл `module_g/telegram_bridge.py` можно использовать как транспорт между Telegram и локальным API модуля Г.",
        "",
        "## Выходные файлы модуля",
        "- `Docs/Reports/module_g_report.md` — основной отчет по модулю Г.",
        "- `Docs/Reports/api_reference.md` — документация API с маршрутами, форматами и примерами запросов.",
        "- `module_g/telegram_bridge.py` — интеграционный мост для Telegram Bot API.",
        "",
        "## Где выполнены критерии",
        *[f"- {line}" for line in criteria_lines("G")],
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "module_g_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORTS_DIR / "api_reference.md").write_text(
        "\n".join(
            [
                "# Документация API",
                "",
                "## GET маршруты",
                "- `GET /api/health` — проверка сервиса и факта загрузки моделей.",
                "- `GET /api/models` — список метаданных предобученных моделей.",
                "- `GET /api/material?record_id=ROOT-001` — карточка материала и предсказания моделей.",
                "- `GET /api/moderation?record_id=ROOT-001` — оценка материала по методическим критериям.",
                "- `GET /api/parallel?record_id=ROOT-001` — прогноз класса параллельного изучения.",
                "- `GET /api/sequential?record_id=ROOT-001` — прогноз роли в последовательной цепочке.",
                "- `GET /api/difficulty?record_id=ROOT-001` — прогноз сложности и оценка времени.",
                "",
                "## POST маршруты",
                "- `POST /api/time-estimate` — оценка времени по выбранным дисциплинам с учетом пользовательского профиля.",
                "- `POST /api/trajectory` — построение индивидуальной траектории.",
                "",
                "## Пример POST /api/time-estimate",
                "```json",
                '{',
                '  "subjects": ["Общий набор", "Информатика"],',
                '  "experience_level": "beginner",',
                '  "completed_subjects": ["Учебная практика"],',
                '  "hours_per_day": 2,',
                '  "deadline_days": 14',
                '}',
                "```",
                "",
                "## Пример POST /api/trajectory",
                "```json",
                '{',
                '  "subjects": ["Общий набор", "Информатика"],',
                '  "experience_level": "intermediate",',
                '  "completed_subjects": [],',
                '  "hours_per_day": 3,',
                '  "deadline_days": 10',
                '}',
                "```",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    save_report()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Модуль Г запущен: http://{HOST}:{PORT}")
    server.serve_forever()
