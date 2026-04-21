from __future__ import annotations

"""
Файл нужен для критерия интеграции с общедоступным мессенджером.
Он не скачивает зависимости и использует только стандартную библиотеку Python.

Как запустить:
1. Запустить модуль Г: `python module_g\\agent.py`
2. Убедиться, что заполнен токен в `Data/telegram_config.json`
   или задана переменная окружения `TELEGRAM_BOT_TOKEN`.
3. Запустить bridge: `python module_g\\telegram_bridge.py`

Поддерживаемые команды в Telegram:
- /start
- /help
- /health
- /models
- /material ROOT-001
- /moderation ROOT-001
- /parallel ROOT-001
- /sequential ROOT-001
- /difficulty ROOT-001
- /time Информатика, Общий набор
- /trajectory Информатика, Общий набор | beginner | 2 | 14
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, parse, request

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import DATA_DIR


CONFIG_PATH = DATA_DIR / "telegram_config.json"
LOCAL_API = "http://127.0.0.1:5082"


def load_token() -> str:
    # Сначала читаем токен из переменной окружения,
    # затем используем локальный конфигурационный файл проекта.
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if env_token:
        return env_token
    if CONFIG_PATH.exists():
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return str(payload.get("bot_token", "")).strip()
    return ""


TOKEN = load_token()
API_URL = f"https://api.telegram.org/bot{TOKEN}" if TOKEN else ""


def send_message(chat_id: int, text: str) -> None:
    # Отправляем ответ пользователю через стандартный Telegram Bot API.
    if not API_URL:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN и не найден Data/telegram_config.json.")
    payload = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    request.urlopen(f"{API_URL}/sendMessage", data=payload, timeout=30).read()


def get_updates(offset: int | None = None) -> dict:
    # Получаем новые сообщения боту через long polling Telegram Bot API.
    if not API_URL:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN и не найден Data/telegram_config.json.")
    url = f"{API_URL}/getUpdates?timeout=25"
    if offset is not None:
        url += f"&offset={offset}"
    return json.loads(request.urlopen(url, timeout=35).read().decode("utf-8"))


def api_get_json(path: str) -> dict:
    # Обращаемся к локальному API модуля Г и возвращаем JSON-ответ.
    with request.urlopen(f"{LOCAL_API}{path}", timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post_json(path: str, payload: dict) -> dict:
    # Отправляем POST-запрос к локальному API модуля Г для оценки времени и траектории.
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{LOCAL_API}{path}",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_subjects(raw: str) -> list[str]:
    # Разбираем список дисциплин, переданный через запятую.
    return [item.strip() for item in raw.split(",") if item.strip()]


def format_json_brief(payload: dict) -> str:
    # Сжимаем JSON-ответ в компактный текст для Telegram,
    # чтобы сообщение было понятным и не слишком длинным.
    if "errors" in payload:
        return "Ошибки:\n- " + "\n- ".join(payload["errors"])
    if "error" in payload:
        return f"Ошибка: {payload['error']}"
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    return pretty[:3500]


def handle_command(text: str) -> str:
    # Разбираем команды Telegram и переводим их в запросы к локальному API.
    # Все ответы строятся на основе уже обученных моделей и данных из SQLite.
    command = text.strip()
    if not command:
        return "Пустое сообщение. Напишите /help."

    if command in {"/start", "/help"}:
        return (
            "Команды бота:\n"
            "/health\n"
            "/models\n"
            "/material ROOT-001\n"
            "/moderation ROOT-001\n"
            "/parallel ROOT-001\n"
            "/sequential ROOT-001\n"
            "/difficulty ROOT-001\n"
            "/time Информатика, Общий набор\n"
            "/trajectory Информатика, Общий набор | beginner | 2 | 14"
        )

    if command == "/health":
        return format_json_brief(api_get_json("/api/health"))

    if command == "/models":
        return format_json_brief(api_get_json("/api/models"))

    if command.startswith("/material "):
        record_id = command.removeprefix("/material ").strip()
        return format_json_brief(api_get_json(f"/api/material?record_id={parse.quote(record_id)}"))

    if command.startswith("/moderation "):
        record_id = command.removeprefix("/moderation ").strip()
        return format_json_brief(api_get_json(f"/api/moderation?record_id={parse.quote(record_id)}"))

    if command.startswith("/parallel "):
        record_id = command.removeprefix("/parallel ").strip()
        return format_json_brief(api_get_json(f"/api/parallel?record_id={parse.quote(record_id)}"))

    if command.startswith("/sequential "):
        record_id = command.removeprefix("/sequential ").strip()
        return format_json_brief(api_get_json(f"/api/sequential?record_id={parse.quote(record_id)}"))

    if command.startswith("/difficulty "):
        record_id = command.removeprefix("/difficulty ").strip()
        return format_json_brief(api_get_json(f"/api/difficulty?record_id={parse.quote(record_id)}"))

    if command.startswith("/time "):
        subjects = parse_subjects(command.removeprefix("/time ").strip())
        payload = {"subjects": subjects}
        return format_json_brief(api_post_json("/api/time-estimate", payload))

    if command.startswith("/trajectory "):
        raw = command.removeprefix("/trajectory ").strip()
        parts = [item.strip() for item in raw.split("|")]
        subjects = parse_subjects(parts[0]) if parts else []
        experience_level = parts[1] if len(parts) > 1 and parts[1] else "beginner"
        hours_per_day = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 2
        deadline_days = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 14
        payload = {
            "subjects": subjects,
            "experience_level": experience_level,
            "completed_subjects": [],
            "hours_per_day": hours_per_day,
            "deadline_days": deadline_days,
        }
        return format_json_brief(api_post_json("/api/trajectory", payload))

    return "Неизвестная команда. Напишите /help."


def extract_message(update: dict) -> tuple[int | None, str]:
    # Достаем chat_id и текст сообщения из входящего update.
    message = update.get("message", {})
    chat = message.get("chat", {})
    text = message.get("text", "")
    return chat.get("id"), text


def run_polling() -> None:
    # Основной цикл long polling:
    # читаем сообщения, отправляем их в локальный API и отвечаем обратно в Telegram.
    if not TOKEN:
        raise RuntimeError("Не найден токен Telegram. Заполни Data/telegram_config.json или TELEGRAM_BOT_TOKEN.")
    print("Telegram bridge запущен. Ожидание сообщений...")
    offset: int | None = None
    while True:
        try:
            payload = get_updates(offset)
            for update in payload.get("result", []):
                offset = update["update_id"] + 1
                chat_id, text = extract_message(update)
                if not chat_id:
                    continue
                try:
                    answer = handle_command(text)
                except error.URLError:
                    answer = "Локальный API модуля Г недоступен. Сначала запусти python module_g\\agent.py"
                except Exception as exc:  # noqa: BLE001
                    answer = f"Ошибка обработки команды: {exc}"
                send_message(chat_id, answer)
        except KeyboardInterrupt:
            print("Остановка Telegram bridge.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка polling: {exc}")
            time.sleep(3)


def config_example_path() -> Path:
    # Возвращаем путь к локальному конфигу, чтобы его можно было явно показать в отчете.
    return CONFIG_PATH


if __name__ == "__main__":
    run_polling()
