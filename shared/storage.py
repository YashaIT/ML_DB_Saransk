from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from shared.config import DATABASE_PATH, SECURITY_PATH, ensure_workspace


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _connect() -> sqlite3.Connection:
    ensure_workspace()
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    _init_schema(connection)
    return connection


def _init_schema(connection: sqlite3.Connection) -> None:
    existing_sources = {row["name"] for row in connection.execute("PRAGMA table_info(sources)")} if _table_exists(connection, "sources") else set()
    if existing_sources and ("id" not in existing_sources or "content_paths_json" not in existing_sources):
        connection.executescript(
            """
            DROP TABLE IF EXISTS sources;
            DROP TABLE IF EXISTS materials;
            DROP TABLE IF EXISTS suggestions;
            DROP TABLE IF EXISTS runs;
            DROP TABLE IF EXISTS metadata;
            """
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            lesson_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_paths_json TEXT NOT NULL,
            media_paths_json TEXT NOT NULL,
            requirements_json TEXT NOT NULL,
            topic_order INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS materials (
            record_id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            lesson_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            text_material TEXT NOT NULL,
            summary TEXT NOT NULL,
            moderation_conclusion TEXT NOT NULL,
            requirement_checks_json TEXT NOT NULL,
            media_descriptions_json TEXT NOT NULL,
            generated INTEGER NOT NULL,
            topic_order INTEGER NOT NULL,
            previous_record_id TEXT,
            next_record_id TEXT,
            word_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            methodical_score INTEGER NOT NULL,
            difficulty_level TEXT NOT NULL,
            parallel_cluster TEXT NOT NULL,
            sequential_cluster TEXT NOT NULL,
            estimated_minutes INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suggestions (
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            reason TEXT NOT NULL,
            generated INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT NOT NULL,
            started_at_utc TEXT NOT NULL,
            result TEXT NOT NULL,
            details TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        """
    )
    connection.commit()


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_database() -> dict[str, Any]:
    with _connect() as connection:
        sources = [
            {
                "id": row["id"],
                "subject": row["subject"],
                "topic": row["topic"],
                "lesson_type": row["lesson_type"],
                "file_path": row["file_path"],
                "content_paths": json.loads(row["content_paths_json"]),
                "media_paths": json.loads(row["media_paths_json"]),
                "requirements": json.loads(row["requirements_json"]),
                "topic_order": row["topic_order"],
            }
            for row in connection.execute("SELECT * FROM sources ORDER BY subject, topic_order, id")
        ]
        materials = [
            {
                "record_id": row["record_id"],
                "subject": row["subject"],
                "topic": row["topic"],
                "lesson_type": row["lesson_type"],
                "source_path": row["source_path"],
                "source_kind": row["source_kind"],
                "text_material": row["text_material"],
                "summary": row["summary"],
                "moderation_conclusion": row["moderation_conclusion"],
                "requirement_checks": json.loads(row["requirement_checks_json"]),
                "media_descriptions": json.loads(row["media_descriptions_json"]),
                "generated": bool(row["generated"]),
                "topic_order": row["topic_order"],
                "previous_record_id": row["previous_record_id"],
                "next_record_id": row["next_record_id"],
                "word_count": row["word_count"],
                "char_count": row["char_count"],
                "methodical_score": row["methodical_score"],
                "difficulty_level": row["difficulty_level"],
                "parallel_cluster": row["parallel_cluster"],
                "sequential_cluster": row["sequential_cluster"],
                "estimated_minutes": row["estimated_minutes"],
                "content_hash": row["content_hash"],
                "updated_at_utc": row["updated_at_utc"],
            }
            for row in connection.execute("SELECT * FROM materials ORDER BY subject, topic_order, record_id")
        ]
        suggestions = [
            {
                "subject": row["subject"],
                "topic": row["topic"],
                "reason": row["reason"],
                "generated": bool(row["generated"]),
            }
            for row in connection.execute("SELECT * FROM suggestions ORDER BY subject, topic")
        ]
        runs = [
            {
                "module": row["module"],
                "started_at_utc": row["started_at_utc"],
                "result": row["result"],
                "details": row["details"],
            }
            for row in connection.execute("SELECT module, started_at_utc, result, details FROM runs ORDER BY id")
        ]
        metadata = {
            row["key"]: json.loads(row["value_json"])
            for row in connection.execute("SELECT key, value_json FROM metadata")
        }
    return {
        "sources": sources,
        "materials": materials,
        "clusters": metadata.get("clusters", {}),
        "models": metadata.get("models", {}),
        "module_a_stats": metadata.get("module_a_stats", {}),
        "module_a_parse_warnings": metadata.get("module_a_parse_warnings", []),
        "module_a_format_stats": metadata.get("module_a_format_stats", {}),
        "analytical_briefs": metadata.get("analytical_briefs", []),
        "methodical_guidelines": metadata.get("methodical_guidelines", []),
        "suggestions": suggestions,
        "runs": runs,
    }


def save_database(payload: dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM sources")
        connection.executemany(
            """
            INSERT INTO sources (id, subject, topic, lesson_type, file_path, content_paths_json, media_paths_json, requirements_json, topic_order)
            VALUES (:id, :subject, :topic, :lesson_type, :file_path, :content_paths_json, :media_paths_json, :requirements_json, :topic_order)
            """,
            [
                {
                    "id": item["id"],
                    "subject": item["subject"],
                    "topic": item["topic"],
                    "lesson_type": item["lesson_type"],
                    "file_path": item["file_path"],
                    "content_paths_json": json.dumps(item.get("content_paths", [item["file_path"]]), ensure_ascii=False),
                    "media_paths_json": json.dumps(item.get("media_paths", []), ensure_ascii=False),
                    "requirements_json": json.dumps(item.get("requirements", []), ensure_ascii=False),
                    "topic_order": item["topic_order"],
                }
                for item in payload.get("sources", [])
            ],
        )

        connection.execute("DELETE FROM materials")
        connection.executemany(
            """
            INSERT INTO materials (
                record_id, subject, topic, lesson_type, source_path, source_kind, text_material, summary,
                moderation_conclusion, requirement_checks_json, media_descriptions_json, generated, topic_order,
                previous_record_id, next_record_id, word_count, char_count, methodical_score, difficulty_level,
                parallel_cluster, sequential_cluster, estimated_minutes, content_hash, updated_at_utc
            )
            VALUES (
                :record_id, :subject, :topic, :lesson_type, :source_path, :source_kind, :text_material, :summary,
                :moderation_conclusion, :requirement_checks_json, :media_descriptions_json, :generated, :topic_order,
                :previous_record_id, :next_record_id, :word_count, :char_count, :methodical_score, :difficulty_level,
                :parallel_cluster, :sequential_cluster, :estimated_minutes, :content_hash, :updated_at_utc
            )
            """,
            [
                {
                    **item,
                    "requirement_checks_json": json.dumps(item.get("requirement_checks", []), ensure_ascii=False),
                    "media_descriptions_json": json.dumps(item.get("media_descriptions", []), ensure_ascii=False),
                    "generated": 1 if item.get("generated") else 0,
                }
                for item in payload.get("materials", [])
            ],
        )

        connection.execute("DELETE FROM suggestions")
        connection.executemany(
            "INSERT INTO suggestions (subject, topic, reason, generated) VALUES (:subject, :topic, :reason, :generated)",
            [
                {
                    "subject": item["subject"],
                    "topic": item["topic"],
                    "reason": item["reason"],
                    "generated": 1 if item.get("generated") else 0,
                }
                for item in payload.get("suggestions", [])
            ],
        )

        connection.execute("DELETE FROM runs")
        connection.executemany(
            "INSERT INTO runs (module, started_at_utc, result, details) VALUES (:module, :started_at_utc, :result, :details)",
            payload.get("runs", []),
        )

        connection.execute("DELETE FROM metadata")
        connection.executemany(
            "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
            [
                ("clusters", json.dumps(payload.get("clusters", {}), ensure_ascii=False)),
                ("models", json.dumps(payload.get("models", {}), ensure_ascii=False)),
                ("module_a_stats", json.dumps(payload.get("module_a_stats", {}), ensure_ascii=False)),
                ("module_a_parse_warnings", json.dumps(payload.get("module_a_parse_warnings", []), ensure_ascii=False)),
                ("module_a_format_stats", json.dumps(payload.get("module_a_format_stats", {}), ensure_ascii=False)),
                ("analytical_briefs", json.dumps(payload.get("analytical_briefs", []), ensure_ascii=False)),
                ("methodical_guidelines", json.dumps(payload.get("methodical_guidelines", []), ensure_ascii=False)),
            ],
        )
        connection.commit()


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def ensure_security_file() -> dict[str, str]:
    ensure_workspace()
    default_payload = {
        "viewer_password": "viewer123",
        "admin_password": "admin123",
    }
    if not SECURITY_PATH.exists():
        save_json(SECURITY_PATH, default_payload)
        return default_payload
    return load_json(SECURITY_PATH, default_payload)
