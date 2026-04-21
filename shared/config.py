from __future__ import annotations

import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).resolve().parent.parent
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
DOCS_DIR = ROOT_DIR / "Docs"
DIST_DIR = ROOT_DIR / "Distribution"
MANIFEST_PATH = DATA_DIR / "input_manifest.json"
DATABASE_PATH = DATA_DIR / "ml_db.sqlite3"
SECURITY_PATH = DATA_DIR / "security.json"
EXPORTS_DIR = DATA_DIR / "Exports"
MODEL_DIR = DATA_DIR / "Models"
REPORTS_DIR = DOCS_DIR / "Reports"
INCOMING_DIR = DATA_DIR / "Incoming"
MODULE_A_INPUT_DIR = INCOMING_DIR / "module_a"


def ensure_workspace() -> None:
    """Создаем все рабочие каталоги до запуска агентов."""
    for directory in (DATA_DIR, DOCS_DIR, DIST_DIR, EXPORTS_DIR, MODEL_DIR, REPORTS_DIR, INCOMING_DIR, MODULE_A_INPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
