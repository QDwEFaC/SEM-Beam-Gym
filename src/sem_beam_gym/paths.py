"""Каталоги артефактов: runs/ (модели), results/ (таблицы), figures/ (рисунки).

Все пути считаются от корня проекта, а не от текущей папки, чтобы код одинаково работал из
командной строки и из notebooks/. Корень - папка с pyproject.toml над src/ (пакет установлен
в режиме editable, как делает `uv sync`). Его можно переопределить переменной SEM_BEAM_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    """Корень проекта для артефактов."""
    # Порядок: явная переменная SEM_BEAM_ROOT (например, для smoke теста в отдельной папке,
    # чтобы не затереть результаты), затем корень репозитория над src/. Если пакет установлен
    # не в режиме editable и pyproject.toml не найден, используется текущая папка.
    if env := os.environ.get("SEM_BEAM_ROOT"):
        return Path(env).resolve()
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() else Path.cwd()


ROOT = _root()
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
