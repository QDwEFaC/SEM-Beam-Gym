"""Полный конвейер: обучение матрицы -> оценка -> память линейного фильтра -> графики.

Команда `sem-beam` (= `python -m sem_beam_gym.main`), из корня проекта - `uv run sem-beam`:

    sem-beam                 # 6 сценариев обучения x 2 условия x 3 seed = 36 запусков PPO по 1M шагов
    sem-beam --skip-train    # только оценка и графики по готовым runs/
    sem-beam --quick         # smoke тест конвейера: минимальный бюджет, мало эпизодов

Запуски обучения независимы и идут параллельно (--jobs, по умолчанию - число ядер),
каждый в одном потоке torch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from sem_beam_gym.scenarios import SCENARIOS, TRAIN_SCENARIOS

CONDS = ("nosens", "sens")
SEEDS = (0, 1, 2)  # выполняем требование - минимум с тремя seed, среднее и разброс


def run(cmd: list[str]) -> int:
    """Шаг конвейера - модуль пакета в отдельном процессе (python -m sem_beam_gym.<cmd[0]>)."""
    # Отдельный процесс на шаг: каждый запуск обучения получает свой интерпретатор и свою память,
    # падение одного шага не роняет конвейер молча, а код возврата проверяется вызывающим.
    return subprocess.run([sys.executable, "-m", f"sem_beam_gym.{cmd[0]}", *cmd[1:]]).returncode


def main():
    """Разобрать аргументы и выполнить конвейер по шагам."""
    # 36 запусков обучения параллельно, затем оценка, проверка памяти фильтра, графики,
    # галерея и визуализация по каждому сценарию. Весь результат воспроизводится одной командой.
    # Потоки (ThreadPoolExecutor) ждут дочерние процессы, сама работа идёт в процессах,
    # поэтому GIL не влияет. --quick проверяет весь путь за несколько минут.
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000, help="шагов обучения на запуск")
    p.add_argument("--episodes", type=int, default=30, help="тестовых эпизодов на точку")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke тест: 4096 шагов, 2 эпизода на точку")
    a = p.parse_args()
    if a.quick:
        a.steps, a.episodes = 4096, 2

    if not a.skip_train:
        cmds = [["train", "--scenario", sc, "--cond", c, "--seed", str(s), "--steps", str(a.steps)]
                for sc in TRAIN_SCENARIOS for c in CONDS for s in SEEDS]
        with ThreadPoolExecutor(a.jobs) as ex:
            if any(ex.map(run, cmds)):
                sys.exit("обучение завершилось с ошибкой")
    steps = [["evaluate", "--episodes", str(a.episodes), "--jobs", str(a.jobs)],
             ["linear_memory"], ["plots"], ["gallery"]]
    steps += [["visualize", "--scenario", sc] for sc in SCENARIOS]
    for cmd in steps:
        if run(cmd):
            sys.exit(f"ошибка на шаге {cmd[0]}")


if __name__ == "__main__":
    main()
