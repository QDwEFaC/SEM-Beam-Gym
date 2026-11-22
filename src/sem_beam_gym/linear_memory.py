"""Проверка после основного эксперимента: сколько памяти нужно линейному фильтру.

Линейный фильтр в baselines.py видит те же 16 шагов (8 мс) датчиков, что и агент. Здесь
тот же фильтр (гребневая регрессия смещения луча по истории магнитометра, акселерометра и
фазе сети) подгоняется с историей L = 16...128 шагов вне среды, на эпизодах без коррекции.
Инерция катушки не учитывается (tau = 0.1 мс, много меньше шага 0.5 мс), поэтому это оценка
снизу для реального фильтра той же длины.

    python -m sem_beam_gym.linear_memory   # -> results/linear_memory.csv
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sem_beam_gym.env import DT, SUB, BeamEnv
from sem_beam_gym.paths import RESULTS
from sem_beam_gym.scenarios import SCENARIOS

LENGTHS = (16, 32, 64, 128)
FIT_SEEDS = range(200_000, 200_150)  # как у линейного фильтра в baselines.py
TEST_SEEDS = range(100_000, 100_030)  # как в evaluate.py


def dataset(env: BeamEnv, seeds, L: int):
    """Пары (L отсчётов датчиков + фаза сети, смещение луча) по эпизодам без коррекции."""
    # Признаки строятся напрямую из сигналов среды, а не из вектора наблюдения: наблюдение
    # среды содержит только 16 шагов истории, а здесь нужно до 128. Масштабы признаков те же,
    # что в наблюдении (1e-3 м/с^2 и 10 нм), чтобы ridge действовал одинаково при любой длине
    # истории. dwell фиксирован, так как на помеху не влияет.
    X, y = [], []
    for s in seeds:
        env.reset(seed=s, options={"dwell": 10e-6})  # dwell не влияет на помеху
        d = env.d.reshape(-1, SUB).mean(1)
        harm = [m for m, _ in env.sc.mains.harmonics]
        for k in range(max(L, env.k0), len(env.acc)):
            ph = 2 * np.pi * env.f_mains * k * DT
            # np.r_ - конкатенация массивов по строкам
            X.append(np.r_[env.acc[k - L : k] / 1e-3, env.mag[k - L : k] / 10,
                           np.sin(np.multiply(harm, ph)), np.cos(np.multiply(harm, ph)), 1.0])
            y.append(d[k])
    return np.array(X), np.array(y)


def run(sc: str) -> list[dict]:
    """RMS остатка линейного фильтра на тестовых эпизодах для каждой длины истории."""
    # Подгонка на seed 200 000+, проверка на тестовых seed 100 000+ - те же, что в основной оценке,
    # поэтому числа сравнимы с RMS RL-агента из results/eval.csv.
    env, rows = BeamEnv(sensors=True, scenario=sc), []
    for L in LENGTHS:
        X, y = dataset(env, FIT_SEEDS, L)
        w = np.linalg.solve(X.T @ X + 1e-2 * np.eye(X.shape[1]), X.T @ y)
        Xt, yt = dataset(env, TEST_SEEDS, L)
        rows.append({"scenario": sc, "hist_steps": L, "hist_ms": L * DT * 1e3,
                     "rms_nm": float(np.sqrt(np.mean((yt - Xt @ w) ** 2)))})
    print(sc, "done", flush=True)
    return rows


def main() -> pd.DataFrame:
    """Проверить все сценарии параллельно и сохранить results/linear_memory.csv."""
    # Сценарии независимы, поэтому считаются в отдельных процессах (около 30 с на 8 потоках).

    # создаем пул процессов
    # в каждом процессе вызываем функцию run для каждого сценария
    # ex.map() - возвращает список словарей (по одной строке для каждого сценария)
    # собираем все словари в один список
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(run, SCENARIOS) for r in rs]
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "linear_memory.csv", index=False)
    print(df.pivot(index="scenario", columns="hist_steps", values="rms_nm").round(2).to_string())
    return df


if __name__ == "__main__":
    main()
