"""Оценка всех стратегий на фиксированных тестовых эпизодах (seed 100_000+), без обучения.

    python -m sem_beam_gym.evaluate [--episodes 30] [--jobs 8]   # -> results/eval.csv

Матрица "сценарий обучения x сценарий теста": каждый агент (и линейный фильтр, подогнанный на
том же сценарии) проверяется на всех сценариях из SCENARIOS и сетке dwell.
Сценарии обучения - TRAIN_SCENARIOS (включая randomized), теста - SCENARIOS.
Отдельно (suite = acc_noise): агенты сценариев building и randomized при разном шуме акселерометра
(тест - сценарий building).

Процедура: жадная (детерминированная) политика, т. е. среднее гауссианы.
Фиксированные seed эпизодов, одни и те же для всех методов и не использованные в обучении.
Матрица - оценка in/out-of-distribution и проверка переноса между гипотезами о вибрации.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from multiprocessing import Pool

import pandas as pd
import torch
from stable_baselines3 import PPO

from sem_beam_gym.baselines import LinearPolicy, ZeroPolicy
from sem_beam_gym.disturbances import Sensors
from sem_beam_gym.env import BeamEnv
from sem_beam_gym.paths import RESULTS
from sem_beam_gym.scenarios import SCENARIOS, TRAIN_SCENARIOS
from sem_beam_gym.train import rollout, run_dir

TEST_SEED = 100_000
N_EPISODES = 30
SEEDS = (0, 1, 2)
DWELLS = (0.3e-6, 1e-6, 3e-6, 10e-6, 30e-6, 100e-6, 500e-6)
ACC_NOISE_UG = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # мкg/sqrt(Гц): от сейсмического датчика до дешёвого MEMS
NOISE_SCENARIO = "building"


def tests(train_scenario: str):
    """(suite, test_scenario, параметр, Scenario, сетка dwell) для агента данного сценария."""
    # Все тестовые сценарии на полной сетке dwell, а для части агентов - ещё и перебор шума
    # акселерометра. Матрица переноса для всех, требование к датчику - только там, где оно
    # интересно (агент сценария "рабочий день", DR-агент и "без коррекции" как ориентир). Перебор шума
    # делается на двух dwell, чтобы не раздувать время оценки.
    for name, sc in SCENARIOS.items():
        yield "matrix", name, "", sc, DWELLS
    if train_scenario in (NOISE_SCENARIO, "randomized", "-"):
        base = SCENARIOS[NOISE_SCENARIO]
        for s in ACC_NOISE_UG:
            sc = replace(base, sensors=Sensors(base.sensors.mag_noise_nm, s))
            yield "acc_noise", NOISE_SCENARIO, s, sc, (1e-6, 10e-6)


def tasks():
    """(метод, сценарий обучения, seed обучения)."""
    # Одна задача - одна стратегия. Задачи независимы и раздаются процессам пула.
    # "none" не зависит от сценария обучения и seed, линейный фильтр - от seed.
    yield "none", "-", -1
    for sc in TRAIN_SCENARIOS:
        yield "linear", sc, -1
        for cond in ("nosens", "sens"):
            for s in SEEDS:
                yield f"rl_{cond}", sc, s


def make_policy(method: str, train_scenario: str, seed: int):
    """Возвращает (sensors, act)."""
    # По имени метода строит стратегию как функцию obs -> action. Задачи пула передаются
    # именами (лямбды и модели SB3 не сериализуются между процессами), а политика собирается
    # уже внутри процесса-исполнителя. RL оценивается детерминированно (среднее гауссианы). 
    # Линейный фильтр подгоняется заново на том же сценарии обучения, что и агент.
    if method == "none":
        return True, ZeroPolicy().act
    if method == "linear":
        lin = LinearPolicy.fit(train_scenario)
        return True, lin.act
    cond = method.removeprefix("rl_")
    model = PPO.load(run_dir(train_scenario, cond, seed) / "model.zip", device="cpu")
    return cond == "sens", lambda o: model.predict(o, deterministic=True)[0]


def evaluate_task(task):
    """Прогнать одну стратегию по всем тестам, вернуть строки таблицы результатов."""
    # Для каждого теста и dwell - n_episodes эпизодов с seed TEST_SEED + i. Одни и те же seed
    # у всех методов, поэтому методы сравниваются на одинаковых помехах. Проверка размерности
    # наблюдения защищает от молчаливой ошибки, если сценарий теста отличается раскладкой
    # (например, другим числом гармоник сети).
    method, train_sc, seed, n_episodes = task
    torch.set_num_threads(1)
    sensors, act = make_policy(method, train_sc, seed)
    rows = []
    for suite, test_sc, param, sc, dwells in tests(train_sc):
        env = BeamEnv(sensors, sc)
        # раскладка наблюдения у всех сценариев одна (те же гармоники сети), агенты переносимы
        ref = BeamEnv(sensors, train_sc if train_sc != "-" else sc)
        assert env.observation_space.shape == ref.observation_space.shape
        for dw in dwells:
            for i in range(n_episodes):
                m = rollout(env, act, TEST_SEED + i, dw)
                rows.append(dict(suite=suite, train=train_sc, test=test_sc, param=param, dwell=dw,
                                 method=method, seed=seed, episode=i, **m))
    print("done", method, train_sc, seed, flush=True)
    return rows


def main():
    """Оценить все стратегии параллельно и сохранить results/eval.csv."""
    # chunksize=1: задачи сильно различаются по времени (подгонка фильтра дольше), так пул
    # балансирует нагрузку лучше. В конце печатается короткая сводка для быстрой проверки.
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=N_EPISODES, help="тестовых эпизодов на точку")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    a = p.parse_args()
    with Pool(a.jobs) as pool:
        parts = pool.map(evaluate_task, [(*t, a.episodes) for t in tasks()], chunksize=1)
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame([r for p in parts for r in p])
    df.to_csv(RESULTS / "eval.csv", index=False)
    m = df[(df.suite == "matrix") & (df.dwell == 10e-6) & ((df.train == df.test) | (df.method == "none"))]
    print(m.groupby(["test", "method"]).rms_true.mean().unstack().round(2))


if __name__ == "__main__":
    main()
