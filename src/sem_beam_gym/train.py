"""Обучение PPO в одном сценарии, одном условии, с одним seed.

    python -m sem_beam_gym.train --scenario building --cond sens --seed 0 --steps 1000000

Условия: nosens (без датчиков) и sens (магнитометр + акселерометр).
Всё остальное - среда, награда, сеть, гиперпараметры, бюджет - одинаково.
Сохраняет runs/<scenario>/<cond>_s<seed>/: model.zip, monitor/ (логи эпизодов), val.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from sem_beam_gym.env import BeamEnv
from sem_beam_gym.paths import RUNS
from sem_beam_gym.scenarios import TRAIN_SCENARIOS

VAL_DWELLS = (0.3e-6, 3e-6, 10e-6, 100e-6)
VAL_SEED = 50_000  # валидационные эпизоды, тестовые seed начинаются с 100_000
N_ENVS = 8

# PPO: rollout 512 шагов x 8 сред = 4096 переходов, по ним
# n_epochs эпох мини-батчами, затем новый сбор свежей политикой (on-policy, без replay buffer).
# По умолчанию SB3 включает ортогональную инициализацию, нормализацию преимуществ
# и раздельные сети актора и критика.
PPO_KW = dict(
    n_steps=512,
    batch_size=512,
    n_epochs=10,
    # постоянный LR. Линейное затухание не нужно: кривые выходят на плато к 0.5M шагов
    learning_rate=3e-4,
    # Дисконт: награда почти мгновенная, а не отложенная. Горизонт
    # 1 / (1 - gamma) = 10 шагов покрывает инерцию катушки и запаздывание обратной связи по строкам.
    gamma=0.9,
    # GAE: lambda сдвигает оценку преимущества между TD (lambda = 0, меньше
    # дисперсия) и Монте-Карло (lambda = 1, меньше смещение).
    gae_lambda=0.95,
    clip_range=0.2,  # epsilon клипирования отношения pi_new / pi_old в L^CLIP
    # Исследование обеспечивает sigma гауссовой политики, без энтропийного бонуса.
    ent_coef=0.0,
    # Гауссова политика для непрерывного действия. В SB3 log(sigma) - обучаемый параметр,
    # не зависящий от состояния (в лекциях - отдельная голова сети).
    # log(sigma) = -1 -> sigma = 0.37 в долях U_MAX, то есть 37 нм.
    policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),
)


def run_dir(scenario: str, cond: str, seed: int, root: Path | str = RUNS) -> Path:
    """Папка запуска: runs/<scenario>/<cond>_s<seed>."""
    # Единое правило именования для обучения, оценки и графиков - чтобы их нельзя было рассинхронизировать.
    return Path(root) / scenario / f"{cond}_s{seed}"


def rollout(env: BeamEnv, act, seed: int, dwell: float | None = None, steps: int | None = None):
    """Один эпизод с заданным seed, act(obs) -> action. Возвращает метрики эпизода."""
    # Полный эпизод с любой стратегией (RL, линейный фильтр, нулевая) - одна процедура для
    # валидации, теста и визуализации, условия сравнимы. Стратегия передаётся функцией, а не
    # объектом SB3, поэтому бейзлайны и агенты проходят ровно через один и тот же код.
    # Фиксированный seed даёт одинаковые помехи у всех.
    opts = {k: v for k, v in (("dwell", dwell), ("steps", steps)) if v is not None}
    obs, _ = env.reset(seed=seed, options=opts)
    done = False
    while not done:
        obs, _, _, done, _ = env.step(act(obs))
    return env.episode_metrics()


class ValCallback(BaseCallback):
    """Каждые `every` шагов: RMS истинной ошибки на фиксированных валидационных эпизодах.

    Как в DQN: валидация жадной (детерминированной) политикой на своих seed, не
    пересекающихся с тестом. RMS - монотонное преобразование средней награды эпизода:
    RMS = SIGMA_REF * sqrt(-mean reward), поэтому это и есть кривая обучения по награде, в нм.
    """

    def __init__(self, env: BeamEnv, every: int, n_per_dwell: int = 4):
        # Отдельная среда, а не обучающая: валидация не должна сбивать состояние векторизованных сред.
        super().__init__()
        self.env, self.every, self.n = env, every, n_per_dwell
        self.rows, self.last = [], -every

    def _evaluate(self):
        """Средняя RMS жадной политики на 4 dwell x n_per_dwell эпизодах."""
        # Несколько dwell, потому что качество сильно зависит от режима съёмки. Одни и те же
        # seed на каждой проверке, поэтому изменение кривой - это изменение политики, а не шум.
        act = lambda o: self.model.predict(o, deterministic=True)[0]  # noqa: E731
        rms = [rollout(self.env, act, VAL_SEED + i, dw)["rms_true"] for dw in VAL_DWELLS for i in range(self.n)]
        self.rows.append({"step": self.num_timesteps, "val_rms": float(np.mean(rms))})

    def _on_step(self):
        """Запустить валидацию, если с прошлой прошло не меньше every шагов."""
        # SB3 вызывает это после каждого шага всех 8 сред (num_timesteps растёт на 8), поэтому
        # сравнивается разность, а не остаток от деления.
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self._evaluate()
        return True

    def _on_training_end(self):
        """Последняя точка кривой - на итоговой модели."""
        self._evaluate()


def train(scenario: str, cond: str, seed: int, steps: int = 1_000_000, root: Path | str = RUNS) -> Path:
    """Один запуск PPO, возвращает папку с model.zip, monitor/ и val.csv."""
    # 8 сред с логами Monitor, PPO с общими гиперпараметрами, периодическая валидация - базовая
    # единица матрицы эксперимента (сценарий x условие x seed). Один поток torch на запуск:
    # сеть маленькая, а запуски идут параллельно процессами, так что многопоточность внутри
    # только мешала бы. device="cpu" - на GPU эта задача медленнее (время уходит на шаги
    # среды, а не на сеть).
    torch.set_num_threads(1)
    out = run_dir(scenario, cond, seed, root)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"sensors": cond == "sens", "scenario": scenario}
    # векторизованные среды: одна модель учится на опыте 8 независимых копий
    venv = make_vec_env(BeamEnv, N_ENVS, seed=seed, env_kwargs=kw, monitor_dir=str(out / "monitor"))
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **PPO_KW)
    cb = ValCallback(BeamEnv(**kw), every=max(steps // 20, 10_000))
    model.learn(steps, callback=cb)
    model.save(out / "model")
    pd.DataFrame(cb.rows).to_csv(out / "val.csv", index=False)
    print("/".join(out.parts[-3:]), cb.rows[-1], flush=True)
    return out


def main():
    """Точка входа командной строки: python -m sem_beam_gym.train ..."""
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(TRAIN_SCENARIOS), required=True)
    p.add_argument("--cond", choices=["nosens", "sens"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", default=str(RUNS))
    a = p.parse_args()
    train(a.scenario, a.cond, a.seed, a.steps, a.out)


if __name__ == "__main__":
    main()
