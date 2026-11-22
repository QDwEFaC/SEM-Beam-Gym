"""Среда Gymnasium: коррекция отклонения луча SEM при сетевой и вибрационной помехах.

Помехи задаются сценарием (scenarios.py): набор источников вибрации, здание, виброизоляция,
мода колонны, сеть, шумы датчиков. Готовые сценарии - SCENARIOS, свой собирается из блоков
disturbances.py.

Модель (одна ось, x - вдоль строки):
  - помеха смещает луч относительно образца: d(t) = d_mains(t) + d_vib(t), нм
  - агент раз в DT = 0.5 мс (2 кГц) задаёт смещение отклоняющей системы u из [-U_MAX, U_MAX],
    катушка отрабатывает команду с инерцией (апериодическое звено)
  - истинная ошибка позиции луча e(t) = d(t) - u(t), награда -mean(e^2) / SIGMA_REF^2 за шаг
  - кадр 2000 px в строке, время на пиксель (dwell) задаётся на эпизод,
    время строки T_line = 2000 * dwell, обратный ход не учитывается

Наблюдения (то, что доступно в реальном времени):
  - всегда: история своих действий, история сдвигов между строками, измеренных по изображению
    (корреляция участка строки с тем же участком предыдущей), своя коррекция строкой раньше,
    фаза сети (сигнал line sync), dwell
  - только при sensors=True: синтетические магнитометр и акселерометр на раме микроскопа

Физическое время моделируется на сетке FS = 20 кГц, SUB = 10 отсчётов на шаг управления.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sem_beam_gym.scenarios import TRAIN_SCENARIOS, Randomized, Scenario, simulate

FS = 20_000  # Гц, частота "физической" сетки
SUB = 10  # отсчётов сетки на один шаг управления
DT = SUB / FS  # 0.5 мс - период обновления коррекции
EP_STEPS = 512  # шагов в обучающем эпизоде (0.256 с)
# Длина истории в наблюдении, шагов. Аналог стека кадров в Atari: история приближает
# POMDP к марковской задаче. 16 шагов = 8 мс, этого мало для моды колонны (README, H4).
HIST = 16
N_PX = 2000  # пикселей в строке
PIXEL_NM = 5.0  # поле зрения 10 мкм / 2000 px
U_MAX = 100.0  # нм, предел коррекции
# Нормализация наблюдений: признаки делятся на фиксированный физический масштаб,
# чтобы быть порядка единицы. Фиксированный, а не бегущий (VecNormalize) - тогда наблюдение
# одинаково при обучении, оценке и переносе агента между сценариями.
SIGMA_REF = 10.0  # нм, масштаб награды и наблюдений
ACC_SCALE = 1e-3  # м/с^2, масштаб акселерометра в наблюдении
DWELL_RANGE = (0.3e-6, 500e-6)  # фиксированная нормировка признака dwell


def dwell_feature(dwell: float) -> float:
    """log10(dwell), нормировано в [-1, 1] по DWELL_RANGE."""
    # dwell меняется на три порядка, поэтому признак логарифмический. Границы фиксированы,
    # а не берутся из сценария, чтобы один и тот же dwell давал везде один и тот же признак.
    lo, hi = np.log10(DWELL_RANGE)
    return float((np.log10(dwell) - (lo + hi) / 2) / ((hi - lo) / 2))


class BeamEnv(gym.Env):
    """Коррекция луча SEM: один шаг - 0.5 мс, действие - смещение отклоняющей системы."""

    metadata = {"render_modes": []}

    def __init__(self, sensors: bool, scenario: Scenario | Randomized | str = "building"):
        # Выбор сценария, раскладка вектора наблюдения, пространства действий и наблюдений.
        # sensors - главный фактор эксперимента, всё остальное у двух условий одинаково.
        # Сценарий можно передать именем (готовые) или объектом (свой). Коэффициенты затухания
        # катушки за шаг считаются один раз, а не на каждом шаге.
        self.sensors = sensors
        self.sc = TRAIN_SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        n_h = len(self.sc.mains.harmonics)
        # раскладка вектора наблюдения
        self.idx_phase = slice(2 * HIST + 2, 2 * HIST + 2 + 2 * n_h)
        self.idx_sensors = slice(self.idx_phase.stop, self.idx_phase.stop + (2 * HIST if sensors else 0))
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx_sensors.stop,), np.float32)
        # непрерывное действие: гауссова политика actor-critic
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._decay = np.exp(-np.arange(1, SUB + 1) / FS / self.sc.actuator_tau)

    def reset(self, *, seed=None, options=None):
        """Новый эпизод: сценарий, dwell, реализация помех и сигналы датчиков."""
        # Разыгрывается сценарий эпизода и dwell, моделируются помехи на весь эпизод сразу,
        # к ним добавляются шумы датчиков, сканируется префикс без коррекции. options позволяют
        # оценке задать dwell и длину эпизода, а seed - повторить эпизод. Помехи не зависят от
        # действий агента, поэтому их можно сгенерировать заранее одним векторным вызовом - это
        # намного быстрее, чем по шагам. Префикс в одну строку нужен, чтобы уже на первом шаге
        # было с чем сравнивать строку и была полная история.
        super().reset(seed=seed)
        opts = options or {}
        rng = self.np_random
        sc = self.ep_sc = self.sc.sample(rng)  # сценарий эпизода (при domain randomization - новый)
        self.dwell = opts.get("dwell") or float(np.exp(rng.uniform(*np.log(sc.dwell))))
        self.nl = max(SUB, round(N_PX * self.dwell * FS))  # длина строки в отсчётах сетки
        # префикс: одна строка + история, сканируется до начала эпизода без коррекции
        self.k0 = math.ceil(self.nl / SUB) + HIST + 1
        n_steps = self.k0 + opts.get("steps", EP_STEPS)
        n = n_steps * SUB

        self.dist = simulate(sc, rng, n, FS)
        self.f_mains = self.dist.f_mains
        self.d = self.dist.d_mains + self.dist.d_vib
        self.u = np.zeros(n)
        self.e = self.d.copy()
        # датчики усредняются за шаг (АЦП с антиалиасингом) и зашумлены
        self.mag = self.dist.mag.reshape(-1, SUB).mean(1) + sc.sensors.mag_noise_nm * rng.standard_normal(n_steps)
        # шум акселерометра: среднее за шаг DT от белого шума плотности S -> СКО S / sqrt(2 * DT)
        acc_sigma = sc.sensors.acc_noise_ug * 1e-6 * 9.81 / math.sqrt(2 * DT)
        self.acc = self.dist.acc.reshape(-1, SUB).mean(1) + acc_sigma * rng.standard_normal(n_steps)
        self.act = np.zeros(n_steps)
        self.shift = np.zeros(n_steps)  # измеренный по изображению сдвиг (с шумом)
        self.jitter = np.zeros(n_steps)  # тот же сдвиг без шума - для метрик
        for k in range(self.k0 - HIST, self.k0):
            self._measure_shift(k)
        self.k = self.k0
        self.k_end = n_steps
        return self._obs(), {}

    def _measure_shift(self, k: int):
        """Сдвиг между текущей и предыдущей строкой на шаге k, как его видит обработка изображения."""
        # Разность ошибки луча сейчас и ровно одну строку назад (в той же точке строки) - это
        # единственная обратная связь без датчиков, корреляция соседних строк кадра. Идеальную
        # корреляцию заменяет разность средних за шаг плюс гауссов шум оценки. Без шума сдвиг
        # сохраняется отдельно, для метрики видимого "дрожания".
        a = self.e[k * SUB : (k + 1) * SUB].mean()
        b = self.e[k * SUB - self.nl : (k + 1) * SUB - self.nl].mean()
        self.jitter[k] = a - b
        self.shift[k] = a - b + self.ep_sc.shift_noise_nm * self.np_random.standard_normal()

    def _obs(self) -> np.ndarray:
        """Вектор наблюдения на текущем шаге."""
        # История действий и сдвигов строк, коррекция строкой раньше, dwell, фаза сети и, если
        # есть, история датчиков - всё, что реально доступно регулятору в момент решения.
        # История вместо рекуррентной сети - простой способ дать MLP память. Фаза сети
        # дана как sin/cos каждой гармоники, чтобы не было разрыва при переходе 2*pi.
        k, h = self.k, slice(self.k - HIST, self.k)
        u_ago = self.u[k * SUB - self.nl : (k + 1) * SUB - self.nl].mean()
        ph = 2 * np.pi * self.f_mains * k * DT  # фаза опорного сигнала сети (line sync)
        parts = [
            self.act[h],
            self.shift[h] / SIGMA_REF,
            [u_ago / U_MAX, dwell_feature(self.dwell)],
            np.concatenate([[np.sin(m * ph), np.cos(m * ph)] for m, _ in self.sc.mains.harmonics]),
        ]
        if self.sensors:
            parts += [self.mag[h] / SIGMA_REF, self.acc[h] / ACC_SCALE]
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        """Применить коррекцию на один шаг (0.5 мс) и вернуть награду."""
        # Команда в нм -> отклик катушки на сетке FS -> ошибка луча -> награда: стандартный шаг
        # Gymnasium. Катушка - апериодическое звено, его отклик внутри шага считается
        # точно (экспонента от предыдущего значения), без численного интегрирования.
        k = self.k
        a = float(np.clip(action[0], -1.0, 1.0))
        cmd = a * U_MAX
        sl = slice(k * SUB, (k + 1) * SUB)
        u_prev = self.u[k * SUB - 1]
        self.u[sl] = cmd + (u_prev - cmd) * self._decay
        self.e[sl] = self.d[sl] - self.u[sl]
        # Плотная немедленная награда, не отложенная и не разреженная. Она равна самой
        # цели (квадрат ошибки луча), а не её косвенной оценке, поэтому в симуляторе нет зазора
        # для reward hacking. Цена - награда привилегированная (README, раздел 3).
        reward = -float(np.mean(self.e[sl] ** 2)) / SIGMA_REF**2
        self.act[k] = a
        self._measure_shift(k)
        self.k += 1
        # Терминальных состояний нет (ни цели, ни провала), эпизод обрезается по времени:
        # terminated=False, truncated=True. SB3 в этом случае бутстрэпит V(s_T),
        # а не считает ценность конца эпизода нулевой.
        truncated = self.k == self.k_end
        return self._obs(), reward, False, truncated, {}

    def episode_metrics(self) -> dict:
        """Метрики завершённого (или текущего) эпизода, без префикса."""
        # Истинная ошибка, видимый межстрочный сдвиг, доля выбросов > 1 px, уровень помех.
        # Истинная ошибка - главная метрика, остальные отвечают на вопрос "что видно на снимке"
        # и позволяют разложить результат по силе событий. Межстрочный сдвиг считается только
        # там, где предыдущая строка снята в том же эпизоде, иначе он сравнивался бы с префиксом
        # без коррекции.
        sl = slice(self.k0 * SUB, self.k * SUB)
        e = self.e[sl]
        j = self.jitter[self.k0 + math.ceil(self.nl / SUB) : self.k]
        return {
            "rms_true": float(np.sqrt(np.mean(e**2))),  # истинная ошибка позиции, нм
            # видимая межстрочная ошибка, нм (делим на sqrt(2): разность двух независимых строк)
            "rms_jitter": float(np.sqrt(np.mean(j**2) / 2)) if len(j) else float("nan"),
            "frac_gt_1px": float(np.mean(np.abs(e) > PIXEL_NM)),
            # уровень помех в эпизоде без коррекции - для анализа по силе событий
            "rms_vib": float(np.sqrt(np.mean(self.dist.d_vib[sl] ** 2))),
            "rms_mains": float(np.sqrt(np.mean(self.dist.d_mains[sl] ** 2))),
        }
