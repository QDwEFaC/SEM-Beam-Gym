"""Сценарий = установка + здание + набор источников помех. И готовые "типичные" сценарии.

Свой сценарий собирается из блоков disturbances.py, например:

    from dataclasses import replace
    from sem_beam_gym.disturbances import Ambient, ForevacuumPump, Isolation, Sensors
    from sem_beam_gym.scenarios import SCENARIOS, Scenario

    my = Scenario("pump_on_air_table", vibration=(Ambient(), ForevacuumPump(rms_um_s=(10, 20))),
                  isolation=Isolation(freq=2.5))
    noisy_acc = replace(SCENARIOS["building"], name="cheap_acc", sensors=Sensors(acc_noise_ug=20))

и передаётся в BeamEnv(sensors=..., scenario=my). Чтобы сценарий попал в общий конвейер
(sem-beam), его нужно добавить в SCENARIOS в этом файле: шаги конвейера - отдельные процессы.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sem_beam_gym.disturbances import (
    Ambient,
    Column,
    Elevators,
    Footsteps,
    ForevacuumPump,
    Isolation,
    Mains,
    Sensors,
    Site,
)

# Виброизоляция и мода колонны - это цифровые фильтры (lfilter). 
# Они стартуют из нулевого состояния. Если начать эпизод сразу, 
# первые мгновения луч почти не смещается: фильтр ещё не "раскачался". 
# Начало эпизода было бы искусственно тихим, и агент видел бы переходный процесс модели, 
# а не установившуюся вибрацию.
BURN = 3.0  # с - разгон фильтров перекрытия/изоляции/колонны перед окном моделирования


@dataclass(frozen=True)
class Scenario:
    """Фиксированный сценарий: все параметры помех и установки заданы явно."""

    name: str = "custom"
    vibration: tuple = (Ambient(),)  # источники вибрации, вклады складываются
    mains: Mains = Mains()
    site: Site = Site()
    isolation: Isolation = Isolation()
    column: Column = Column()
    sensors: Sensors = Sensors()
    shift_noise_nm: float = 1.0  # СКО оценки сдвига между строками по изображению
    actuator_tau: float = 1e-4  # с, постоянная времени отклоняющей системы
    dwell: tuple[float, float] = (0.3e-6, 500e-6)  # с/пиксель, лог-равномерно на эпизод

    def sample(self, rng: np.random.Generator) -> "Scenario":
        """Сценарий конкретного эпизода. У фиксированного сценария - он сам."""
        # Общий интерфейс с Randomized: среда в reset() всегда вызывает sample() и не различает,
        # фиксированный сценарий ей передали или рандомизированный.
        return self


@dataclass(frozen=True)
class Scaled:
    """Источник вибрации с амплитудой, умноженной на k."""

    source: object
    k: float

    def floor_velocity(self, rng, n, fs, site):
        """Виброскорость исходного источника, умноженная на k, мкм/с."""
        # Обёртка нужна для domain randomization: масштаб амплитуды меняется без копирования и
        # перенастройки параметров каждого класса источника.
        return self.k * self.source.floor_velocity(rng, n, fs, site)


@dataclass(frozen=True)
class Randomized:
    """Domain randomization: в каждом эпизоде сценарий разыгрывается заново.

    Фон есть всегда. Каждый из остальных источников включается с вероятностью p_source.
    Амплитуды всех источников, частоты колонны и перекрытия и шумы датчиков случайны в
    диапазонах, покрывающих неопределённость модели (лог-равномерно). Сеть, развёртка и
    отклоняющая система - как в base (это свойства установки, известные из паспорта).

    Domain Randomization: симуляция не точнее, а разнообразнее -
    варьируются физические свойства и шумы сенсоров. Схема эксперимента повторяет пример из
    задания про LunarLander: там гравитация, здесь гипотеза о вибрации.
    """

    name: str = "randomized"
    base: Scenario = Scenario()
    optional: tuple = (ForevacuumPump(), Elevators(), Footsteps())
    p_source: float = 0.5
    amp_scale: tuple[float, float] = (0.5, 2.0)
    column_freq: tuple[float, float] = (80.0, 140.0)  # Гц
    column_damping: tuple[float, float] = (0.03, 0.08)
    slab_freq: tuple[float, float] = (9.0, 16.0)  # Гц
    acc_noise_ug: tuple[float, float] = (0.3, 3.0)
    mag_noise_nm: tuple[float, float] = (0.5, 2.0)

    # параметры, общие для всех эпизодов (нужны среде при создании)
    mains = property(lambda self: self.base.mains)
    actuator_tau = property(lambda self: self.base.actuator_tau)
    dwell = property(lambda self: self.base.dwell)
    sensors = property(lambda self: self.base.sensors)

    def sample(self, rng: np.random.Generator) -> Scenario:
        """Новый фиксированный сценарий для одного эпизода."""
        # Случайный набор источников, масштаб каждого, параметры колонны и перекрытия, шумы
        # датчиков, остальное берётся из base. Агент учится на широком семействе "лабораторий",
        # а не на одной гипотезе. Множители и шумы - лог-равномерно, потому что неопределённость
        # у них относительная (x2 вверх так же вероятно, как x2 вниз). Частоты - равномерно в
        # физически правдоподобных диапазонах.

        # линтер запрещает присваивать переменной lambda функцию, поэтому используем noqa: E731
        lu = lambda r: float(math.exp(rng.uniform(*np.log(r))))  # noqa: E731  лог-равномерно
        sources = list(self.base.vibration) + [s for s in self.optional if rng.random() < self.p_source]
        b = self.base
        return replace(
            b,
            name=self.name,
            vibration=tuple(Scaled(s, lu(self.amp_scale)) for s in sources),
            column=replace(b.column, freq=rng.uniform(*self.column_freq), damping=rng.uniform(*self.column_damping)),
            site=replace(b.site, slab_freq=rng.uniform(*self.slab_freq)),
            sensors=Sensors(mag_noise_nm=lu(self.mag_noise_nm), acc_noise_ug=lu(self.acc_noise_ug)),
        )


@dataclass
class Disturbance:
    """Реализация помех на сетке fs (n отсчётов)."""

    f_mains: float
    d_mains: np.ndarray  # смещение луча от сети, нм
    d_vib: np.ndarray  # смещение луча от вибрации, нм
    mag: np.ndarray  # магнитометр без шума, нм-экв.
    acc: np.ndarray  # ускорение рамы (акселерометр без шума), м/с^2
    v_floor: np.ndarray  # виброскорость перекрытия, мкм/с


def simulate(sc: Scenario, rng: np.random.Generator, n: int, fs: float) -> Disturbance:
    """Одна реализация всех помех сценария: n отсчётов на сетке fs."""
    # Сумма виброскоростей источников -> ускорение (производная, мкм/с -> м/с) ->
    # виброизоляция -> мода колонны, отдельно - сетевая помеха и сигнал магнитометра.
    # Единственная точка, где сценарий превращается в сигналы для среды и графиков.
    # Фильтры перекрытия, изоляции и колонны "разгоняются" BURN секунд до окна, иначе начало
    # эпизода было бы искусственно тихим (переходный процесс из нуля).
    burn = int(BURN * fs)
    v = np.zeros(n + burn)
    for src in sc.vibration:
        v += src.floor_velocity(rng, n + burn, fs, sc.site)
    a_frame = sc.isolation.apply(np.gradient(v * 1e-6, 1 / fs), fs)
    x = sc.column.displacement_nm(a_frame, fs)
    f, d_m, mag = sc.mains.generate(rng, n, fs)
    return Disturbance(f, d_m, x[burn:], mag, a_frame[burn:], v[burn:])


# ---------- типичные сценарии: микроскоп на 5-м этаже 6-этажного здания ----------

_BASE = (Ambient(),)
SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("quiet", vibration=_BASE),  # ночь: только фон здания
        Scenario("pump", vibration=_BASE + (ForevacuumPump(),)),  # насос в соседней комнате
        Scenario("elevators", vibration=_BASE + (Elevators(),)),  # 2 лифта
        Scenario("footsteps", vibration=_BASE + (Footsteps(),)),  # люди в коридоре
        Scenario("building", vibration=_BASE + (ForevacuumPump(), Elevators(), Footsteps())),  # рабочий день
    )
}
# Обучение - на каждом тестовом сценарии и дополнительно с domain randomization.
# Тестирование - только на фиксированных сценариях SCENARIOS.
TRAIN_SCENARIOS: dict[str, Scenario | Randomized] = {**SCENARIOS, "randomized": Randomized(base=SCENARIOS["quiet"])}
