"""Строительные блоки модели помех SEM.

Цепочка для вибрации:

    источники -> виброскорость перекрытия v(t), мкм/с  (сумма вкладов всех источников)
              -> ускорение перекрытия a = dv/dt
              -> виброизоляция (опционально) -> ускорение рамы микроскопа (его видит акселерометр)
              -> мода колонны (кинематическое возбуждение) -> смещение луча относительно образца, нм

Амплитуды источников задаются виброскоростью перекрытия в точке установки микроскопа, то есть в тех
же единицах, что и критерии VC для чистых помещений (VC-A = 50 мкм/с, VC-C = 12.5, VC-E = 3.1).

Электромагнитная помеха от сети моделируется отдельно (класс Mains) и сразу даёт смещение луча.

Свой источник вибрации - любой объект с методом
    floor_velocity(rng, n, fs, site) -> np.ndarray  # n отсчётов виброскорости, мкм/с
где отсчёт 0 соответствует началу окна моделирования. Процессы, которые "идут" и до начала окна
(поездки лифта, проход человека), источник досчитывает сам, начиная с момента -lookback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfilt

# ---------- вспомогательные сигналы ----------


def band_noise(rng, n: int, fs: float, lo: float, hi: float) -> np.ndarray:
    """Полосовой гауссов шум с единичным СКО."""
    # Белый шум пропускается через полосовой Баттерворт 2-го порядка и нормируется на СКО.
    # Так получается "несущая" для фона здания и шума направляющих лифта. SOS-форма устойчива
    # численно. Первые burn отсчётов (4 периода нижней частоты) отбрасываются, чтобы не попал
    # переходный процесс фильтра. Единичное СКО позволяет задавать амплитуду снаружи одним множителем.
    sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    burn = int(4 * fs / lo)
    x = sosfilt(sos, rng.standard_normal(n + burn))[burn:]
    return x / x.std()


def slow_noise(rng, n: int, fs: float, period: float) -> np.ndarray:
    """Гладкий случайный процесс (кусочно-линейная интерполяция узлов через `period` с)."""
    # Случайные значения в узлах через period секунд, между ними линейная интерполяция.
    # Нужен медленный дрейф частоты насоса (нагрузка, давление): дёшево и без высоких частот,
    # которые для дрейфа нетипичны.
    m = int(n / fs / period) + 3
    return np.interp(np.arange(n) / fs / period, np.arange(m), rng.standard_normal(m))


def ringing(times, amps, n: int, fs: float, f: float, zeta: float) -> np.ndarray:
    """
    Отклик перекрытия на удары: каждый импульс даёт затухающую синусоиду с пиком ~ amp.
    times - моменты ударов, с (могут быть < 0: "звон" от ударов до начала окна попадает в окно).
    """
    # Импульсы ставятся в буфер и пропускаются через резонатор 2-го порядка (мода перекрытия).
    # Шаг, тормоз лифта и дверь - короткие удары, глазами датчика это "звон" перекрытия.
    # Рекурсивный фильтр (AR(2)) считает отклик на любое число ударов за O(n), без свёртки с
    # длинной импульсной характеристикой. Буфер начинается на lead отсчётов раньше окна: за это
    # время отклик затухает в e^5 раз, поэтому удары до начала окна учтены корректно.
    w = 2 * np.pi * f / fs
    r = math.exp(-zeta * w)
    lead = int(5 / (zeta * 2 * np.pi * f) * fs)
    buf = np.zeros(n + lead)
    idx = np.round(np.asarray(times, float) * fs).astype(int) + lead
    keep = (idx >= 0) & (idx < len(buf))
    np.add.at(buf, idx[keep], np.asarray(amps, float)[keep])
    # импульсная характеристика r^k * sin(k*w): пик ~ 1 при k*w ~ pi/2, поэтому amp и есть пик отклика
    return lfilter([0.0, r * math.sin(w)], [1.0, -2 * r * math.cos(w), r * r], buf)[lead:]


def attenuation(r, r0: float):
    """Затухание вибрации с расстоянием r, м (геометрическое расхождение, без фанатизма)."""
    # Множитель 1 / (1 + r/r0): удар у шахты лифта или шаг в конце коридора слабее, чем рядом
    # с микроскопом. Простейший закон, ограниченный при r -> 0 (в отличие от 1/r) и спадающий
    # при больших r. Точная волновая модель перекрытия для этой задачи избыточна.
    return 1.0 / (1.0 + np.asarray(r) / r0)


def poisson_times(rng, rate_per_hour: float, t0: float, t1: float) -> np.ndarray:
    """Моменты событий пуассоновского потока на интервале [t0, t1), с."""
    # Накопленная сумма экспоненциальных интервалов со средним 3600 / rate даёт моменты вызовов
    # лифта и появления пешеходов - независимые редкие события. Интервалы генерируются одним
    # векторным вызовом с запасом (x1.5 + 20), лишние отбрасываются: быстрее цикла и тот же
    # пуассоновский поток.
    if rate_per_hour <= 0:
        return np.zeros(0)
    mean = 3600 / rate_per_hour
    k = int((t1 - t0) / mean * 1.5) + 20
    t = t0 + np.cumsum(rng.exponential(mean, k))
    return t[t < t1]


# ---------- здание и микроскоп ----------


@dataclass(frozen=True)
class Site:
    """Здание и место установки микроскопа."""

    floors: int = 6
    floor_height: float = 3.5  # м
    microscope_floor: int = 5  # 1 - первый этаж
    slab_freq: float = 12.0  # Гц, собственная частота перекрытия (типично 8-20 Гц)
    slab_damping: float = 0.03

    def floor_z(self, floor) -> np.ndarray:
        """Высота этажа над первым, м."""
        # Нужна для расстояния от кабины лифта до микроскопа. Этажи считаются с 1, как в здании.
        return (np.asarray(floor) - 1) * self.floor_height


@dataclass(frozen=True)
class Isolation:
    """Пассивная виброизоляция (пневмоопоры, резиновые опоры). freq=None - жёстко на полу."""

    freq: float | None = None  # Гц
    damping: float = 0.1

    def apply(self, a: np.ndarray, fs: float) -> np.ndarray:
        """Ускорение рамы микроскопа по ускорению пола."""
        # Передаточная функция опоры (2*zeta*omega*s + omega^2) / (s^2 + 2*zeta*omega*s + omega^2).
        # На опорах колонна и акселерометр видят уже ослабленную вибрацию. Это стандартная модель
        # пассивной опоры (масса на пружине с демпфером). Непрерывная функция переводится в
        # цифровой фильтр билинейным преобразованием, которое сохраняет устойчивость.
        if self.freq is None:
            return a
        w = 2 * np.pi * self.freq
        b, a_ = bilinear([2 * self.damping * w, w * w], [1, 2 * self.damping * w, w * w], fs)
        return lfilter(b, a_, a)


@dataclass(frozen=True)
class Column:
    """Мода "колонна относительно столика с образцом" при кинематическом возбуждении основания.

    x'' + 2*zeta*omega*x' + omega^2*x = -a_base  ->  ниже резонанса x ~ -a/omega^2,
    у резонанса усиление 1/(2*zeta).
    """

    freq: float = 110.0  # Гц
    damping: float = 0.05

    def displacement_nm(self, a: np.ndarray, fs: float) -> np.ndarray:
        """Смещение луча относительно образца, нм, по ускорению рамы, м/с^2."""
        # Решение уравнения моды из docstring класса цифровым фильтром: именно это относительное
        # смещение искажает снимок. Одна мода - минимальная модель, которая даёт и рост ошибки
        # на низких частотах (x ~ a/omega^2) и резонансное усиление. Множитель -1e9 переводит
        # метры в нанометры и учитывает знак правой части.
        w = 2 * np.pi * self.freq
        b, a_ = bilinear([-1e9], [1, 2 * self.damping * w, w * w], fs)
        return lfilter(b, a_, a)


# ---------- источники вибрации ----------


@dataclass(frozen=True)
class Ambient:
    """Фон перекрытия: транспорт, вентиляция, оборудование здания - полосовой шум."""

    rms_um_s: tuple[float, float] = (0.5, 1.0)  # СКО виброскорости, мкм/с (на эпизод - равномерно)
    band: tuple[float, float] = (4.0, 100.0)  # Гц

    def floor_velocity(self, rng, n, fs, site):
        """Виброскорость фона, мкм/с."""
        # Фон - сумма многих независимых источников, поэтому моделируется гауссовым полосовым
        # шумом. Уровень случаен на эпизод, чтобы агент не привыкал к одному значению.
        return rng.uniform(*self.rms_um_s) * band_noise(rng, n, fs, *self.band)


@dataclass(frozen=True)
class ForevacuumPump:
    """Пластинчато-роторный форвакуумный насос в соседней комнате - тональная вибрация.

    Асинхронный 4-полюсный двигатель: 1500 об/мин минус скольжение -> f_rot ~ 24 Гц.
    Гармоники: x1 - дисбаланс ротора, x2 - проход двух пластин, x3 - следующая.
    Частота медленно дрейфует с нагрузкой (газобалласт, давление).
    """

    rms_um_s: tuple[float, float] = (4.0, 8.0)  # в точке микроскопа (между комнатами стена)
    f_rot: tuple[float, float] = (23.8, 24.6)  # Гц
    harmonics: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 0.6), (3, 0.3))
    drift_hz: float = 0.05  # СКО медленного дрейфа частоты, Гц
    drift_period: float = 2.0  # с

    def floor_velocity(self, rng, n, fs, site):
        """Виброскорость от насоса, мкм/с."""
        # Сумма гармоник с общей фазой, полученной интегрированием дрейфующей частоты.
        # Нужна для гипотезы H2: тональную помеху агент может предсказать по изображению.
        # Фаза как интеграл частоты не даёт скачков при дрейфе. Амплитуды гармоник слегка
        # случайны (+-20 %), итог нормируется на заданное СКО.
        f = rng.uniform(*self.f_rot) + self.drift_hz * slow_noise(rng, n, fs, self.drift_period)
        phase = 2 * np.pi * np.cumsum(f) / fs
        v = sum(rel * rng.uniform(0.8, 1.2) * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
                for h, rel in self.harmonics)
        return v * (rng.uniform(*self.rms_um_s) / v.std())


@dataclass(frozen=True)
class Elevators:
    """Пассажирские лифты, машинное отделение на крыше.

    Вызовы каждого лифта - пуассоновский поток. Вызов: этаж посадки (50 % - первый этаж, иначе
    любой) -> этаж назначения (с первого - любой другой, с остальных 70 % на первый). Лифт едет к
    этажу посадки, стоит с открытыми дверями, везёт пассажира, снова стоит. Вызовы, пришедшие во
    время поездки, ждут в очереди. Движение - трапециевидный профиль скорости (разгон, ход,
    торможение).

    Вклад в вибрацию у микроскопа:
      * шум направляющих при движении: пропорционален скорости кабины, затухает с расстоянием
      * машина на крыше во время движения
      * удары при старте/остановке (тормоз) и при открывании/закрывании дверей - "звон" перекрытия
    """

    n_cars: int = 2
    calls_per_hour: float = 40.0  # на один лифт
    speed: float = 1.0  # м/с
    accel: float = 0.8  # м/с^2
    door_time: tuple[float, float] = (4.0, 9.0)  # с, стоянка с открытыми дверями
    shaft_distance: float = 15.0  # м по горизонтали от микроскопа до шахты
    run_um_s: float = 15.0  # СКО виброскорости от направляющих на расстоянии r0 при номинальной скорости
    machine_um_s: float = 10.0  # СКО от машины на расстоянии r0
    jerk_um_s: float = 150.0  # пик удара тормоза на расстоянии r0
    door_um_s: float = 60.0  # пик удара двери на расстоянии r0
    r0: float = 3.0  # м
    lookback: float = 900.0  # с - "разгон" очереди вызовов до начала окна

    def _travel_time(self, dist: float) -> tuple[float, float, float]:
        """(время разгона, время хода, пиковая скорость) для перемещения на dist метров."""
        # Трапециевидный профиль: разгон с accel до speed, ход, торможение. На короткой поездке
        # (один этаж) кабина не успевает набрать speed - профиль становится треугольным.
        t_acc = self.speed / self.accel
        if self.accel * t_acc**2 >= dist:  # не успевает разогнаться
            t_acc = math.sqrt(dist / self.accel)
            return t_acc, 0.0, self.accel * t_acc
        return t_acc, (dist - self.accel * t_acc**2) / self.speed, self.speed

    def schedule(self, rng, t_end: float, site: Site):
        """Поездки одного лифта: список (t_старта, этаж_откуда, этаж_куда) и стоянок (t, этаж, длит.)."""
        # Очередь вызовов обслуживается по порядку (FIFO): подъезд к этажу посадки, двери,
        # поездка к этажу назначения, двери. Из расписания строятся и шум движения, и удары
        # тормоза и дверей. FIFO - простейшая логика диспетчера, достаточная для статистики
        # событий. Моделирование начинается за lookback секунд до окна, чтобы в начале окна
        # очередь и положение кабины были "установившимися", а не пустыми.
        F = site.floors
        moves, stops = [], []
        cur, free = int(rng.integers(1, F + 1)), -self.lookback
        for tc in poisson_times(rng, self.calls_per_hour, -self.lookback, t_end):
            origin = 1 if rng.random() < 0.5 else int(rng.integers(1, F + 1))
            if origin == 1:
                dest = int(rng.integers(2, F + 1))
            elif rng.random() < 0.7:
                dest = 1
            else:
                dest = int(rng.choice([f for f in range(1, F + 1) if f != origin]))
            t = max(tc, free)
            for target in (origin, dest):
                if target != cur:
                    moves.append((t, cur, target))
                    ta, tc_, _ = self._travel_time(abs(target - cur) * site.floor_height)
                    t += 2 * ta + tc_
                    cur = target
                door = rng.uniform(*self.door_time)
                stops.append((t, cur, door))
                t += door
            free = t
        return moves, stops

    def floor_velocity(self, rng, n, fs, site):
        """Виброскорость от лифтов, мкм/с."""
        # По расписанию каждой кабины строятся огибающие шума направляющих и машины и список
        # ударов (тормоз на старте и в конце поездки, двери). Лифт - самый "событийный" источник,
        # нужный для проверки H2 и H3. Шум - полосовой гауссов, умноженный на огибающую
        # (амплитуда растёт со скоростью и падает с расстоянием до кабины). Удары идут через
        # общий резонатор перекрытия. Поездки, целиком лежащие вне окна (с запасом 3 с на
        # "звон"), пропускаются.
        T = n / fs
        t = np.arange(n) / fs
        zm = site.floor_z(site.microscope_floor)
        dist = lambda z: np.hypot(self.shaft_distance, np.asarray(z) - zm)  # noqa: E731
        k_machine = attenuation(dist(site.floors * site.floor_height + 1.0), self.r0)
        run_env, mach_env = np.zeros(n), np.zeros(n)
        imp_t, imp_a = [], []
        for _ in range(self.n_cars):
            moves, stops = self.schedule(rng, T, site)
            for t0, f_from, f_to in moves:
                ta, tc, vp = self._travel_time(abs(f_to - f_from) * site.floor_height)
                dur = 2 * ta + tc
                if t0 + dur < -3 or t0 > T:
                    continue
                z0, sgn = site.floor_z(f_from), np.sign(f_to - f_from)
                imp_t += [t0, t0 + dur]
                imp_a += [self.jerk_um_s * attenuation(dist(z0), self.r0),
                          self.jerk_um_s * attenuation(dist(site.floor_z(f_to)), self.r0)]
                i = (t >= t0) & (t < t0 + dur)
                if not i.any():
                    continue
                tau = t[i] - t0
                # скорость и путь по трапециевидному профилю
                vel = np.minimum.reduce([self.accel * tau, np.full_like(tau, vp), self.accel * (dur - tau)])
                s = np.where(tau < ta, self.accel * tau**2 / 2,
                             np.where(tau < ta + tc, self.accel * ta**2 / 2 + vp * (tau - ta),
                                      2 * self.accel * ta**2 / 2 + vp * tc - self.accel * (dur - tau) ** 2 / 2))
                z = z0 + sgn * s
                run_env[i] += self.run_um_s * (vel / self.speed) * attenuation(dist(z), self.r0)
                mach_env[i] += self.machine_um_s * k_machine * (vel > 0)
            for t0, fl, door in stops:
                if -3 < t0 < T:
                    k = self.door_um_s * attenuation(dist(site.floor_z(fl)), self.r0)
                    imp_t += [t0 + 0.5, t0 + door - 1.0]
                    imp_a += [k, k]
        v = run_env * band_noise(rng, n, fs, 10.0, 60.0) + mach_env * band_noise(rng, n, fs, 20.0, 80.0)
        return v + ringing(imp_t, imp_a, n, fs, site.slab_freq, site.slab_damping)


@dataclass(frozen=True)
class Footsteps:
    """Люди проходят по коридору этажа мимо комнаты с микроскопом.

    Появление пешеходов - пуассоновский поток. Каждый проходит коридор целиком в случайную
    сторону с постоянной скоростью и частотой шага. Каждый шаг - удар по перекрытию, затухающий
    с расстоянием до микроскопа, перекрытие "звенит" на своей собственной частоте.
    """

    walkers_per_hour: float = 60.0
    corridor_length: float = 40.0  # м
    room_position: float = 20.0  # м вдоль коридора - напротив комнаты с микроскопом
    room_offset: float = 5.0  # м от линии движения до микроскопа
    speed: tuple[float, float] = (1.1, 1.6)  # м/с
    step_freq: tuple[float, float] = (1.7, 2.2)  # Гц
    step_um_s: tuple[float, float] = (200.0, 600.0)  # пик виброскорости от шага на расстоянии r0
    r0: float = 2.0  # м

    def floor_velocity(self, rng, n, fs, site):
        """Виброскорость от шагов, мкм/с."""
        # Для каждого пешехода - моменты шагов и их амплитуды с учётом расстояния до микроскопа.
        # Все шаги проходят через резонатор перекрытия. Импульсная нестационарная помеха -
        # главный контраст с тональным насосом (H2). Шаги идут с небольшим разбросом во времени
        # (10 мс) и силе (+-15 %), как у живого человека. Пешеходы, вошедшие в коридор до начала
        # окна, тоже учитываются.
        T = n / fs
        lookback = self.corridor_length / self.speed[0] + 3
        times, amps = [], []
        for ta in poisson_times(rng, self.walkers_per_hour, -lookback, T):
            v, fst, amp = rng.uniform(*self.speed), rng.uniform(*self.step_freq), rng.uniform(*self.step_um_s)
            k = np.arange(int(self.corridor_length / v * fst))
            tk = ta + k / fst + rng.normal(0, 0.01, len(k))
            x = k / fst * v
            if rng.random() < 0.5:
                x = self.corridor_length - x
            r = np.hypot(self.room_offset, x - self.room_position)
            times.append(tk)
            amps.append(amp * attenuation(r, self.r0) * rng.uniform(0.85, 1.15, len(k)))
        if not times:
            return np.zeros(n)
        return ringing(np.concatenate(times), np.concatenate(amps), n, fs, site.slab_freq, site.slab_damping)


# ---------- сеть и датчики ----------


@dataclass(frozen=True)
class Mains:
    """ЭМ-помеха от сети: гармоники, случайные амплитуды и фазы, медленная АМ.

    Магнитометр стоит не на оси колонны: поле в точке датчика отличается от поля, отклоняющего
    луч, усилением и фазой по гармоникам (экраны и вихревые токи сильнее на высоких частотах).
    """

    freq: tuple[float, float] = (49.9, 50.1)  # Гц
    # (номер гармоники, диапазон амплитуды смещения луча, нм)
    harmonics: tuple[tuple[int, tuple[float, float]], ...] = ((1, (6.0, 14.0)), (3, (2.0, 6.0)), (5, (1.0, 3.0)))
    am_depth: float = 0.15
    mag_gain: tuple[float, ...] = (1.0, 0.8, 0.6)
    mag_phase: tuple[float, ...] = (0.2, 0.5, 0.9)  # рад

    def generate(self, rng, n, fs):
        """-> (частота сети, смещение луча нм, сигнал магнитометра в нм-эквиваленте без шума)."""
        # Нечётные гармоники сети со случайными амплитудами и фазами и общей медленной
        # амплитудной модуляцией (нагрузка в сети меняется). При строке 20 мс помеха одинакова
        # во всех строках и поэтому не видна по изображению - ключевой механизм гипотезы H1.
        # Магнитометр получает те же гармоники, но с другим усилением и фазой, поэтому агенту
        # нужно выучить пересчёт, а не просто скопировать сигнал датчика.
        t = np.arange(n) / fs
        f = rng.uniform(*self.freq)
        am = 1 + self.am_depth * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 2 * np.pi))
        d, mag = np.zeros(n), np.zeros(n)
        for (h, amp), g, dphi in zip(self.harmonics, self.mag_gain, self.mag_phase):
            a, phi = rng.uniform(*amp), rng.uniform(0, 2 * np.pi)
            d += a * am * np.sin(2 * np.pi * h * f * t + phi)
            mag += g * a * am * np.sin(2 * np.pi * h * f * t + phi + dphi)
        return f, d, mag


@dataclass(frozen=True)
class Sensors:
    """Шумы синтетических датчиков."""

    mag_noise_nm: float = 1.0  # магнитометр: СКО на отсчёт, нм-эквивалент смещения луча
    acc_noise_ug: float = 1.0  # акселерометр на раме: спектральная плотность шума, мкg/sqrt(Гц)
