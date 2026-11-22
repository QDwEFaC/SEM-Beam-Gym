"""Галерея сценариев помех - без агентов, только модель.

    python -m sem_beam_gym.gallery   # -> figures/scenarios.png, figures/elevators_schedule.png

Для каждого сценария: 120 с виброскорости перекрытия у микроскопа (с уровнями критериев VC),
смещение луча от вибрации и его спектр. Отдельно - движение кабин лифтов за 30 минут.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from sem_beam_gym.disturbances import Elevators
from sem_beam_gym.env import FS
from sem_beam_gym.paths import FIGURES
from sem_beam_gym.plots import SC_LABEL
from sem_beam_gym.scenarios import SCENARIOS, simulate

SEED = 400_000
DURATION = 120.0  # с
OUT = FIGURES
VC = {"VC-A": 50.0, "VC-C": 12.5, "VC-E": 3.1}  # мкм/с, СКО (упрощённо)


def scenarios_figure():
    """Сетка сценарий x (виброскорость пола, смещение луча, спектр смещения)."""
    # Одна реализация каждого сценария длиной 120 с с одним и тем же seed. Показывает, как
    # выглядят и насколько сильны помехи. Линии VC привязывают модель к принятым
    # в отрасли критериям для чистых помещений. 120 с, чтобы в окно попали редкие события
    # (лифт, пешеходы). Сигнал прорежен в 20 раз только для рисования, спектр (метод Уэлча)
    # считается по полному сигналу.
    names = list(SCENARIOS)
    fig, ax = plt.subplots(len(names), 3, figsize=(16, 2.6 * len(names)),
                           gridspec_kw={"width_ratios": [2.2, 2.2, 1.2]})
    t = np.arange(int(DURATION * FS)) / FS
    for r, name in enumerate(names):
        d = simulate(SCENARIOS[name], np.random.default_rng(SEED), len(t), FS)
        dec = slice(None, None, 20)  # прореживание для рисования
        ax[r, 0].plot(t[dec], d.v_floor[dec], lw=0.4, color="#555")
        for k, (lbl, v) in enumerate(VC.items()):
            ax[r, 0].axhline(v, lw=0.6, ls="--", color=f"C{k + 3}", label=lbl if r == 0 else None)
        ax[r, 0].set_ylabel(f"{SC_LABEL[name]}\nv пола, мкм/с")
        ax[r, 1].plot(t[dec], d.d_vib[dec], lw=0.4, color="#1f77b4")
        ax[r, 1].set_ylabel("смещение луча, нм")
        f, pxx = welch(d.d_vib, fs=FS, nperseg=2**15)
        ax[r, 2].semilogy(f, pxx, lw=0.8)
        ax[r, 2].set(xlim=(0, 200), ylim=(1e-4, 1e3), ylabel=r"нм$^2$/Гц")
        ax[r, 1].text(0.01, 0.92, f"RMS {d.d_vib.std():.1f} нм, |max| {np.abs(d.d_vib).max():.0f} нм",
                      transform=ax[r, 1].transAxes, fontsize=8, va="top")
        for a in ax[r]:
            a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7, loc="upper right")
    ax[0, 0].set_title("Виброскорость перекрытия у микроскопа (пунктир - уровни VC)")
    ax[0, 1].set_title("Смещение луча от вибрации (после моды колонны)")
    ax[0, 2].set_title("Спектр смещения")
    ax[-1, 0].set_xlabel("время, с")
    ax[-1, 1].set_xlabel("время, с")
    ax[-1, 2].set_xlabel("частота, Гц")
    fig.suptitle(f"Сценарии вибрации, одна реализация {DURATION:.0f} с (сетевая ЭМ-помеха не показана)")
    fig.tight_layout()
    fig.savefig(OUT / "scenarios.png", dpi=100)
    return fig


def elevators_figure():
    """Этаж каждой кабины во времени за 30 минут."""
    # Ступенчатый график по расписанию из Elevators.schedule: модель лифтов даёт правдоподобный
    # поток поездок (очереди, частые рейсы на первый этаж), а не случайный шум. Профиль
    # скорости внутри поездки опущен - на масштабе минут он не виден.
    sc = SCENARIOS["elevators"]
    el = next(s for s in sc.vibration if isinstance(s, Elevators))
    rng = np.random.default_rng(SEED)
    T = 1800.0
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for car in range(el.n_cars):
        moves, stops = el.schedule(rng, T, sc.site)
        pts = [(t0, f0) for t0, f0, _ in moves] + [(t0 + 0.0, f) for t0, f, _ in stops]
        pts = sorted(p for p in pts if 0 <= p[0] <= T)
        if pts:
            ax.step([p[0] / 60 for p in pts], [p[1] for p in pts], where="post", lw=1, label=f"лифт {car + 1}")
    ax.axhline(sc.site.microscope_floor, color="k", ls=":", lw=1, label="этаж микроскопа")
    ax.set(xlabel="время, мин", ylabel="этаж", yticks=range(1, sc.site.floors + 1),
           title=f"Лифты: {el.calls_per_hour:.0f} вызовов/ч на лифт (пуассоновский поток), 30 минут")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "elevators_schedule.png", dpi=100)
    return fig


def main():
    """Построить оба рисунка галереи."""
    OUT.mkdir(exist_ok=True)
    scenarios_figure()
    elevators_figure()


if __name__ == "__main__":
    main()
