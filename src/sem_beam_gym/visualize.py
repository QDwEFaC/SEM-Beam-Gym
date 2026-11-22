"""Поведение итоговых агентов на одной и той же реализации помех.

    python -m sem_beam_gym.visualize --scenario building   # -> figures/<scenario>_{images,traces,psd,distortion}.png

Все стратегии прогоняются с одним seed эпизода, поэтому помеха у них идентична.
Используются агенты seed 0, обученные на том же сценарии.
Снимок: синтетический образец (решётка кругов) сканируется построчно. Пиксель p строки i
снимается в момент t = i * T_line + p * dwell, луч попадает в точку x = p + e(t) / PIXEL_NM.
Показан фрагмент 256 x 256 px из середины строки (столбцы 872-1127).
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from sem_beam_gym.env import FS, N_PX, PIXEL_NM, SUB, BeamEnv
from sem_beam_gym.evaluate import make_policy
from sem_beam_gym.paths import FIGURES
from sem_beam_gym.plots import COLOR, LABEL, SC_LABEL
from sem_beam_gym.scenarios import SCENARIOS

SEED = 300_000
METHODS = ("none", "linear", "rl_nosens", "rl_sens")
CROP = 256
COL0 = N_PX // 2 - CROP // 2
OUT = FIGURES


def sample(x, y):
    """Интенсивность образца: круги радиуса 12 px на сетке 40 px, мягкий край."""
    # Круги, а не реальный снимок: на правильной решётке любое смещение луча сразу видно глазом.
    # Мягкий край (логистическая функция) имитирует конечный размер пучка и убирает ступеньки.
    r = np.hypot(np.mod(x, 40) - 20, np.mod(y, 40) - 20)
    return 0.15 + 0.85 / (1 + np.exp((r - 12) / 0.6))


def long_rollout(method, scenario, dwell, n_lines):
    """Ошибка луча на сетке FS за n_lines строк для одной стратегии."""
    # Один длинный эпизод (не 0.256 с, а столько, сколько нужно на n_lines строк): снимок
    # 256 строк при 10 мкс/px - это 5 с, обычный эпизод слишком короткий. Агент работает и на
    # эпизодах длиннее обучающих, потому что его наблюдение не зависит от номера шага. Seed
    # общий для всех стратегий - помехи одинаковые.
    sensors, act = make_policy(method, scenario, -1 if method in ("none", "linear") else 0)
    env = BeamEnv(sensors, scenario)
    nl = max(SUB, round(N_PX * dwell * FS))
    obs, _ = env.reset(seed=SEED, options={"dwell": dwell, "steps": math.ceil(n_lines * nl / SUB) + 1})
    done = False
    while not done:
        obs, _, _, done, _ = env.step(act(obs))
    return env.e[env.k0 * SUB :]  # ошибка на сетке FS от начала эпизода


def pixel_error(e, dwell, lines, cols, sub=4):
    """Ошибка в нм для каждого (строка, столбец, подотсчёт внутри пикселя)."""
    # Время съёмки каждой точки известно из развёртки (строки идут подряд, без обратного хода),
    # ошибка в этот момент берётся линейной интерполяцией по сетке FS. Подотсчёты внутри пикселя
    # нужны, чтобы при длинном dwell пиксель считался функцией от отсчетов.
    i, p, q = np.meshgrid(lines, cols, (np.arange(sub) + 0.5) / sub, indexing="ij")
    t = (i * N_PX + p + q) * dwell
    return np.interp(t * FS, np.arange(len(e)), e), p + q - 0.5, i


def render(e, dwell):
    """Синтетический снимок фрагмента CROP x CROP при ошибке луча e."""
    # Луч смещён на e / PIXEL_NM пикселей вдоль строки. Яркость берётся из образца в этой точке.
    err, x, y = pixel_error(e, dwell, np.arange(CROP), COL0 + np.arange(CROP))
    # Подотсчёты усредняются (размытие при длинном dwell, усреднение - допустимое упрощение системы)
    # В реальном микроскопе функция может быть другой.
    return sample(x + err / PIXEL_NM, y).mean(-1)  


def visualize(sc: str) -> dict:
    """Четыре рисунка поведения агентов на сценарии sc. Сохраняет их в figures/ и возвращает."""
    # Снимки, траектории ошибки, спектр и профиль искажения для всех стратегий. Задание требует
    # визуализировать поведение агента, числа RMS здесь становятся видимыми. Два режима -
    # быстрая развёртка (где работает обратная связь по изображению) и типичный (где строка
    # 20 мс равна периоду сети). Рисунки возвращаются, чтобы показать их в ноутбуке.
    OUT.mkdir(exist_ok=True)
    figs = {}
    dwells = [(1e-6, "1 мкс/px (быстрая развёртка)"), (10e-6, "10 мкс/px (типичный, кадр 40 с)")]
    runs = {(m, dw): long_rollout(m, sc, dw, CROP) for m in METHODS for dw, _ in dwells}

    # Снимки
    fig, ax = plt.subplots(2, 5, figsize=(17, 8.2))
    i, px = np.meshgrid(np.arange(CROP), COL0 + np.arange(CROP), indexing="ij")
    for r, (dw, title) in enumerate(dwells):
        ax[r, 0].imshow(sample(px, i), cmap="gray", vmin=0, vmax=1)
        ax[r, 0].set_title("идеальный снимок")
        ax[r, 0].set_ylabel(title, fontsize=11)
        for c, m in enumerate(METHODS, start=1):
            e = runs[(m, dw)]
            n = int(CROP * N_PX * dw * FS)
            ax[r, c].imshow(render(e, dw), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_title(f"{LABEL[m]}\nRMS {np.sqrt(np.mean(e[:n] ** 2)):.1f} нм")
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{SC_LABEL[sc]}: фрагмент 256 x 256 px синтетического образца (1 px = 5 нм), "
                 "одна реализация помех")
    fig.tight_layout()
    fig.savefig(OUT / f"{sc}_images.png", dpi=110)
    figs["images"] = fig

    # Траектории ошибки в типичном режиме: весь снятый фрагмент (256 строк, около 5 с)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5))
    e0 = runs[("none", 10e-6)]
    t = np.arange(len(e0)) / FS
    for m in METHODS:
        ax[0].plot(t, runs[(m, 10e-6)], lw=0.6, color=COLOR[m], label=LABEL[m])
    z = slice(0, int(0.1 * FS))
    for m in METHODS:
        ax[1].plot(t[z] * 1e3, runs[(m, 10e-6)][z], lw=1.1, color=COLOR[m])
    ax[0].set(xlabel="время, с", ylabel="e(t), нм", title=f"{SC_LABEL[sc]}, 10 мкс/px: ошибка позиции луча")
    ax[1].set(xlabel="время, мс (пунктир - начало строки, 20 мс)", ylabel="e(t), нм", title="первые 100 мс")
    for x in np.arange(0, 100, 20):
        ax[1].axvline(x, color="k", lw=0.4, ls=":")
    ax[0].legend(ncol=4, loc="upper right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{sc}_traces.png", dpi=110)
    figs["traces"] = fig

    # Спектр остаточной ошибки
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m in METHODS:
        f, pxx = welch(runs[(m, 10e-6)], fs=FS, nperseg=8192)
        ax.semilogy(f, pxx, lw=1, color=COLOR[m], label=LABEL[m])
    for h in (50, 150, 250):
        ax.axvline(h, color="k", lw=0.5, ls=":")
    ax.set(xlim=(0, 400), ylim=(1e-5, 100), xlabel="частота, Гц (пунктир - 50/150/250 Гц)",
           ylabel=r"СПМ ошибки, нм$^2$/Гц",
           title=f"{SC_LABEL[sc]}: спектр остаточной ошибки, 10 мкс/px (Уэлч, около 5 с)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / f"{sc}_psd.png", dpi=110)
    figs["psd"] = fig

    # Статичное искажение вдоль строки и межстрочный разброс
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in METHODS:
        err, _, _ = pixel_error(runs[(m, 10e-6)], 10e-6, np.arange(CROP), np.arange(N_PX), sub=1)
        ax[0].plot(err.mean(0), color=COLOR[m], lw=1.2, label=LABEL[m])
        ax[1].plot(err.std(0), color=COLOR[m], lw=1.0)
    ax[0].set(title="Средняя по строкам ошибка: статичное искажение геометрии", xlabel="положение в строке, px",
              ylabel="нм")
    ax[1].set(title='СКО по строкам: "зубцы" на краях', xlabel="положение в строке, px", ylabel="нм")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{SC_LABEL[sc]}, 10 мкс/px (строка 20 мс = период сети), 256 строк одной реализации помех")
    fig.tight_layout()
    fig.savefig(OUT / f"{sc}_distortion.png", dpi=110)
    figs["distortion"] = fig
    return figs


def main():
    """Точка входа командной строки: python -m sem_beam_gym.visualize --scenario <имя>."""
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIOS), default="building")
    visualize(p.parse_args().scenario)


if __name__ == "__main__":
    main()
