"""Графики обучения и сравнения, сводные таблицы.

    python -m sem_beam_gym.plots   # читает runs/ и results/eval.csv -> figures/*.png, results/summary.md

Усреднение для RL: метрика усредняется по тестовым эпизодам для каждого seed обучения,
затем показывается среднее +- СКО по трём seed. Для none/linear (нет seed обучения)
полоса - +-1.96 * SE по эпизодам. В таблицах дополнительно IQM (interquartile mean, среднее
средних 50 % значений) по всем эпизодам всех seed - устойчиво к редким сильным событиям.

Подписи на рисунках и в summary.md используют LaTeX ($\\pm$, $\\sqrt{...}$).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from sem_beam_gym.paths import FIGURES, RESULTS, RUNS
from sem_beam_gym.scenarios import SCENARIOS, TRAIN_SCENARIOS

OUT = FIGURES
LABEL = {"none": "без коррекции", "linear": "линейный фильтр + датчики",
         "rl_nosens": "RL без датчиков", "rl_sens": "RL + датчики"}
COLOR = {"none": "#888888", "linear": "#d62728", "rl_nosens": "#1f77b4", "rl_sens": "#2ca02c"}
COND = {"nosens": "rl_nosens", "sens": "rl_sens"}
SC_LABEL = {"quiet": "тихо (фон)", "pump": "форвакуумный насос", "elevators": "2 лифта",
            "footsteps": "шаги в коридоре", "building": "рабочий день (всё)",
            "randomized": "domain randomization"}
NAMES = list(SCENARIOS)  # сценарии теста
TRAIN = list(TRAIN_SCENARIOS)  # сценарии обучения
PM = r"$\pm$"  # знак "плюс-минус" в подписях и таблицах
ACC_UNIT = r"мкg/$\sqrt{\mathrm{Гц}}$"  # единица шума акселерометра


def iqm(x) -> float:
    """Interquartile mean: среднее значений между 25-м и 75-м перцентилями."""
    # scipy.stats.trim_mean отрезает по 25 % с каждой стороны.
    return float(trim_mean(np.asarray(x), 0.25))


def agg(g: pd.DataFrame, metric: str) -> tuple[float, float]:
    """(среднее, разброс): для RL - СКО по seed, иначе +-1.96 * SE по эпизодам."""
    # Для RL главный источник разброса - seed обучения (задание требует показать его), поэтому
    # сначала среднее по эпизодам внутри seed, затем СКО по seed. У бейзлайнов seed обучения нет,
    # и разброс - это 95 % доверительный интервал среднего по эпизодам.
    if g.method.iloc[0].startswith("rl_"):
        per_seed = g.groupby("seed")[metric].mean()
        return per_seed.mean(), per_seed.std(ddof=1)
    return g[metric].mean(), 1.96 * g[metric].std(ddof=1) / np.sqrt(len(g))


def diag(df: pd.DataFrame) -> pd.DataFrame:
    """Строки, где метод проверяется на своём сценарии обучения (none - на всех)."""
    # "Диагональ" матрицы переноса: основное сравнение без эффекта переноса между сценариями.
    return df[(df.suite == "matrix") & ((df.train == df.test) | (df.method == "none"))]


def curve(ax, df, x, metric, xscale=1.0):
    """Метрика по оси x для всех методов: линия - среднее, полоса - разброс из agg()."""
    # Общая функция для графиков по dwell и по шуму датчика. Ось x логарифмическая,
    # т.к. оба параметра меняются на порядки.
    for m in LABEL:
        d = df[df.method == m]
        if d.empty:
            continue
        pts = sorted((xv, *agg(g, metric)) for xv, g in d.groupby(x))
        xs, mean, err = (np.array(v) for v in zip(*pts))
        ax.plot(xs * xscale, mean, "o-", ms=4, color=COLOR[m], label=LABEL[m])
        ax.fill_between(xs * xscale, mean - err, mean + err, color=COLOR[m], alpha=0.2)
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")


def train_returns(run: Path, smooth: int = 20) -> pd.Series:
    """Награда за обучающий эпизод по логам Monitor.

    Каждая из N_ENVS сред пишет свой monitor.csv. Среды шагают синхронно, и эпизоды одной
    длины, поэтому i-й эпизод всех сред заканчивается на шаге (i + 1) * длина * N_ENVS.
    Награды усредняются по средам и сглаживаются скользящим средним по `smooth` точкам.
    """
    # Сглаживание нужно потому, что dwell и помехи в каждом эпизоде случайны и награда отдельного
    # эпизода сильно шумит. Файлы обрезаются до общей длины, чтобы среды можно было усреднить.
    logs = [pd.read_csv(f, skiprows=1) for f in sorted((run / "monitor").glob("*.monitor.csv"))]
    n = min(map(len, logs))
    r = np.mean([lg.r.to_numpy()[:n] for lg in logs], axis=0)
    steps = (np.arange(n) + 1) * logs[0].l.iloc[0] * len(logs)
    return pd.Series(r, index=steps).rolling(smooth, min_periods=1).mean()


def band(ax, v: pd.DataFrame, color, label):
    """Среднее +- СКО по столбцам (seed)."""
    # Столбцы - seed, строки - шаги. Так видно и динамику, и разброс между запусками.
    ax.plot(v.index, v.mean(axis=1), color=color, label=label)
    ax.fill_between(v.index, v.mean(axis=1) - v.std(axis=1), v.mean(axis=1) + v.std(axis=1),
                    color=color, alpha=0.2)


def learning_curves():
    """Кривые обучения всех сценариев: награда (сверху) и валидационная RMS (снизу)."""
    # Две строки, потому что они отвечают на разные вопросы: награда - как идёт обучение
    # стохастической политики, RMS на фиксированных эпизодах - качество жадной
    # политики, которая потом оценивается. По одной колонке на сценарий обучения.
    fig, axes = plt.subplots(2, len(TRAIN), figsize=(3.6 * len(TRAIN), 7), sharex=True)
    for col, sc in enumerate(TRAIN):
        top, bottom = axes[0, col], axes[1, col]
        for cond, m in COND.items():
            runs = sorted((RUNS / sc).glob(f"{cond}_s*"))
            if not runs:
                continue
            label = f"{LABEL[m]} (n={len(runs)})"
            band(top, pd.concat([train_returns(r) for r in runs], axis=1), COLOR[m], label)
            band(bottom, pd.concat([pd.read_csv(r / "val.csv").set_index("step").val_rms for r in runs],
                                   axis=1), COLOR[m], label)
        top.set_title(SC_LABEL[sc])
        # у награды большой разброс от -6000 (случайная политика) до -20
        # без лог-шкалы финальная разница не видна
        top.set_yscale("symlog", linthresh=10)
        top.set_ylim(top=-5)  # награда не бывает положительной
        bottom.set_xlabel("шаги взаимодействия")
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("награда за эпизод (512 шагов), symlog")
    axes[1, 0].set_ylabel("RMS истинной ошибки, нм")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Кривые обучения PPO, среднее {PM} СКО по seed.\n"
                 "Сверху: награда обучающих эпизодов (стохастическая политика, среднее по 8 средам, "
                 "скользящее среднее по 20 точкам). Снизу: валидация жадной политикой на 16 фиксированных "
                 "эпизодах.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=110)
    return fig


def dwell_by_scenario(df):
    """RMS в зависимости от dwell для всех методов, по сценариям (обучение = тест)."""
    # Главный рисунок для H1: видно, где агент без датчиков перестаёт работать.
    d = diag(df)
    fig, axes = plt.subplots(1, len(NAMES), figsize=(4 * len(NAMES), 4), sharey=True)
    for ax, sc in zip(axes, NAMES):
        curve(ax, d[d.test == sc], "dwell", "rms_true", 1e6)
        ax.axvline(10, color="k", ls=":", lw=0.8)
        ax.set(title=SC_LABEL[sc], xlabel="время на пиксель, мкс")
    axes[0].set_ylabel("RMS истинной ошибки, нм")
    axes[0].legend(fontsize=8)
    fig.suptitle("Обучение и тест на одном сценарии (пунктир - типичный режим 10 мкс/px)")
    fig.tight_layout()
    fig.savefig(OUT / "dwell_by_scenario.png", dpi=110)
    return fig


def matrix_table(df, method, dwell):
    """Таблица "сценарий обучения x сценарий теста" средней RMS для метода и dwell."""
    # reindex фиксирует порядок строк и столбцов, чтобы все матрицы были сопоставимы.
    d = df[(df.suite == "matrix") & (df.method == method) & np.isclose(df.dwell, dwell)]
    return d.groupby(["train", "test"]).rms_true.mean().unstack().reindex(index=TRAIN, columns=NAMES)


def matrices(df):
    """Тепловые карты переноса для трёх методов при 1 и 10 мкс/px."""
    # Для H3 и H5. Общая шкала цвета в строке, чтобы разница была видна визуально. 
    # Под подписью сценария теста - уровень без коррекции как ориентир. 
    # Диагональ выделена жирным.
    dwells = [(1e-6, "1 мкс/px"), (10e-6, "10 мкс/px")]
    methods = ["rl_sens", "rl_nosens", "linear"]
    fig, axes = plt.subplots(len(dwells), len(methods), figsize=(15, 10.5))
    for r, (dw, dlabel) in enumerate(dwells):
        none = df[(df.method == "none") & np.isclose(df.dwell, dw)].groupby("test").rms_true.mean()
        tabs = [matrix_table(df, m, dw) for m in methods]
        vmax = max(np.nanmax(t.values) for t in tabs)
        for c, (m, t) in enumerate(zip(methods, tabs)):
            ax = axes[r, c]
            im = ax.imshow(t.values, cmap="viridis_r", vmin=0, vmax=vmax)
            for i in range(len(TRAIN)):
                for j in range(len(NAMES)):
                    ax.text(j, i, f"{t.values[i, j]:.1f}", ha="center", va="center",
                            color="w" if t.values[i, j] > vmax / 2 else "k", fontsize=9,
                            fontweight="bold" if TRAIN[i] == NAMES[j] else "normal")
            ax.set_xticks(range(len(NAMES)), [f"{SC_LABEL[n]}\n(без корр. {none[n]:.1f})" for n in NAMES],
                          rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(TRAIN)), [SC_LABEL[n] for n in TRAIN], fontsize=8)
            ax.set_title(f"{LABEL[m]}, {dlabel}")
            if c == 0:
                ax.set_ylabel("сценарий обучения")
            ax.set_xlabel("сценарий теста")
            fig.colorbar(im, ax=ax, shrink=0.75, label="RMS, нм")
    fig.suptitle("Перенос между гипотезами о вибрации: RMS истинной ошибки, нм (диагональ - свой сценарий)")
    fig.tight_layout()
    fig.savefig(OUT / "matrix.png", dpi=110)
    return fig


def sensor_gain(df):
    """Во сколько раз датчики снижают ошибку (RL без датчиков / RL + датчики), свой сценарий."""
    # Отношение, а не разность: сценарии различаются уровнем помех, а отношение сравнимо между
    # ними. Разброс считается по всем 3 x 3 парам seed двух условий.
    d = diag(df)
    groups = [(1e-6, "1 мкс/px"), (10e-6, "10 мкс/px"), (500e-6, "500 мкс/px")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(groups)
    for k, (dw, label) in enumerate(groups):
        ratios, errs = [], []
        for sc in NAMES:
            g = d[(d.test == sc) & np.isclose(d.dwell, dw)]
            ns = g[g.method == "rl_nosens"].groupby("seed").rms_true.mean()
            s = g[g.method == "rl_sens"].groupby("seed").rms_true.mean()
            r = ns.values[:, None] / s.values[None, :]  # все пары seed
            ratios.append(r.mean())
            errs.append(r.std())
        ax.bar(np.arange(len(NAMES)) + (k - 1) * w, ratios, w, yerr=errs, capsize=3, label=label)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(len(NAMES)), [SC_LABEL[n] for n in NAMES])
    ax.set(ylabel="RMS без датчиков / RMS с датчиками",
           title="Выигрыш от датчиков (RL, обучение и тест на одном сценарии)")
    ax.legend(title="время на пиксель")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "sensor_gain.png", dpi=110)
    return fig


def acc_noise(df):
    """Требование к акселерометру: RMS в зависимости от его шума, рабочий день."""
    # Сплошные линии - агенты сценария "рабочий день" (обучены при 1 мкg/sqrt(Гц)), пунктир - DR-агенты,
    # которые видели шум 0.3-3 при обучении (серая полоса). Демонстрирует, помогает ли DR устойчивости.
    d = df[df.suite == "acc_noise"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, dw in zip(axes, (1e-6, 10e-6)):
        g = d[np.isclose(d.dwell, dw)]
        curve(ax, g[g.train.isin(["building", "-"])], "param", "rms_true")
        for m in ("rl_sens", "linear"):  # обученные с domain randomization (шум 0.3-3 мкg/sqrt(Гц))
            r = g[(g.train == "randomized") & (g.method == m)]
            pts = sorted((p, *agg(gp, "rms_true")) for p, gp in r.groupby("param"))
            if pts:
                xs, mean, _ = (np.array(v) for v in zip(*pts))
                ax.plot(xs, mean, "s--", ms=4, color=COLOR[m], label=f"{LABEL[m]} [DR]")
        ax.axvspan(0.3, 3.0, color="k", alpha=0.05)
        ax.set(title=f"тест: {SC_LABEL['building']}, {dw * 1e6:g} мкс/px",
               xlabel=f"шум акселерометра, {ACC_UNIT}")
    axes[0].set_ylabel("RMS истинной ошибки, нм")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Сплошные - обучение при 1 {ACC_UNIT}, пунктир - domain randomization, "
                 "серая полоса - диапазон DR", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "acc_noise.png", dpi=110)
    return fig


def events(df):
    """Ошибка эпизода в зависимости от силы вибрации в нём (рабочий день, все dwell >= 3 мкс)."""
    # Средние скрывают редкие сильные события (лифт, шаги у двери). Точечный график по эпизодам
    # показывает, как методы ведут себя в хвосте. dwell < 3 мкс исключены: там иной режим работы.
    d = diag(df)
    d = d[(d.test == "building") & (d.dwell >= 3e-6)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in LABEL:
        g = d[d.method == m]
        ax.scatter(g.rms_vib, g.rms_true, s=6, alpha=0.4, color=COLOR[m], label=LABEL[m])
    ax.set(xlabel="RMS вибрационной помехи в эпизоде, нм", ylabel="RMS истинной ошибки, нм",
           title=r"Рабочий день: ошибка и сила вибрации (эпизоды 0.26 с, dwell $\geq$ 3 мкс)")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "events.png", dpi=110)
    return fig


def dr_vs_specialist(df):
    """Агент с domain randomization против специалистов и агентов чужих сценариев."""
    # Для H5: на каждом тестовом сценарии три столбца - свой специалист, DR-агент и худший из
    # агентов, обученных на других фиксированных сценариях (то, чем рискуем при неверной гипотезе).
    m = df[(df.suite == "matrix") & df.method.str.startswith("rl_")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
    for r, method in enumerate(("rl_sens", "rl_nosens")):
        for c, dw in enumerate((1e-6, 10e-6)):
            ax = axes[r, c]
            d = m[(m.method == method) & np.isclose(m.dwell, dw)]
            x = np.arange(len(NAMES))
            variants = [("свой сценарий (специалист)", lambda t: d[(d.train == t) & (d.test == t)]),
                        ("domain randomization", lambda t: d[(d.train == "randomized") & (d.test == t)]),
                        ("худший чужой специалист", None)]
            w = 0.27
            for k, (label, sel) in enumerate(variants):
                means, errs = [], []
                for t in NAMES:
                    if sel is None:  # худший из агентов, обученных на других фиксированных сценариях
                        per = [d[(d.train == o) & (d.test == t)].groupby("seed").rms_true.mean()
                               for o in NAMES if o != t]
                        g = max(per, key=lambda v: v.mean())
                    else:
                        g = sel(t).groupby("seed").rms_true.mean()
                    means.append(g.mean())
                    errs.append(g.std(ddof=1))
                ax.bar(x + (k - 1) * w, means, w, yerr=errs, capsize=3, label=label)
            none = df[(df.method == "none") & np.isclose(df.dwell, dw)].groupby("test").rms_true.mean()
            ax.scatter(x, [none[t] for t in NAMES], marker="_", s=600, color="k", label="без коррекции", zorder=3)
            ax.set_xticks(x, [SC_LABEL[t] for t in NAMES], fontsize=8)
            ax.set_title(f"{LABEL[method]}, {dw * 1e6:g} мкс/px")
            ax.grid(alpha=0.3, axis="y")
        axes[r, 0].set_ylabel("RMS истинной ошибки, нм")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Domain randomization против обучения на одной гипотезе о вибрации (среднее {PM} СКО по seed)")
    fig.tight_layout()
    fig.savefig(OUT / "dr_vs_specialist.png", dpi=110)
    return fig


def cell(g: pd.DataFrame, metric: str = "rms_true") -> str:
    """Ячейка таблицы: "среднее +- разброс (IQM)"."""
    # Среднее и разброс - для сравнения условий, IQM - чтобы видеть влияние редких сильных эпизодов.
    mean, err = agg(g, metric)
    return f"{mean:.2f} {PM} {err:.2f} ({iqm(g[metric]):.2f})"


def fmt_table(rows: dict, index_name: str) -> str:
    """Словарь строк {имя: {столбец: ячейка}} -> таблица Markdown."""
    t = pd.DataFrame(rows).T
    t.index.name = index_name
    return t.to_markdown()


def write_summary(df):
    """Все итоговые таблицы в results/summary.md."""
    # Таблицы по сценариям, дополнительные метрики, матрицы переноса, свип шума, разброс по
    # seed, DR - числа для README и проверки гипотез в одном файле, без ручного переписывания.
    # Markdown читается и в редакторе, и на GitHub.
    lines = ["# Сводка результатов (генерируется plots.py)", "",
             f"RMS истинной ошибки, нм. Формат ячейки: среднее {PM} разброс (IQM). Разброс для RL - СКО по seed,",
             r"для остальных - $\pm 1.96 \cdot SE$ по эпизодам. IQM - по всем эпизодам всех seed.", ""]
    d = diag(df)
    for sc in NAMES:
        rows = {}
        for dw, g in d[d.test == sc].groupby("dwell"):
            rows[f"{dw * 1e6:g}"] = {LABEL[m]: cell(gm) for m in LABEL if not (gm := g[g.method == m]).empty}
        lines += [f"## {SC_LABEL[sc]} - обучение и тест на этом сценарии", "", fmt_table(rows, "dwell, мкс"), ""]
    # дополнительные метрики (раздел 1 README). Межстрочный сдвиг определён, только если
    # предыдущая строка снята в том же эпизоде, т. е. при dwell <= 100 мкс
    d = d.assign(frac_pct=100 * d.frac_gt_1px)
    for metric, title in (("rms_jitter", "Видимый межстрочный сдвиг (RMS), нм"),
                          ("frac_pct", "Доля времени с ошибкой > 1 px (5 нм), %")):
        rows = {}
        for sc in NAMES:
            for dw in (1e-6, 10e-6, 100e-6):
                g = d[(d.test == sc) & np.isclose(d.dwell, dw)]
                rows[f"{SC_LABEL[sc]}, {dw * 1e6:g} мкс"] = {
                    LABEL[m]: cell(gm, metric) for m in LABEL if not (gm := g[g.method == m]).empty}
        lines += [f"## {title} - обучение и тест на одном сценарии", "", fmt_table(rows, "тест"), ""]
    d = diag(df)
    for m in ("rl_sens", "rl_nosens", "linear"):
        for dw in (1e-6, 10e-6):
            t = matrix_table(df, m, dw).round(2)
            t.index = [SC_LABEL[i] for i in t.index]
            t.columns = [SC_LABEL[c] for c in t.columns]
            t.index.name = "обучение \\ тест"
            lines += [f"## Матрица переноса: {LABEL[m]}, {dw * 1e6:g} мкс/px", "", t.to_markdown(), ""]
    a = df[df.suite == "acc_noise"]
    for dw in (1e-6, 10e-6):
        rows = {}
        for p, g in a[np.isclose(a.dwell, dw)].groupby("param"):
            rows[f"{p:g}"] = {f"{LABEL[m]} [{SC_LABEL[t] if t != '-' else '-'}]": cell(gm)
                              for (m, t), gm in g.groupby(["method", "train"])}
        lines += [f"## Шум акселерометра (тест: {SC_LABEL['building']}), {dw * 1e6:g} мкс/px",
                  "", "В скобках - сценарий обучения.", "", fmt_table(rows, ACC_UNIT), ""]
    t = d[d.method.str.startswith("rl_")].groupby(["test", "method", "seed"]).rms_true.mean().unstack().round(2)
    lines += ["## RMS по seed обучения (свой сценарий, среднее по всем dwell), нм", "", t.to_markdown(), ""]
    r = df[(df.suite == "matrix") & (df.train == "randomized")]
    rows = {}
    for (t, dw), g in r.groupby(["test", "dwell"]):
        if np.isclose(dw, 1e-6) or np.isclose(dw, 10e-6):
            rows[f"{SC_LABEL[t]}, {dw * 1e6:g} мкс"] = {LABEL[m]: cell(gm) for m, gm in g.groupby("method")}
    lines += ["## Обучение с domain randomization, тест на фиксированных сценариях", "",
              fmt_table(rows, "тест"), ""]
    (RESULTS / "summary.md").write_text("\n".join(lines))


def main():
    """Построить все рисунки и сводку по runs/ и results/eval.csv."""
    OUT.mkdir(exist_ok=True)
    learning_curves()
    df = pd.read_csv(RESULTS / "eval.csv")
    dwell_by_scenario(df)
    matrices(df)
    sensor_gain(df)
    dr_vs_specialist(df)
    acc_noise(df)
    events(df)
    write_summary(df)


if __name__ == "__main__":
    main()
