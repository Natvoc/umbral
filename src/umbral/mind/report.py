"""Learning curves of the predictive mind and the order in which each sense is learned.

The curve of a channel is ``model MSE / persistence MSE`` over training updates
(log scale; below 1 means better than "the next step equals this one"). The
order of learning is read from the efference model's curve: in log space,
progress(u) = (L_0 - L(u)) / (L_0 - L_final), with L_final the mean of the last 3
logged points; we report the first update reaching 50 % and 90 % of the progress.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from umbral.mind.model import CHANNELS  # noqa: E402

COLORS = {"proprio": "#3b6fb6", "touch": "#d43d3d", "vision": "#3d9a50", "intero": "#c08a2b"}
NAMES_ES = {"proprio": "propiocepción", "touch": "tacto", "vision": "visión", "intero": "interocepción"}


def read_curves(metrics_path: str | Path, prefix: str = "", kind: str = "mse"
                ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """{predictor: {name: (updates, value)}} for metrics ``<prefix><predictor>/<kind>_<name>``."""
    series: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    tag = f"/{kind}_"
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        name = m["name"]
        if not name.startswith(prefix) or tag not in name:
            continue
        if not prefix and name.startswith("ckpt_eval/"):
            continue
        pred, ch = name[len(prefix):].split(tag)
        series[pred][ch].append((m["step"], m["value"]))
    return {p: {c: (np.array([s for s, _ in v]), np.array([x for _, x in v])) for c, v in d.items()}
            for p, d in series.items()}


def learning_order(curves: dict, predictor: str = "efference") -> dict[str, dict[str, int | None]]:
    out = {}
    for ch in CHANNELS:
        steps, mse = curves[predictor][ch]
        _, base = curves["persistence"][ch]
        logr = np.log(mse / base)
        l0, lf = logr[0], logr[-3:].mean()
        res: dict[str, int | None] = {}
        for frac in (0.5, 0.9):
            if l0 - lf <= 0:
                res[f"update_{int(frac * 100)}"] = None
                continue
            prog = (l0 - logr) / (l0 - lf)
            hit = np.flatnonzero(prog >= frac)
            res[f"update_{int(frac * 100)}"] = int(steps[hit[0]]) if len(hit) else None
        out[ch] = res
    return out


def plot_learning_curves(metrics_path: str | Path, out_path: str | Path) -> dict[str, dict[str, int | None]]:
    curves = read_curves(metrics_path)
    order = learning_order(curves)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=110)
    for ch in CHANNELS:
        steps, base = curves["persistence"][ch]
        for pred, style in (("efference", "-"), ("no_efference", "--")):
            s, mse = curves[pred][ch]
            label = f"{NAMES_ES[ch]}" if pred == "efference" else None
            ax.plot(s, mse / base, style, color=COLORS[ch], lw=1.8 if pred == "efference" else 1.1, label=label)
        u90 = order[ch]["update_90"]
        if u90 is not None:
            ax.axvline(u90, color=COLORS[ch], lw=0.6, alpha=0.5)
    ax.axhline(1.0, color="#555", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("actualización de entrenamiento")
    ax.set_ylabel("error / error de persistencia (log)")
    ax.set_title("Error de predicción por canal (línea continua: con copia eferente;\n"
                 "discontinua: sin copia eferente; verticales: 90 % de la mejora)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return order


def plot_checkpoint_eval(metrics_path: str | Path, out_path: str | Path) -> bool:
    """Curves from the light evaluation run at every checkpoint (same seeds each time).

    Left: per-channel error relative to persistence. Right: touch onset error of the
    efference model relative to the no-efference model (self and external contact).
    Returns False if the run has no checkpoint evaluations.
    """
    mse = read_curves(metrics_path, prefix="ckpt_eval/", kind="mse")
    touch = read_curves(metrics_path, prefix="ckpt_eval/", kind="touch")
    if "efference" not in mse:
        return False
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=110)
    for ch in CHANNELS:
        s, base = mse["persistence"][ch]
        for pred, style in (("efference", "-o"), ("no_efference", "--")):
            _, v = mse[pred][ch]
            ax.plot(s, v / base, style, color=COLORS[ch], ms=3, lw=1.6 if pred == "efference" else 1.0,
                    label=NAMES_ES[ch] if pred == "efference" else None)
    ax.axhline(1.0, color="#555", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("actualización")
    ax.set_ylabel("error / persistencia (log)")
    ax.set_title("Evaluación en cada checkpoint (continua: con copia eferente)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    for ev, color in (("onset_self", "#7b3fb6"), ("onset_external", "#888888"), ("onset", "#222222")):
        if ev in touch.get("efference", {}) and ev in touch.get("no_efference", {}):
            s, e = touch["efference"][ev]
            s2, n = touch["no_efference"][ev]
            k = min(len(s), len(s2))
            bx.plot(s[:k], e[:k] / n[:k], "-o", ms=3, color=color, label=ev.replace("_", " "))
    bx.axhline(1.0, color="#555", lw=0.8)
    bx.set_xlabel("actualización")
    bx.set_ylabel("error con copia / sin copia")
    bx.set_title("Tacto: inicio de contacto (< 1: la copia eferente ayuda)", fontsize=9)
    bx.legend(fontsize=8, frameon=False)
    bx.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True
