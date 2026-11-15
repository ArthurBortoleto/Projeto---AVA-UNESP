"""Gráficos por poço: dados observados e curva ajustada, com os parâmetros anotados.

Usa a API orientada a objetos do matplotlib (``Figure``), sem ``pyplot``: não depende
de interface gráfica, não guarda figuras abertas em memória e funciona em qualquer máquina.
Os gráficos ficam no espaço do ajuste, y = ln(OD/OD0).
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .models import MODELS

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_DPI = 120


# ---------------------------------------------------------------- textos

def plot_filename(well: str) -> str:
    """Nome de arquivo seguro para o poço (caracteres fora de A-Z a-z 0-9 . _ - viram '_')."""
    return _UNSAFE.sub("_", str(well)) + ".png"


def title_text(result) -> str:
    if result.status == "ok":
        title = f"{result.well} — ajuste {result.model}"
    else:
        title = f"{result.well} — {result.status}: {result.reason}"
    return textwrap.fill(title, 70)


def annotation_text(result) -> str:
    """Parâmetros anotados no gráfico (vazio quando o poço não foi ajustado)."""
    if result.status != "ok":
        return ""
    return "\n".join([
        f"modelo: {result.model}",
        f"μ = {result.mu_max:.3f} h⁻¹",
        f"λ = {result.lag_h:.2f} h",
        f"A = {result.A:.3f}",
        f"K = {result.K:.3f} (OD)",
        f"t_d = {result.doubling_time_h:.2f} h",
        f"R² = {result.r2:.4f}",
        f"AIC = {result.aic:.1f}",
    ])


# ---------------------------------------------------------------- figuras

def build_figure(series, result) -> Figure:
    """Monta a figura de um poço (``series`` = WellSeries, ``result`` = FitResult)."""
    fig = Figure(figsize=(6.4, 4.4), layout="constrained")
    ax = fig.subplots()
    t = np.asarray(series.time_h, dtype=float)
    y = np.asarray(series.y, dtype=float)

    if series.n_points:
        ax.plot(t, y, "o", markersize=4, label="observado")
    else:
        ax.text(0.5, 0.5, "sem pontos válidos", transform=ax.transAxes,
                ha="center", va="center", color="gray")

    if result.status == "ok" and series.n_points:
        dense = np.linspace(t.min(), t.max(), 300)
        curve = MODELS[result.model](dense, result.A, result.mu_max, result.lag_h)
        ax.plot(dense, curve, "-", linewidth=2, label=f"ajuste ({result.model})")
        ax.text(0.98, 0.04, annotation_text(result), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85,
                      "edgecolor": "lightgray"})
    if series.n_points:
        ax.legend(loc="upper left")

    ax.set_title(title_text(result), fontsize=10)
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("ln(OD/OD0)")
    ax.grid(alpha=0.3)
    return fig


def plot_well(series, result, path) -> Path:
    """Grava o gráfico de um poço em PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_figure(series, result).savefig(path, dpi=_DPI)
    return path


def plot_all(series_by_well, results, plots_dir) -> list:
    """Um PNG por poço analisado, em ``plots_dir/<poço>.png``.

    Poços sem ajuste (no_growth, fit_failed) também ganham gráfico, só com os dados
    observados e o motivo no título. Se dois poços gerarem o mesmo nome de arquivo
    depois da limpeza dos caracteres, o segundo ganha o sufixo _2, _3, ...
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    paths = []
    for result in results:
        stem = plot_filename(result.well)[:-4]
        name, k = stem, 1
        while name in used:
            k += 1
            name = f"{stem}_{k}"
        used.add(name)
        paths.append(plot_well(series_by_well[result.well], result, plots_dir / f"{name}.png"))
    return paths