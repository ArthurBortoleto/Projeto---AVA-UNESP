"""Ajuste dos modelos de crescimento por poço e seleção por AIC.

O ajuste é feito em y = ln(OD/OD0) (saída de ``preprocess``), por mínimos quadrados
não lineares (``scipy.optimize.curve_fit``). R² e AIC são calculados nesse mesmo
espaço (y), não em OD.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .io import InputError
from .models import MODELS, N_PARAMS

MIN_POINTS = 6
DEFAULT_MIN_GROWTH = 0.2
DEFAULT_MODELS = ("logistic", "gompertz")

STATUS_OK = "ok"
STATUS_NO_GROWTH = "no_growth"
STATUS_FIT_FAILED = "fit_failed"

_SLOPE_WINDOW = 5      # pontos usados para estimar a inclinação máxima (chute inicial)
_MAX_NFEV = 2000
_NAN = float("nan")


@dataclass(frozen=True)
class FitResult:
    """Resultado do ajuste de um poço (campos numéricos = nan quando não há ajuste)."""

    well: str
    status: str
    model: str | None = None
    mu_max: float = _NAN
    lag_h: float = _NAN
    A: float = _NAN
    K: float = _NAN
    doubling_time_h: float = _NAN
    r2: float = _NAN
    aic: float = _NAN
    reason: str = ""
    n_points: int = 0


# ---------------------------------------------------------------- utilidades

def parse_models(models) -> list:
    """Aceita 'logistic,gompertz' ou uma lista; valida os nomes."""
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    models = list(models)
    if not models:
        raise InputError("nenhum modelo informado")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise InputError(
            f"modelo(s) desconhecido(s): {', '.join(unknown)}; "
            f"disponíveis: {', '.join(sorted(MODELS))}"
        )
    return models


def aic_gaussian(rss: float, n: int, k: int) -> float:
    """AIC com erros gaussianos: n*ln(RSS/n) + 2k (RSS com piso para evitar -inf)."""
    return n * math.log(max(rss, 1e-300) / n) + 2 * k


def initial_guess(t, y):
    """Chute inicial (A, mu, lam) derivado dos dados.

    A   : máximo de y
    mu  : maior inclinação de uma regressão linear em janelas de _SLOPE_WINDOW pontos
    lam : onde a tangente nesse ponto cruza y = 0 (limitado a [0, t_final])
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    w = min(_SLOPE_WINDOW, n)
    best_slope, t_c, y_c = -np.inf, t[0], y[0]
    for i in range(n - w + 1):
        tt, yy = t[i:i + w], y[i:i + w]
        dt = tt - tt.mean()
        slope = float(np.sum(dt * (yy - yy.mean())) / np.sum(dt * dt))
        if slope > best_slope:
            best_slope, t_c, y_c = slope, float(tt.mean()), float(yy.mean())
    mu0 = max(best_slope, 1e-3)
    A0 = max(float(np.max(y)), 1e-3)
    lam0 = float(np.clip(t_c - y_c / mu0, 0.0, t[-1]))
    return A0, mu0, lam0


def _starts(guess):
    """Pontos de partida determinísticos: o chute e duas variações dele."""
    A, mu, lam = guess
    return [(A, mu, lam), (1.2 * A, 0.6 * mu, 0.5 * lam), (0.9 * A, 1.5 * mu, 1.5 * lam + 0.5)]


def _fit_model(name, t, y, guess):
    """Ajusta um modelo; devolve (params, rss) do melhor ponto de partida, ou None."""
    func = MODELS[name]
    ymax = float(np.max(y))
    lower = np.array([0.05 * ymax, 1e-6, 0.0])
    upper = np.array([10.0 * ymax, 50.0, float(t[-1])])
    span = upper - lower
    best = None
    for p0 in _starts(guess):
        p0 = np.clip(p0, lower + 1e-6 * span, upper - 1e-6 * span)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _ = curve_fit(func, t, y, p0=p0, bounds=(lower, upper),
                                      max_nfev=_MAX_NFEV)
        except (RuntimeError, ValueError):
            continue
        if not np.all(np.isfinite(params)):
            continue
        rss = float(np.sum((y - func(t, *params)) ** 2))
        if not math.isfinite(rss):
            continue
        if best is None or rss < best[1]:
            best = (params, rss)
    return best


# ---------------------------------------------------------------- ajuste de um poço

def fit_well(series, models=DEFAULT_MODELS, min_growth: float = DEFAULT_MIN_GROWTH,
             min_points: int = MIN_POINTS) -> FitResult:
    """Ajusta os modelos a um ``WellSeries`` e escolhe o de menor AIC."""
    models = parse_models(models)
    well, n = series.well, series.n_points

    if n < min_points:
        return FitResult(well, STATUS_FIT_FAILED, n_points=n,
                         reason=f"menos de {min_points} pontos válidos ({n})")
    t = np.asarray(series.time_h, dtype=float)
    y = np.asarray(series.y, dtype=float)

    if float(np.max(y)) < min_growth:
        return FitResult(well, STATUS_NO_GROWTH, n_points=n,
                         reason=f"max(y) = {np.max(y):.3f} < min-growth ({min_growth})")

    guess = initial_guess(t, y)
    tss = float(np.sum((y - y.mean()) ** 2))
    candidates = []
    for name in models:
        fitted = _fit_model(name, t, y, guess)
        if fitted is None:
            continue
        params, rss = fitted
        aic = aic_gaussian(rss, n, N_PARAMS[name])
        candidates.append((aic, name, params, rss))

    if not candidates:
        return FitResult(well, STATUS_FIT_FAILED, n_points=n,
                         reason="nenhum modelo convergiu")

    aic, name, (A, mu, lam), rss = min(candidates, key=lambda c: c[0])
    return FitResult(
        well=well, status=STATUS_OK, model=name,
        mu_max=float(mu), lag_h=float(lam), A=float(A),
        K=float(series.od0 * math.exp(A)),
        doubling_time_h=math.log(2) / float(mu),
        r2=float(1.0 - rss / tss) if tss > 0 else _NAN,
        aic=float(aic), n_points=n,
    )


def fit_all(series_by_well, models=DEFAULT_MODELS, min_growth: float = DEFAULT_MIN_GROWTH,
            min_points: int = MIN_POINTS) -> list:
    """Ajusta todos os poços; um poço problemático nunca interrompe os demais."""
    models = parse_models(models)
    return [fit_well(s, models, min_growth, min_points) for s in series_by_well.values()]