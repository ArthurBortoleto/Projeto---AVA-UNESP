"""Pré-processamento: correção de branco, suavização e transformação y = ln(OD/OD0).

Ordem (como no spec): correção de branco -> suavização por mediana móvel ->
descarte de OD <= 0 -> OD0 -> y = ln(OD/OD0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import InputError

# O spec diz "média dos primeiros pontos" sem dar o número; fixado aqui em 3.
N_BASELINE = 3


@dataclass(frozen=True)
class WellSeries:
    """Série de um poço já pré-processada (só os pontos mantidos).

    time_h, od, y        : arrays alinhados (od = OD corrigida e suavizada)
    od0                  : média dos primeiros N_BASELINE pontos mantidos (nan se não há pontos)
    n_nonpositive        : pontos descartados por OD <= 0
    n_no_blank           : pontos descartados por não haver branco no mesmo instante
    """

    well: str
    time_h: np.ndarray
    od: np.ndarray
    y: np.ndarray
    od0: float
    n_nonpositive: int = 0
    n_no_blank: int = 0

    @property
    def n_points(self) -> int:
        return int(self.time_h.size)

    @property
    def n_discarded(self) -> int:
        return self.n_nonpositive + self.n_no_blank


@dataclass
class Preprocessed:
    series: dict                      # {poço: WellSeries}, só poços que não são branco
    blank_wells: list                 # poços-branco (excluídos de series)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- suavização

def _check_window(window) -> None:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) \
            or window < 1 or window % 2 == 0:
        raise InputError(
            f"janela de suavização inválida ({window!r}): use um inteiro ímpar "
            "positivo (1 desativa a suavização)"
        )


def moving_median(values, window: int) -> np.ndarray:
    """Mediana móvel centrada. Nas bordas a janela encolhe (usa só os pontos existentes)."""
    _check_window(window)
    values = np.asarray(values, dtype=float)
    if window == 1 or values.size == 0:
        return values.copy()
    return (
        pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    )


# ---------------------------------------------------------------- correção de branco

def blank_correct(long: pd.DataFrame, blank_wells):
    """Subtrai, de cada medida, a média dos brancos no mesmo instante.

    Devolve (tabela longa só com poços que não são branco, {poço: pontos sem branco
    descartados}, avisos). Sem poços-branco (ou sem nenhuma medida deles), a OD não é
    corrigida e um aviso é registrado.
    """
    warnings = []
    blank_wells = list(blank_wells)
    is_blank = long["well"].isin(blank_wells)
    data = long.loc[~is_blank].copy()
    blank_data = long.loc[is_blank]

    if blank_data.empty:
        if blank_wells:
            warnings.append(
                "poços-branco sem nenhuma medida válida: OD não corrigida; o início de "
                "cada poço serve de linha de base (OD0)"
            )
        else:
            warnings.append(
                "nenhum poço-branco nos metadados: OD não corrigida; o início de cada "
                "poço serve de linha de base (OD0)"
            )
        return data.reset_index(drop=True), {}, warnings

    blank_mean = blank_data.groupby("time_h")["od"].mean()
    data["od"] = data["od"] - data["time_h"].map(blank_mean)

    no_blank = data["od"].isna()
    dropped = {}
    if no_blank.any():
        dropped = data.loc[no_blank].groupby("well").size().to_dict()
        warnings.append(
            f"{int(no_blank.sum())} ponto(s) sem medida de branco no mesmo instante "
            "foram descartados"
        )
        data = data.loc[~no_blank]
    return data.reset_index(drop=True), dropped, warnings


# ---------------------------------------------------------------- transformação

def log_transform(well, time_h, od, n_no_blank=0, n_baseline: int = N_BASELINE) -> WellSeries:
    """Descarta OD <= 0, calcula OD0 e y = ln(OD/OD0)."""
    time_h = np.asarray(time_h, dtype=float)
    od = np.asarray(od, dtype=float)
    keep = od > 0
    n_nonpositive = int((~keep).sum())
    time_h, od = time_h[keep], od[keep]
    if od.size == 0:
        od0 = float("nan")
        y = np.array([])
    else:
        od0 = float(od[:n_baseline].mean())
        y = np.log(od / od0)
    return WellSeries(well, time_h, od, y, od0, n_nonpositive, int(n_no_blank))


# ---------------------------------------------------------------- pipeline

def preprocess(long: pd.DataFrame, metadata: pd.DataFrame, smooth_window: int = 3,
               n_baseline: int = N_BASELINE) -> Preprocessed:
    """Pipeline completo.

    long     : tabela longa (well, time_h, od) de ``io.read_plate``
    metadata : metadados alinhados aos poços (``io.align_metadata``), uma linha por poço
    """
    _check_window(smooth_window)
    blank_wells = list(metadata.loc[metadata["is_blank"], "well"])
    corrected, dropped_no_blank, warnings = blank_correct(long, blank_wells)

    series = {}
    for well in metadata.loc[~metadata["is_blank"], "well"]:
        g = corrected.loc[corrected["well"] == well].sort_values("time_h")
        od = moving_median(g["od"].to_numpy(), smooth_window)
        series[well] = log_transform(
            well, g["time_h"].to_numpy(), od,
            n_no_blank=dropped_no_blank.get(well, 0), n_baseline=n_baseline,
        )
    return Preprocessed(series=series, blank_wells=blank_wells, warnings=warnings)