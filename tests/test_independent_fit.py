"""Conferência do ajuste contra um ajuste independente.

O ajuste do projeto usa mínimos quadrados locais (curve_fit, a partir de chutes derivados
dos dados). Aqui cada curva é ajustada de novo por um método sem relação com ele:
evolução diferencial (busca global, sem chute inicial, só com os limites dos parâmetros).
Se os dois métodos chegam aos mesmos parâmetros e à mesma soma de quadrados, o ajuste
do projeto encontrou o mínimo global e não um mínimo local.
"""

import numpy as np
import pytest
from scipy.optimize import differential_evolution

from growthcurves.fit import fit_well
from growthcurves.models import MODELS
from growthcurves.preprocess import preprocess
from growthcurves.synthetic import example_plate


@pytest.fixture(scope="module")
def example_series():
    plate, meta = example_plate()
    long = (plate.melt(id_vars="time_h", var_name="well", value_name="od")
            .dropna(subset=["od"]).reset_index(drop=True)[["well", "time_h", "od"]])
    return preprocess(long, meta).series


def independent_fit(model, t, y):
    """Mesmo problema (mesma função, mesmos limites), resolvido por evolução diferencial."""
    func = MODELS[model]
    ymax = float(np.max(y))
    bounds = [(0.05 * ymax, 10.0 * ymax), (1e-6, 50.0), (0.0, float(t[-1]))]

    def rss(params):
        return float(np.sum((y - func(t, *params)) ** 2))

    result = differential_evolution(rss, bounds, seed=0, tol=1e-12, maxiter=3000,
                                    popsize=30, polish=True)
    return result.x, result.fun


@pytest.mark.parametrize("well, model", [("A2", "logistic"), ("C2", "gompertz"),
                                         ("B1", "logistic"), ("D2", "gompertz")])
def test_project_fit_matches_independent_global_fit(example_series, well, model):
    series = example_series[well]
    t, y = series.time_h, series.y
    ours = fit_well(series, models=[model])
    assert ours.status == "ok"

    (A, mu, lam), rss_independent = independent_fit(model, t, y)
    rss_ours = float(np.sum((y - MODELS[model](t, ours.A, ours.mu_max, ours.lag_h)) ** 2))

    assert ours.mu_max == pytest.approx(mu, rel=1e-3)
    assert ours.lag_h == pytest.approx(lam, rel=1e-3, abs=1e-3)
    assert ours.A == pytest.approx(A, rel=1e-3)
    assert rss_ours == pytest.approx(rss_independent, rel=1e-6)