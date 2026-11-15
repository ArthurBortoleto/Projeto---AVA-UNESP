import warnings

import numpy as np
import pytest

from growthcurves.models import MODELS, N_PARAMS, gompertz, logistic

PARAMS = dict(A=2.0, mu=0.5, lam=3.0)


def inflection(model, A, mu, lam):
    """Ponto de inflexão (t, y) de cada modelo, onde a inclinação é máxima (= mu)."""
    if model is logistic:
        return lam + A / (2 * mu), A / 2
    return lam + A / (np.e * mu), A / np.e


def test_registry_has_both_models():
    assert set(MODELS) == {"logistic", "gompertz"}
    assert MODELS["logistic"] is logistic
    assert MODELS["gompertz"] is gompertz
    assert N_PARAMS == {"logistic": 3, "gompertz": 3}


def test_value_at_lag_logistic():
    assert logistic(3.0, **PARAMS) == pytest.approx(2.0 / (1 + np.exp(2)))


def test_value_at_lag_gompertz():
    assert gompertz(3.0, **PARAMS) == pytest.approx(2.0 * np.exp(-np.e))


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_starts_near_zero(model):
    assert model(0.0, **PARAMS) < 0.01 * PARAMS["A"]


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_approaches_asymptote_A(model):
    assert model(1000.0, **PARAMS) == pytest.approx(PARAMS["A"], rel=1e-9)


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_monotonic_increasing(model):
    t = np.linspace(0, 40, 400)
    assert np.all(np.diff(model(t, **PARAMS)) >= 0)


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_max_slope_equals_mu(model):
    t = np.linspace(0, 40, 40001)
    slope = np.gradient(model(t, **PARAMS), t)
    assert slope.max() == pytest.approx(PARAMS["mu"], rel=1e-3)


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_tangent_at_inflection_crosses_zero_at_lag(model):
    # Definição de lam: a tangente de inclinação máxima cruza y = 0 em t = lam.
    t_i, y_i = inflection(model, **PARAMS)
    assert model(t_i, **PARAMS) == pytest.approx(y_i)
    assert t_i - y_i / PARAMS["mu"] == pytest.approx(PARAMS["lam"])


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_scaling_A_and_mu_together_scales_curve(model):
    t = np.linspace(0, 30, 61)
    base = model(t, A=2.0, mu=0.5, lam=3.0)
    scaled = model(t, A=4.0, mu=1.0, lam=3.0)
    np.testing.assert_allclose(scaled, 2 * base)


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_accepts_scalar_list_and_array(model):
    assert np.ndim(model(1.0, **PARAMS)) == 0
    from_list = model([0.0, 1.0, 2.0], **PARAMS)
    from_array = model(np.array([0.0, 1.0, 2.0]), **PARAMS)
    assert from_list.shape == (3,)
    np.testing.assert_array_equal(from_list, from_array)


@pytest.mark.parametrize("model", [logistic, gompertz])
def test_no_overflow_for_extreme_times(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = model(np.array([-1e6, 1e6]), **PARAMS)
    assert np.all(np.isfinite(y))
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y[1] == pytest.approx(PARAMS["A"])