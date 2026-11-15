"""Modelos de crescimento (parametrização de Zwietering et al., 1990).

Todos os modelos descrevem y(t) = ln(OD(t) / OD0) e usam a mesma assinatura
``modelo(t, A, mu, lam)``, em que:

- A   : amplitude assintótica em ln(OD/OD0)
- mu  : taxa máxima de crescimento específico (h^-1)
- lam : duração da fase lag (h)
"""

from __future__ import annotations

import numpy as np

# Limite do argumento de exp() para evitar overflow (exp(709) é o máximo em float64).
_EXP_LIMIT = 700.0


def _exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -_EXP_LIMIT, _EXP_LIMIT))


def logistic(t, A, mu, lam):
    """Logístico: y(t) = A / (1 + exp(4*mu/A*(lam - t) + 2))."""
    t = np.asarray(t, dtype=float)
    return A / (1.0 + _exp(4.0 * mu / A * (lam - t) + 2.0))


def gompertz(t, A, mu, lam):
    """Gompertz modificado: y(t) = A * exp(-exp(mu*e/A*(lam - t) + 1))."""
    t = np.asarray(t, dtype=float)
    return A * _exp(-_exp(mu * np.e / A * (lam - t) + 1.0))


MODELS = {
    "logistic": logistic,
    "gompertz": gompertz,
}

# Número de parâmetros livres de cada modelo (usado no AIC: 2k).
N_PARAMS = {
    "logistic": 3,
    "gompertz": 3,
}