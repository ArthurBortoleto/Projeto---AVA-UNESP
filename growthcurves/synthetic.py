"""Gerador de dados sintéticos (semente fixa) para testes e para o exemplo do README.

Para gerar os arquivos de exemplo em data/synthetic/:

    python -m growthcurves.synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .models import MODELS

DEFAULT_SEED = 42
# OD inicial padrão das curvas simuladas. Com ruído de 0,01 em OD, valores menores
# (ex.: 0,1) deixam o ruído relativo alto demais e os critérios de aceite não fecham.
DEFAULT_OD0 = 0.2
EXAMPLE_DIR = "data/synthetic"


def time_grid(t_max: float = 24.0, step: float = 0.5) -> np.ndarray:
    """Eixo de tempo em horas: 0, step, 2*step, ..., t_max."""
    return np.arange(0.0, t_max + step / 2, step)


def simulate_od(model, t, A, mu, lam, od0=DEFAULT_OD0, noise_sd=0.01, rng=None):
    """OD simulada = od0 * exp(y(t)) + ruído gaussiano (desvio-padrão em unidades de OD)."""
    if model not in MODELS:
        raise ValueError(f"Modelo desconhecido: {model!r}. Use um de {sorted(MODELS)}.")
    rng = np.random.default_rng(DEFAULT_SEED) if rng is None else rng
    t = np.asarray(t, dtype=float)
    od = od0 * np.exp(MODELS[model](t, A, mu, lam))
    if noise_sd > 0:
        od = od + rng.normal(0.0, noise_sd, size=t.shape)
    return od


def make_plate(specs, t=None, blank_wells=(), blank_level=0.04, noise_sd=0.01,
               seed=DEFAULT_SEED):
    """Monta uma placa sintética em formato largo e a tabela de metadados.

    specs: dict {poço: {"model": "logistic" | "gompertz" | "none",
                        "A", "mu", "lam", "od0" (opcional),
                        "strain", "condition" (opcionais)}}
           model "none" gera um poço sem crescimento (OD constante + ruído).
    Os poços-branco recebem apenas ``blank_level`` + ruído; os demais somam
    ``blank_level`` à OD simulada (assim a correção de branco devolve a OD original).
    """
    t = time_grid() if t is None else np.asarray(t, dtype=float)
    rng = np.random.default_rng(seed)
    data = {"time_h": t}
    meta = []

    for well, s in specs.items():
        od0 = s.get("od0", DEFAULT_OD0)
        if s["model"] == "none":
            od = od0 + rng.normal(0.0, noise_sd, size=t.shape)
        else:
            od = simulate_od(s["model"], t, s["A"], s["mu"], s["lam"], od0, noise_sd, rng)
        data[well] = blank_level + od
        meta.append({"well": well, "strain": s.get("strain", "S1"),
                     "condition": s.get("condition", "C1"), "is_blank": False})

    for well in blank_wells:
        data[well] = blank_level + rng.normal(0.0, noise_sd, size=t.shape)
        meta.append({"well": well, "strain": "", "condition": "meio", "is_blank": True})

    return pd.DataFrame(data), pd.DataFrame(meta)


def default_plate(seed=DEFAULT_SEED):
    """Placa de exemplo: 3 curvas com crescimento, 1 sem crescimento e 3 brancos."""
    specs = {
        "A1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0,
               "strain": "S1", "condition": "LB"},
        "A2": {"model": "gompertz", "A": 2.0, "mu": 0.4, "lam": 4.0,
               "strain": "S1", "condition": "LB"},
        "A3": {"model": "logistic", "A": 1.5, "mu": 0.3, "lam": 6.0,
               "strain": "S2", "condition": "LB"},
        "B1": {"model": "none", "strain": "S2", "condition": "LB"},
    }
    return make_plate(specs, blank_wells=("H1", "H2", "H3"), seed=seed)


def hours_to_hhmmss(hours: float) -> str:
    """Converte horas decimais para 'hh:mm:ss' (horas podem passar de 24)."""
    total = int(round(hours * 3600))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_plate_csv(plate: pd.DataFrame, path, time_format: str = "time_h") -> Path:
    """Grava a placa em CSV largo, com tempo em 'time_h' (horas) ou 'time' (hh:mm:ss)."""
    if time_format not in ("time_h", "time"):
        raise ValueError("time_format deve ser 'time_h' ou 'time'.")
    out = plate.copy()
    if time_format == "time":
        out["time"] = [hours_to_hhmmss(h) for h in out["time_h"]]
        out = out.drop(columns="time_h")
        out = out[["time"] + [c for c in out.columns if c != "time"]]
    path = Path(path)
    out.to_csv(path, index=False, lineterminator="\n")
    return path


def write_metadata_csv(metadata: pd.DataFrame, path) -> Path:
    """Grava os metadados com is_blank em minúsculas (true/false), como no spec."""
    out = metadata.copy()
    out["is_blank"] = out["is_blank"].map({True: "true", False: "false"})
    path = Path(path)
    out.to_csv(path, index=False, lineterminator="\n")
    return path


def example_plate(seed=DEFAULT_SEED):
    """Placa de exemplo do README: 2 cepas x 2 meios, 3 réplicas cada.

    - A1-A3: S1 em meio_A (logístico);  B1-B3: S1 em meio_B (logístico, mais lento)
    - C1-C3: S2 em meio_A (Gompertz);   D1-D2: S2 em meio_B (Gompertz, mais lento)
    - D3: sem crescimento;  H1-H3: poços-branco
    - A2 tem uma célula vazia (t = 10 h), para mostrar o descarte de pontos no relatório.
    """
    groups = [
        ("A", "S1", "meio_A", "logistic", 2.0, 0.50, 3.0),
        ("B", "S1", "meio_B", "logistic", 1.8, 0.30, 5.0),
        ("C", "S2", "meio_A", "gompertz", 2.0, 0.45, 4.0),
        ("D", "S2", "meio_B", "gompertz", 1.6, 0.25, 6.0),
    ]
    specs = {}
    for row, strain, condition, model, A, mu, lam in groups:
        for replicate in (1, 2, 3):
            shift = replicate - 2  # -1, 0, +1: pequena variação entre réplicas
            specs[f"{row}{replicate}"] = {
                "model": model, "A": A,
                "mu": mu * (1 + 0.06 * shift), "lam": lam * (1 + 0.05 * shift),
                "strain": strain, "condition": condition,
            }
    specs["D3"] = {"model": "none", "strain": "S2", "condition": "meio_B"}
    plate, meta = make_plate(specs, blank_wells=("H1", "H2", "H3"), seed=seed)
    plate.loc[20, "A2"] = np.nan
    return plate, meta


def write_example(output_dir=EXAMPLE_DIR, seed=DEFAULT_SEED):
    """Grava plate.csv e metadata.csv do exemplo; devolve os dois caminhos."""
    plate, meta = example_plate(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return (write_plate_csv(plate, out / "plate.csv"),
            write_metadata_csv(meta, out / "metadata.csv"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m growthcurves.synthetic",
        description="Gera os dados sintéticos de exemplo (plate.csv e metadata.csv).",
    )
    parser.add_argument("--output-dir", default=EXAMPLE_DIR,
                        help="pasta de saída (padrão: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="semente do gerador (padrão: %(default)s)")
    args = parser.parse_args(argv)
    plate_path, meta_path = write_example(args.output_dir, args.seed)
    print("Arquivos gerados:")
    print(f"  {plate_path}")
    print(f"  {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())