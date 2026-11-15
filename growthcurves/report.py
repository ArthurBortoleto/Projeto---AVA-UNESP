"""Saídas de texto da análise: parameters.csv, qc_report.md e run_info.json."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__

PARAMETER_COLUMNS = [
    "well", "strain", "condition", "model", "mu_max", "lag_h", "A", "K",
    "doubling_time_h", "r2", "aic", "status",
]
STATUS_BLANK = "blank"

# Seis algarismos significativos: suficiente para a análise e mantém o arquivo
# idêntico entre execuções (e entre plataformas).
_FLOAT_FORMAT = "%.6g"


# ---------------------------------------------------------------- parameters.csv

def build_parameters_table(results, metadata: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por poço do arquivo de medidas, na ordem das colunas do arquivo.

    results  : lista de ``FitResult`` (poços que não são branco)
    metadata : metadados alinhados aos poços (``io.align_metadata``)
    Poços-branco entram com status ``blank`` e os demais campos vazios.
    """
    by_well = {r.well: r for r in results}
    rows = []
    for _, m in metadata.iterrows():
        well = m["well"]
        row = {"well": well, "strain": m["strain"], "condition": m["condition"]}
        if m["is_blank"]:
            row["status"] = STATUS_BLANK
        else:
            if well not in by_well:
                raise ValueError(f"poço '{well}' não tem resultado de ajuste")
            r = by_well[well]
            row.update(
                model=r.model, mu_max=r.mu_max, lag_h=r.lag_h, A=r.A, K=r.K,
                doubling_time_h=r.doubling_time_h, r2=r.r2, aic=r.aic, status=r.status,
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=PARAMETER_COLUMNS)


def write_parameters(table: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=_FLOAT_FORMAT, lineterminator="\n",
                 encoding="utf-8")
    return path


# ---------------------------------------------------------------- qc_report.md

def build_qc_report(results, pre, plate, warnings=()) -> str:
    """Texto (Markdown) do relatório de qualidade.

    results  : lista de ``FitResult``
    pre      : ``preprocess.Preprocessed`` (séries, poços-branco, avisos do pré-processamento)
    plate    : ``io.PlateData`` (células vazias descartadas)
    warnings : avisos anteriores ao pré-processamento (ex.: metadados × medidas)
    """
    by_status = {"ok": [], "no_growth": [], "fit_failed": []}
    for r in results:
        by_status[r.status].append(r)

    n_blank = len(pre.blank_wells)
    n_nonpositive = sum(s.n_nonpositive for s in pre.series.values())
    n_no_blank = sum(s.n_no_blank for s in pre.series.values())

    lines = [
        "# Relatório de qualidade",
        "",
        "## Resumo",
        "",
        "| Item | Poços |",
        "|---|---|",
        f"| Poços no arquivo | {len(plate.wells)} |",
        f"| Poços-branco | {n_blank} |",
        f"| Poços analisados | {len(results)} |",
        f"| Ajustados (ok) | {len(by_status['ok'])} |",
        f"| Sem crescimento (no_growth) | {len(by_status['no_growth'])} |",
        f"| Falha no ajuste (fit_failed) | {len(by_status['fit_failed'])} |",
        "",
        "## Pontos descartados",
        "",
        f"- Células vazias (medida ausente): {plate.n_missing}",
        f"- OD corrigido ≤ 0: {n_nonpositive}",
        f"- Sem medida de branco no mesmo instante: {n_no_blank}",
    ]

    detail = []
    for well, series in pre.series.items():
        empty = plate.missing_by_well.get(well, 0)
        if empty or series.n_nonpositive or series.n_no_blank:
            detail.append(f"| {well} | {empty} | {series.n_nonpositive} | {series.n_no_blank} |")
    if detail:
        lines += ["", "| Poço | Vazias | OD ≤ 0 | Sem branco |", "|---|---|---|---|", *detail]

    lines += ["", "## Poços sem crescimento", ""]
    if by_status["no_growth"]:
        lines += [f"- {r.well}: {r.reason}" for r in by_status["no_growth"]]
    else:
        lines.append("Nenhum.")

    lines += ["", "## Poços com falha no ajuste", ""]
    if by_status["fit_failed"]:
        lines += [f"- {r.well}: {r.reason}" for r in by_status["fit_failed"]]
    else:
        lines.append("Nenhum.")

    all_warnings = [*warnings, *pre.warnings]
    lines += ["", "## Avisos", ""]
    if all_warnings:
        lines += [f"- {w}" for w in all_warnings]
    else:
        lines.append("Nenhum aviso.")

    return "\n".join(lines) + "\n"


def write_qc_report(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------- run_info.json

def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_info(input_path, metadata_path, parameters: dict, now=None) -> dict:
    """Versão do programa, parâmetros usados, data (UTC) e hash dos arquivos de entrada."""
    now = datetime.now(timezone.utc) if now is None else now
    return {
        "program": "growthcurves",
        "version": __version__,
        "date_utc": now.replace(microsecond=0).isoformat(),
        "input": {"path": str(input_path), "sha256": file_sha256(input_path)},
        "metadata": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in parameters.items()},
    }


def write_run_info(info: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    return path