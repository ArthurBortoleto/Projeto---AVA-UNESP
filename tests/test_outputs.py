import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import growthcurves
from growthcurves.fit import fit_all
from growthcurves.io import align_metadata, read_metadata, read_plate
from growthcurves.preprocess import preprocess
from growthcurves.report import (
    PARAMETER_COLUMNS,
    build_parameters_table,
    build_qc_report,
    build_run_info,
    file_sha256,
    write_parameters,
    write_qc_report,
    write_run_info,
)
from growthcurves.synthetic import (
    DEFAULT_OD0,
    default_plate,
    make_plate,
    write_metadata_csv,
    write_plate_csv,
)


def run_pipeline(tmp_path, plate, meta):
    """Escreve os CSVs de entrada e roda leitura -> pré-processamento -> ajuste."""
    plate_path = write_plate_csv(plate, tmp_path / "plate.csv")
    meta_path = write_metadata_csv(meta, tmp_path / "metadata.csv")
    data = read_plate(plate_path)
    aligned, io_warnings = align_metadata(data.wells, read_metadata(meta_path))
    pre = preprocess(data.long, aligned)
    results = fit_all(pre.series)
    return {"plate_path": plate_path, "meta_path": meta_path, "data": data, "meta": aligned,
            "io_warnings": io_warnings, "pre": pre, "results": results}


@pytest.fixture
def default_run(tmp_path):
    plate, meta = default_plate()
    return run_pipeline(tmp_path, plate, meta)


@pytest.fixture
def problem_run(tmp_path):
    """Placa com um poço de cada problema: sem crescimento, OD negativa, poucos pontos,
    célula vazia, e um poço sem metadados."""
    specs = {
        "G1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0},
        "F1": {"model": "none"},
        "N1": {"model": "none", "od0": -0.1},
        "S1": {"model": "gompertz", "A": 2.0, "mu": 0.4, "lam": 4.0},
        "X1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0},
    }
    plate, meta = make_plate(specs, blank_wells=("H1", "H2", "H3"), seed=11)
    plate.loc[5:, "S1"] = np.nan          # poucos pontos
    plate.loc[3, "G1"] = np.nan           # uma célula vazia
    meta = meta[meta["well"] != "X1"]     # X1 sem metadados
    return run_pipeline(tmp_path, plate, meta)


# ---------------------------------------------------------------- tabela de parâmetros

def test_parameter_columns_match_the_spec():
    assert PARAMETER_COLUMNS == ["well", "strain", "condition", "model", "mu_max", "lag_h",
                                 "A", "K", "doubling_time_h", "r2", "aic", "status"]


def test_table_has_one_row_per_plate_well_in_file_order(default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    assert list(table.columns) == PARAMETER_COLUMNS
    assert list(table["well"]) == ["A1", "A2", "A3", "B1", "H1", "H2", "H3"]


def test_blank_wells_have_status_blank_and_empty_fields(default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    blanks = table[table["status"] == "blank"]
    assert list(blanks["well"]) == ["H1", "H2", "H3"]
    numeric = ["mu_max", "lag_h", "A", "K", "doubling_time_h", "r2", "aic"]
    assert blanks[numeric].isna().all().all()
    assert blanks["model"].isna().all()
    assert (blanks["condition"] == "meio").all()


def test_growth_rows_carry_metadata_and_fit_values(default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    a1 = table[table["well"] == "A1"].iloc[0]
    assert a1["strain"] == "S1" and a1["condition"] == "LB"
    assert a1["status"] == "ok"
    assert a1["model"] in ("logistic", "gompertz")
    assert a1["mu_max"] == pytest.approx(0.5, rel=0.25)
    assert a1["K"] == pytest.approx(DEFAULT_OD0 * math.exp(2.0), rel=0.10)
    assert a1["doubling_time_h"] == pytest.approx(math.log(2) / a1["mu_max"])


def test_non_ok_rows_have_empty_numeric_fields(problem_run):
    table = build_parameters_table(problem_run["results"], problem_run["meta"])
    by_well = table.set_index("well")
    for well, status in [("F1", "no_growth"), ("N1", "fit_failed"), ("S1", "fit_failed")]:
        row = by_well.loc[well]
        assert row["status"] == status
        assert pd.isna(row["model"]) and pd.isna(row["mu_max"]) and pd.isna(row["aic"])
    assert by_well.loc["G1", "status"] == "ok"


def test_well_without_metadata_has_empty_strain_and_condition(problem_run):
    table = build_parameters_table(problem_run["results"], problem_run["meta"])
    x1 = table[table["well"] == "X1"].iloc[0]
    assert x1["strain"] == "" and x1["condition"] == ""
    assert x1["status"] == "ok"


def test_missing_result_for_non_blank_well_raises(default_run):
    with pytest.raises(ValueError, match="A1"):
        build_parameters_table(default_run["results"][1:], default_run["meta"])


# ---------------------------------------------------------------- parameters.csv

def test_write_parameters_creates_nested_dirs_and_header(tmp_path, default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    path = write_parameters(table, tmp_path / "out" / "deep" / "parameters.csv")
    assert path.exists()
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == ",".join(PARAMETER_COLUMNS)


def test_parameters_csv_reads_back_with_blank_and_empty_cells(tmp_path, default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    path = write_parameters(table, tmp_path / "parameters.csv")
    back = pd.read_csv(path, keep_default_na=True)
    assert list(back["well"]) == list(table["well"])
    assert back.loc[back["well"] == "H1", "mu_max"].isna().all()
    ok = back[back["status"] == "ok"]
    assert ok["mu_max"].notna().all()
    for _, row in ok.iterrows():
        original = table.loc[table["well"] == row["well"], "mu_max"].iloc[0]
        assert row["mu_max"] == pytest.approx(original, rel=1e-5)


def test_parameters_csv_uses_unix_line_endings(tmp_path, default_run):
    table = build_parameters_table(default_run["results"], default_run["meta"])
    path = write_parameters(table, tmp_path / "parameters.csv")
    assert b"\r" not in path.read_bytes()


def test_two_runs_produce_identical_parameters_csv(tmp_path):
    outputs = []
    for name in ("run1", "run2"):
        folder = tmp_path / name
        folder.mkdir()
        plate, meta = default_plate()
        run = run_pipeline(folder, plate, meta)
        table = build_parameters_table(run["results"], run["meta"])
        outputs.append(write_parameters(table, folder / "parameters.csv").read_bytes())
    assert outputs[0] == outputs[1]


# ---------------------------------------------------------------- qc_report.md

def _report(run):
    return build_qc_report(run["results"], run["pre"], run["data"], run["io_warnings"])


def test_qc_report_summary_counts(problem_run):
    text = _report(problem_run)
    n_wells = len(problem_run["data"].wells)
    assert "# Relatório de qualidade" in text
    assert f"| Poços no arquivo | {n_wells} |" in text
    assert "| Poços-branco | 3 |" in text
    assert f"| Poços analisados | {n_wells - 3} |" in text
    assert "| Ajustados (ok) | 2 |" in text
    assert "| Sem crescimento (no_growth) | 1 |" in text
    assert "| Falha no ajuste (fit_failed) | 2 |" in text


def test_qc_report_counts_discarded_points(problem_run):
    text = _report(problem_run)
    series = problem_run["pre"].series
    n_nonpositive = sum(s.n_nonpositive for s in series.values())
    assert n_nonpositive > 0
    assert f"- Células vazias (medida ausente): {problem_run['data'].n_missing}" in text
    assert f"- OD corrigido ≤ 0: {n_nonpositive}" in text
    assert "- Sem medida de branco no mesmo instante: 0" in text
    assert "| Poço | Vazias | OD ≤ 0 | Sem branco |" in text
    assert "| G1 | 1 | " in text                       # a célula vazia de G1
    assert f"| N1 | 0 | {len(problem_run['data'].long['time_h'].unique())} | 0 |" in text


def test_qc_report_lists_no_growth_and_failures_with_reasons(problem_run):
    text = _report(problem_run)
    assert "- F1: max(y) =" in text and "min-growth" in text
    assert "- S1: menos de 6 pontos válidos (5)" in text
    assert "- N1: menos de 6 pontos válidos (0)" in text


def test_qc_report_includes_warnings(problem_run):
    text = _report(problem_run)
    assert "## Avisos" in text
    assert "poço 'X1' sem entrada nos metadados" in text


def test_qc_report_without_problems(default_run):
    text = _report(default_run)
    assert "- Células vazias (medida ausente): 0" in text
    assert "Nenhum aviso." in text
    assert "| Poço | Vazias |" not in text              # sem tabela de detalhe


def test_qc_report_includes_preprocess_warnings(tmp_path):
    plate, meta = default_plate()
    plate = plate.drop(columns=["H1", "H2", "H3"])
    meta = meta[~meta["is_blank"]]
    run = run_pipeline(tmp_path, plate, meta)
    assert "nenhum poço-branco" in _report(run)


def test_write_qc_report(tmp_path, default_run):
    path = write_qc_report(_report(default_run), tmp_path / "out" / "qc_report.md")
    assert path.read_text(encoding="utf-8").startswith("# Relatório de qualidade")
    assert b"\r" not in path.read_bytes()


# ---------------------------------------------------------------- run_info.json

PARAMS = {"models": ["logistic", "gompertz"], "min_growth": 0.2, "smooth_window": 3,
          "min_points": 6}


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert file_sha256(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_run_info_content(default_run):
    now = datetime(2026, 11, 14, 12, 30, 45, 123456, tzinfo=timezone.utc)
    info = build_run_info(default_run["plate_path"], default_run["meta_path"], PARAMS, now=now)
    assert info["program"] == "growthcurves"
    assert info["version"] == growthcurves.__version__
    assert info["date_utc"] == "2026-11-14T12:30:45+00:00"
    assert info["input"]["sha256"] == hashlib.sha256(
        default_run["plate_path"].read_bytes()).hexdigest()
    assert info["metadata"]["sha256"] == hashlib.sha256(
        default_run["meta_path"].read_bytes()).hexdigest()
    assert info["parameters"] == PARAMS


def test_run_info_default_date_is_utc_now(default_run):
    before = datetime.now(timezone.utc).replace(microsecond=0)
    info = build_run_info(default_run["plate_path"], default_run["meta_path"], PARAMS)
    stamp = datetime.fromisoformat(info["date_utc"])
    assert stamp.tzinfo is not None
    assert before <= stamp <= datetime.now(timezone.utc)


def test_run_info_converts_paths_to_strings(default_run, tmp_path):
    params = {**PARAMS, "output_dir": tmp_path / "results"}
    info = build_run_info(default_run["plate_path"], default_run["meta_path"], params)
    assert info["parameters"]["output_dir"] == str(tmp_path / "results")


def test_hash_changes_when_input_changes(default_run):
    before = file_sha256(default_run["plate_path"])
    with open(default_run["plate_path"], "a", encoding="utf-8") as fh:
        fh.write("\n")
    assert file_sha256(default_run["plate_path"]) != before


def test_write_run_info_is_valid_json(tmp_path, default_run):
    info = build_run_info(default_run["plate_path"], default_run["meta_path"], PARAMS)
    path = write_run_info(info, tmp_path / "out" / "run_info.json")
    assert json.loads(path.read_text(encoding="utf-8")) == info