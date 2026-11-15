import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import growthcurves
from growthcurves.cli import build_parser, main
from growthcurves.synthetic import (
    default_plate,
    make_plate,
    write_metadata_csv,
    write_plate_csv,
)

ROOT = Path(__file__).resolve().parents[1]

SIMPLE_META = "well,strain,condition,is_blank\nA1,S1,LB,false\n"


def make_inputs(folder, plate=None, meta=None, time_format="time_h"):
    folder.mkdir(parents=True, exist_ok=True)
    if plate is None:
        plate, meta = default_plate()
    plate_path = write_plate_csv(plate, folder / "plate.csv", time_format=time_format)
    meta_path = write_metadata_csv(meta, folder / "metadata.csv")
    return plate_path, meta_path


def run(plate_path, meta_path, out_dir, *extra):
    return main(["--input", str(plate_path), "--metadata", str(meta_path),
                 "--output-dir", str(out_dir), *extra])


def read_table(out_dir):
    return pd.read_csv(Path(out_dir) / "parameters.csv")


# ---------------------------------------------------------------- execução normal

def test_full_run_writes_all_outputs(tmp_path, capsys):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "results"
    assert run(plate_path, meta_path, out) == 0
    for name in ("parameters.csv", "qc_report.md", "run_info.json"):
        assert (out / name).is_file()
    assert sorted(p.name for p in (out / "plots").iterdir()) == [
        "A1.png", "A2.png", "A3.png", "B1.png"]
    table = read_table(out)
    assert len(table) == 7
    assert list(table["status"]).count("blank") == 3
    assert (table.loc[table["well"].isin(["A1", "A2", "A3"]), "status"] == "ok").all()
    assert "Resultados em:" in capsys.readouterr().out


def test_summary_reports_counts(tmp_path, capsys):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    run(plate_path, meta_path, tmp_path / "out")
    text = capsys.readouterr().out
    assert "4 poços analisados (3 ok, 1 sem crescimento, 0 com falha no ajuste)" in text
    assert "3 poços-branco" in text
    assert "Avisos" not in text


def test_run_creates_nested_output_dir(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "a" / "b" / "results"
    assert run(plate_path, meta_path, out) == 0
    assert (out / "parameters.csv").is_file()


def test_default_output_dir_is_results_in_current_directory(tmp_path, monkeypatch):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    monkeypatch.chdir(tmp_path)
    assert main(["--input", str(plate_path), "--metadata", str(meta_path)]) == 0
    assert (tmp_path / "results" / "parameters.csv").is_file()


def test_problem_wells_do_not_stop_the_run(tmp_path, capsys):
    specs = {
        "G1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0},
        "F1": {"model": "none"},
        "N1": {"model": "none", "od0": -0.1},
        "S1": {"model": "gompertz", "A": 2.0, "mu": 0.4, "lam": 4.0},
        "X1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0},
    }
    plate, meta = make_plate(specs, blank_wells=("H1", "H2", "H3"), seed=11)
    plate.loc[5:, "S1"] = np.nan
    meta = meta[meta["well"] != "X1"]
    plate_path, meta_path = make_inputs(tmp_path / "in", plate, meta)
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out) == 0
    status = read_table(out).set_index("well")["status"].to_dict()
    assert status["G1"] == "ok" and status["X1"] == "ok"
    assert status["F1"] == "no_growth"
    assert status["N1"] == "fit_failed" and status["S1"] == "fit_failed"
    assert "Avisos: 1" in capsys.readouterr().out
    assert "poço 'X1' sem entrada nos metadados" in (out / "qc_report.md").read_text("utf-8")


# ---------------------------------------------------------------- critérios de aceite

def test_time_h_and_time_inputs_give_identical_parameters_csv(tmp_path):
    out_h, out_t = tmp_path / "out_h", tmp_path / "out_t"
    p_h, m_h = make_inputs(tmp_path / "in_h", time_format="time_h")
    p_t, m_t = make_inputs(tmp_path / "in_t", time_format="time")
    assert run(p_h, m_h, out_h) == 0
    assert run(p_t, m_t, out_t) == 0
    assert (out_h / "parameters.csv").read_bytes() == (out_t / "parameters.csv").read_bytes()


def test_two_runs_give_identical_parameters_csv(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    assert run(plate_path, meta_path, tmp_path / "run1") == 0
    assert run(plate_path, meta_path, tmp_path / "run2") == 0
    assert (tmp_path / "run1" / "parameters.csv").read_bytes() == \
        (tmp_path / "run2" / "parameters.csv").read_bytes()


def test_rerunning_into_the_same_folder_overwrites_cleanly(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out) == 0
    first = (out / "parameters.csv").read_bytes()
    assert run(plate_path, meta_path, out) == 0
    assert (out / "parameters.csv").read_bytes() == first


def test_run_info_records_parameters_and_input_hash(tmp_path):
    import hashlib

    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    run(plate_path, meta_path, out, "--min-growth", "0.3", "--smooth-window", "5",
        "--min-points", "8", "--models", "gompertz")
    info = json.loads((out / "run_info.json").read_text("utf-8"))
    assert info["version"] == growthcurves.__version__
    assert info["input"]["sha256"] == hashlib.sha256(plate_path.read_bytes()).hexdigest()
    assert info["parameters"]["models"] == ["gompertz"]
    assert info["parameters"]["min_growth"] == 0.3
    assert info["parameters"]["smooth_window"] == 5
    assert info["parameters"]["min_points"] == 8
    assert info["parameters"]["n_baseline"] == 3


# ---------------------------------------------------------------- opções

def test_defaults_match_the_spec():
    args = build_parser().parse_args(["--input", "a", "--metadata", "b"])
    assert args.output_dir == "results/"
    assert args.models == "logistic,gompertz"
    assert args.smooth_window == 3
    assert args.min_points == 6
    assert args.min_growth == 0.2


def test_models_option_restricts_the_fitted_models(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out, "--models", "logistic") == 0
    table = read_table(out)
    assert set(table.loc[table["status"] == "ok", "model"]) == {"logistic"}


def test_min_growth_option_changes_no_growth_detection(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out, "--min-growth", "5") == 0
    table = read_table(out)
    analyzed = table[table["status"] != "blank"]
    assert (analyzed["status"] == "no_growth").all()


def test_min_points_option_changes_failure_threshold(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out, "--min-points", "100") == 0
    table = read_table(out)
    analyzed = table[table["status"] != "blank"]
    assert (analyzed["status"] == "fit_failed").all()
    assert "menos de 100 pontos" in (out / "qc_report.md").read_text("utf-8")


def test_smooth_window_option_changes_the_result(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    assert run(plate_path, meta_path, tmp_path / "w3") == 0
    assert run(plate_path, meta_path, tmp_path / "w1", "--smooth-window", "1") == 0
    assert (tmp_path / "w3" / "parameters.csv").read_bytes() != \
        (tmp_path / "w1" / "parameters.csv").read_bytes()


# ---------------------------------------------------------------- erros de entrada

BAD_PLATES = [
    ("time_h,A1\n0,0.1\nabc,0.2\n", "linha 3"),
    ("time,A1\n00:75:00,0.1\n", "00:75:00"),
    ("time_h,A1,A1\n0,1,2\n", "duplicadas"),
    ("A1,A2\n0.1,0.2\n", "exatamente uma coluna de tempo"),
    ("time_h,time,A1\n0,00:00:00,0.1\n", "exatamente uma coluna de tempo"),
    ("time_h\n0\n", "nenhuma coluna de poço"),
    ("time_h,A1\n0,0.1\n0,0.2\n", "estritamente crescente"),
    ("time_h,A1\n0,0.1\n1,xyz\n", "coluna 'A1'"),
    ("", "vazio"),
]


@pytest.mark.parametrize("text, expected", BAD_PLATES)
def test_bad_measurement_file_exits_1_with_message_and_no_outputs(tmp_path, capsys, text, expected):
    plate_path = tmp_path / "plate.csv"
    plate_path.write_text(text, encoding="utf-8")
    meta_path = tmp_path / "metadata.csv"
    meta_path.write_text(SIMPLE_META, encoding="utf-8")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out) == 1
    err = capsys.readouterr().err
    assert err.startswith("erro:")
    assert expected in err
    assert not out.exists()


def test_missing_input_file_exits_1(tmp_path, capsys):
    _, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(tmp_path / "nao_existe.csv", meta_path, out) == 1
    assert "arquivo não encontrado" in capsys.readouterr().err
    assert not out.exists()


def test_missing_metadata_file_exits_1(tmp_path, capsys):
    plate_path, _ = make_inputs(tmp_path / "in")
    assert run(plate_path, tmp_path / "nada.csv", tmp_path / "out") == 1
    assert "arquivo não encontrado" in capsys.readouterr().err


def test_bad_metadata_exits_1(tmp_path, capsys):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    meta_path.write_text("well,strain,condition,is_blank\nA1,S1,LB,talvez\n", encoding="utf-8")
    assert run(plate_path, meta_path, tmp_path / "out") == 1
    assert "is_blank" in capsys.readouterr().err


@pytest.mark.parametrize("extra, expected", [
    (("--models", "richards"), "desconhecido"),
    (("--smooth-window", "2"), "janela de suavização inválida"),
    (("--smooth-window", "0"), "janela de suavização inválida"),
    (("--min-points", "3"), "--min-points"),
    (("--min-growth", "-1"), "--min-growth"),
])
def test_invalid_options_exit_1_without_outputs(tmp_path, capsys, extra, expected):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    assert run(plate_path, meta_path, out, *extra) == 1
    assert expected in capsys.readouterr().err
    assert not out.exists()


def test_unwritable_output_dir_exits_1(tmp_path, capsys):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    blocker = tmp_path / "ocupado.txt"
    blocker.write_text("arquivo, não pasta", encoding="utf-8")
    assert run(plate_path, meta_path, blocker / "results") == 1
    assert "não foi possível gravar" in capsys.readouterr().err


def test_missing_required_arguments_exit_2():
    with pytest.raises(SystemExit) as exc:
        main(["--input", "plate.csv"])
    assert exc.value.code == 2


def test_non_numeric_option_value_exits_2():
    with pytest.raises(SystemExit) as exc:
        main(["--input", "a", "--metadata", "b", "--min-growth", "abc"])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert growthcurves.__version__ in capsys.readouterr().out


# ---------------------------------------------------------------- como módulo (python -m)

def _run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "growthcurves.cli", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )


def test_python_dash_m_runs_the_analysis(tmp_path):
    plate_path, meta_path = make_inputs(tmp_path / "in")
    out = tmp_path / "out"
    proc = _run_module("--input", str(plate_path), "--metadata", str(meta_path),
                       "--output-dir", str(out))
    assert proc.returncode == 0, proc.stderr
    assert "Análise concluída" in proc.stdout
    assert (out / "parameters.csv").is_file()


def test_python_dash_m_exits_nonzero_on_input_error(tmp_path):
    _, meta_path = make_inputs(tmp_path / "in")
    proc = _run_module("--input", str(tmp_path / "nao_existe.csv"),
                       "--metadata", str(meta_path), "--output-dir", str(tmp_path / "out"))
    assert proc.returncode == 1
    assert "erro: arquivo não encontrado" in proc.stderr


def test_python_dash_m_help_exits_0():
    proc = _run_module("--help")
    assert proc.returncode == 0
    assert "--min-points" in proc.stdout