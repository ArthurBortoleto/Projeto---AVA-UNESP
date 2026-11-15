import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from growthcurves.cli import main as cli_main
from growthcurves.io import read_metadata, read_plate
from growthcurves.synthetic import (
    DEFAULT_SEED,
    EXAMPLE_DIR,
    example_plate,
    main,
    write_example,
)

ROOT = Path(__file__).resolve().parents[1]


def test_example_plate_structure():
    plate, meta = example_plate()
    growth = [f"{row}{i}" for row in "ABCD" for i in (1, 2, 3)]
    assert list(plate.columns) == ["time_h", *growth, "H1", "H2", "H3"]
    assert set(meta["well"]) == set(plate.columns) - {"time_h"}
    assert list(meta.loc[meta["is_blank"], "well"]) == ["H1", "H2", "H3"]


def test_example_metadata_has_two_strains_and_two_conditions():
    _, meta = example_plate()
    growth = meta[~meta["is_blank"]]
    assert set(growth["strain"]) == {"S1", "S2"}
    assert set(growth["condition"]) == {"meio_A", "meio_B"}
    assert growth.groupby(["strain", "condition"]).size().eq(3).all()


def test_example_plate_has_one_empty_cell_in_A2():
    plate, _ = example_plate()
    assert plate.isna().sum().sum() == 1
    assert pd.isna(plate.loc[20, "A2"])
    assert plate.loc[20, "time_h"] == 10.0


def test_example_plate_is_reproducible_and_seed_dependent():
    a, _ = example_plate(seed=DEFAULT_SEED)
    b, _ = example_plate(seed=DEFAULT_SEED)
    c, _ = example_plate(seed=DEFAULT_SEED + 1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_write_example_creates_readable_files(tmp_path):
    plate_path, meta_path = write_example(tmp_path / "nested" / "synthetic")
    assert plate_path.name == "plate.csv" and meta_path.name == "metadata.csv"
    data = read_plate(plate_path)
    assert len(data.wells) == 15
    assert data.n_missing == 1 and data.missing_by_well == {"A2": 1}
    assert int(read_metadata(meta_path)["is_blank"].sum()) == 3


def test_written_csv_files_use_unix_line_endings(tmp_path):
    plate_path, meta_path = write_example(tmp_path)
    assert b"\r" not in plate_path.read_bytes()
    assert b"\r" not in meta_path.read_bytes()


def test_write_example_is_byte_identical_across_calls(tmp_path):
    first = write_example(tmp_path / "a")
    second = write_example(tmp_path / "b")
    for f, s in zip(first, second):
        assert f.read_bytes() == s.read_bytes()


def test_main_writes_files_and_prints_paths(tmp_path, capsys):
    assert main(["--output-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "plate.csv").is_file()
    assert (tmp_path / "out" / "metadata.csv").is_file()
    assert "Arquivos gerados" in capsys.readouterr().out


def test_python_dash_m_generates_the_files(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "growthcurves.synthetic", "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "plate.csv").is_file()


def test_cli_analysis_of_the_example(tmp_path):
    plate_path, meta_path = write_example(tmp_path / "data")
    out = tmp_path / "results"
    assert cli_main(["--input", str(plate_path), "--metadata", str(meta_path),
                     "--output-dir", str(out)]) == 0
    table = pd.read_csv(out / "parameters.csv").set_index("well")
    assert list(table["status"]).count("ok") == 11
    assert table.loc["D3", "status"] == "no_growth"
    assert list(table["status"]).count("blank") == 3
    assert len(list((out / "plots").glob("*.png"))) == 12
    assert "- Células vazias (medida ausente): 1" in (out / "qc_report.md").read_text("utf-8")


def test_example_analysis_recovers_the_known_parameters(tmp_path):
    plate_path, meta_path = write_example(tmp_path / "data")
    out = tmp_path / "results"
    cli_main(["--input", str(plate_path), "--metadata", str(meta_path),
              "--output-dir", str(out)])
    table = pd.read_csv(out / "parameters.csv").set_index("well")
    assert table.loc["A2", "mu_max"] == pytest.approx(0.50, rel=0.10)   # réplica central
    assert table.loc["B2", "mu_max"] == pytest.approx(0.30, rel=0.10)
    assert table.loc["C2", "mu_max"] == pytest.approx(0.45, rel=0.10)
    assert table.loc["D2", "mu_max"] == pytest.approx(0.25, rel=0.10)
    assert table.loc["A2", "K"] == pytest.approx(0.2 * 7.389, rel=0.05)  # 0,2 * exp(2)


def test_committed_example_files_match_the_generator(tmp_path):
    committed = ROOT / EXAMPLE_DIR
    if not (committed / "plate.csv").exists():
        pytest.skip("data/synthetic/ ainda não foi gerado (python -m growthcurves.synthetic)")
    fresh_plate, fresh_meta = write_example(tmp_path)
    # Compara valores, não bytes: o último dígito de um decimal pode variar entre máquinas
    # (e finais de linha CRLF/LF não importam). Mudança real no gerador continua sendo pega.
    problems = (pd.errors.EmptyDataError, pd.errors.ParserError, AssertionError)
    for name, fresh in [("plate.csv", fresh_plate), ("metadata.csv", fresh_meta)]:
        try:
            pd.testing.assert_frame_equal(pd.read_csv(committed / name), pd.read_csv(fresh),
                                          rtol=1e-9, atol=1e-12)
        except problems as exc:
            pytest.fail(f"data/synthetic/{name} está diferente do gerador; "
                        f"rode python -m growthcurves.synthetic\n{exc}")