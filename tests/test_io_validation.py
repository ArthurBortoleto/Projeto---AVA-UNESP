import numpy as np
import pandas as pd
import pytest

from growthcurves.io import (
    InputError,
    align_metadata,
    read_metadata,
    read_plate,
)


def write(tmp_path, text, name="plate.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


# ---------------------------------------------------------------- arquivo

def test_missing_file_raises(tmp_path):
    with pytest.raises(InputError, match="não encontrado"):
        read_plate(tmp_path / "nao_existe.csv")


def test_directory_instead_of_file_raises(tmp_path):
    with pytest.raises(InputError, match="não foi possível ler"):
        read_plate(tmp_path)


def test_empty_file_raises(tmp_path):
    with pytest.raises(InputError, match="vazio"):
        read_plate(write(tmp_path, ""))


def test_header_without_data_rows_raises(tmp_path):
    with pytest.raises(InputError, match="nenhuma linha de dados"):
        read_plate(write(tmp_path, "time_h,A1\n"))


def test_undecodable_file_raises(tmp_path):
    path = tmp_path / "plate.csv"
    path.write_bytes(b"time_h,A1\n0,\xff\xfe\n")
    with pytest.raises(InputError, match="não foi possível ler"):
        read_plate(path)


def test_semicolon_separator_gets_a_hint(tmp_path):
    with pytest.raises(InputError, match="ponto e vírgula"):
        read_plate(write(tmp_path, "time_h;A1;A2\n0;0.1;0.2\n"))


def test_utf8_bom_is_handled(tmp_path):
    path = write(tmp_path, "time_h,A1\n0,0.1\n1,0.2\n", encoding="utf-8-sig")
    assert read_plate(path).wells == ["A1"]


# ---------------------------------------------------------------- cabeçalho

def test_no_time_column_raises(tmp_path):
    with pytest.raises(InputError, match="exatamente uma coluna de tempo"):
        read_plate(write(tmp_path, "A1,A2\n0.1,0.2\n"))


def test_both_time_columns_raise(tmp_path):
    with pytest.raises(InputError, match="exatamente uma coluna de tempo"):
        read_plate(write(tmp_path, "time_h,time,A1\n0,00:00:00,0.1\n"))


def test_no_well_columns_raise(tmp_path):
    with pytest.raises(InputError, match="nenhuma coluna de poço"):
        read_plate(write(tmp_path, "time_h\n0\n1\n"))


def test_duplicate_well_columns_raise(tmp_path):
    # O pandas renomearia a segunda coluna para "A1.1" sem avisar; aqui deve falhar.
    with pytest.raises(InputError, match=r"duplicadas.*A1"):
        read_plate(write(tmp_path, "time_h,A1,A1\n0,0.1,0.2\n"))


def test_empty_column_name_raises(tmp_path):
    with pytest.raises(InputError, match="nome de coluna vazio"):
        read_plate(write(tmp_path, "time_h,A1,\n0,0.1,0.2\n"))


def test_ragged_row_raises_with_line(tmp_path):
    with pytest.raises(InputError, match=r"linha 3 tem 2 campos"):
        read_plate(write(tmp_path, "time_h,A1,A2\n0,0.1,0.2\n1,0.3\n"))


# ---------------------------------------------------------------- valores

def test_non_numeric_time_h_reports_line_and_column(tmp_path):
    with pytest.raises(InputError, match=r"linha 3, coluna 'time_h'.*abc"):
        read_plate(write(tmp_path, "time_h,A1\n0,0.1\nabc,0.2\n"))


def test_non_numeric_well_value_reports_line_and_column(tmp_path):
    with pytest.raises(InputError, match=r"linha 3, coluna 'A2'.*xyz"):
        read_plate(write(tmp_path, "time_h,A1,A2\n0,0.1,0.2\n1,0.3,xyz\n"))


@pytest.mark.parametrize("bad", ["abc", "nan", "inf", "1_0", "0,5", "--1", "1.2.3"])
def test_rejects_non_plain_numbers(tmp_path, bad):
    path = write(tmp_path, f'time_h,A1\n0,0.1\n1,"{bad}"\n')
    with pytest.raises(InputError, match=r"linha 3, coluna 'A1'"):
        read_plate(path)


@pytest.mark.parametrize("good, value", [("1e-2", 0.01), (".5", 0.5), ("+0.3", 0.3), (" 0.25 ", 0.25)])
def test_accepts_common_number_formats(tmp_path, good, value):
    path = write(tmp_path, f'time_h,A1\n0,"{good}"\n1,0.1\n')
    assert read_plate(path).long["od"].iloc[0] == pytest.approx(value)


def test_negative_time_raises(tmp_path):
    with pytest.raises(InputError, match="tempo negativo na linha 2"):
        read_plate(write(tmp_path, "time_h,A1\n-1,0.1\n0,0.2\n"))


@pytest.mark.parametrize("times", [[0, 1, 1], [0, 2, 1]])
def test_repeated_or_decreasing_time_raises(tmp_path, times):
    body = "".join(f"{t},0.1\n" for t in times)
    with pytest.raises(InputError, match="estritamente crescente"):
        read_plate(write(tmp_path, "time_h,A1\n" + body))


def test_blank_lines_are_ignored_but_counted_in_line_numbers(tmp_path):
    path = write(tmp_path, "time_h,A1\n0,0.1\n\n1,abc\n")
    with pytest.raises(InputError, match="linha 4"):
        read_plate(path)


# ---------------------------------------------------------------- formato longo e células vazias

def test_long_format_structure(tmp_path):
    path = write(tmp_path, "time_h,A1,A2\n0,0.1,0.2\n0.5,0.3,0.4\n")
    data = read_plate(path)
    assert list(data.long.columns) == ["well", "time_h", "od"]
    assert list(data.long["well"]) == ["A1", "A1", "A2", "A2"]
    np.testing.assert_array_equal(data.long["time_h"], [0.0, 0.5, 0.0, 0.5])
    np.testing.assert_array_equal(data.long["od"], [0.1, 0.3, 0.2, 0.4])
    assert data.long["od"].dtype == float
    assert data.n_missing == 0
    assert data.missing_by_well == {}


def test_wells_keep_file_order(tmp_path):
    path = write(tmp_path, "B2,time_h,A1\n0.2,0,0.1\n0.3,1,0.2\n")
    assert read_plate(path).wells == ["B2", "A1"]


def test_empty_cells_are_dropped_and_counted(tmp_path):
    path = write(tmp_path, "time_h,A1,A2\n0,0.1,0.2\n1,,0.3\n2,0.3,\n3,0.4,0.5\n")
    data = read_plate(path)
    assert data.n_missing == 2
    assert data.missing_by_well == {"A1": 1, "A2": 1}
    assert len(data.long) == 6
    a1_times = list(data.long.loc[data.long["well"] == "A1", "time_h"])
    a2_times = list(data.long.loc[data.long["well"] == "A2", "time_h"])
    assert a1_times == [0.0, 2.0, 3.0]
    assert a2_times == [0.0, 1.0, 3.0]
    assert data.long["od"].notna().all()


# ---------------------------------------------------------------- metadados

def test_read_metadata_ok(tmp_path):
    path = write(tmp_path, (
        "well,strain,condition,is_blank\n"
        "A1,S1,LB,false\n"
        "H1,,meio,TRUE\n"
        "H2,,meio,True\n"
    ), name="meta.csv")
    meta = read_metadata(path)
    assert list(meta.columns) == ["well", "strain", "condition", "is_blank"]
    assert list(meta["well"]) == ["A1", "H1", "H2"]
    assert list(meta["is_blank"]) == [False, True, True]
    assert meta["is_blank"].dtype == bool
    assert meta.loc[1, "strain"] == ""  # vazio continua string, não NaN


def test_read_metadata_missing_file(tmp_path):
    with pytest.raises(InputError, match="não encontrado"):
        read_metadata(tmp_path / "nada.csv")


def test_read_metadata_missing_columns(tmp_path):
    path = write(tmp_path, "well,strain\nA1,S1\n", name="meta.csv")
    with pytest.raises(InputError, match=r"ausentes.*condition.*is_blank"):
        read_metadata(path)


def test_read_metadata_invalid_is_blank(tmp_path):
    path = write(tmp_path, "well,strain,condition,is_blank\nA1,S1,LB,talvez\n", name="meta.csv")
    with pytest.raises(InputError, match=r"linha 2, coluna 'is_blank'.*talvez"):
        read_metadata(path)


def test_read_metadata_duplicate_well(tmp_path):
    path = write(tmp_path, (
        "well,strain,condition,is_blank\nA1,S1,LB,false\nA1,S2,LB,false\n"
    ), name="meta.csv")
    with pytest.raises(InputError, match=r"'A1' repetido"):
        read_metadata(path)


def test_read_metadata_empty_well_id(tmp_path):
    path = write(tmp_path, "well,strain,condition,is_blank\n,S1,LB,false\n", name="meta.csv")
    with pytest.raises(InputError, match="identificador de poço vazio"):
        read_metadata(path)


# ---------------------------------------------------------------- alinhamento

def _meta(rows):
    return pd.DataFrame(rows, columns=["well", "strain", "condition", "is_blank"])


def test_align_metadata_all_matching_has_no_warnings():
    meta = _meta([("A1", "S1", "LB", False), ("H1", "", "meio", True)])
    aligned, warnings = align_metadata(["A1", "H1"], meta)
    assert warnings == []
    assert list(aligned["well"]) == ["A1", "H1"]
    assert list(aligned["is_blank"]) == [False, True]


def test_align_metadata_follows_plate_order():
    meta = _meta([("H1", "", "meio", True), ("A1", "S1", "LB", False)])
    aligned, _ = align_metadata(["A1", "H1"], meta)
    assert list(aligned["well"]) == ["A1", "H1"]
    assert aligned.loc[0, "strain"] == "S1"


def test_well_without_metadata_gets_empty_fields_and_warning():
    meta = _meta([("A1", "S1", "LB", False)])
    aligned, warnings = align_metadata(["A1", "A2"], meta)
    assert list(aligned["well"]) == ["A1", "A2"]
    row = aligned.loc[aligned["well"] == "A2"].iloc[0]
    assert row["strain"] == "" and row["condition"] == ""
    assert not row["is_blank"]
    assert len(warnings) == 1 and "A2" in warnings[0]


def test_metadata_well_without_measurements_gives_warning_and_is_dropped():
    meta = _meta([("A1", "S1", "LB", False), ("Z9", "S3", "LB", False)])
    aligned, warnings = align_metadata(["A1"], meta)
    assert list(aligned["well"]) == ["A1"]
    assert len(warnings) == 1 and "Z9" in warnings[0]