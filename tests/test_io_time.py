import pandas as pd
import pytest

from growthcurves.io import InputError, parse_hhmmss, read_plate
from growthcurves.synthetic import default_plate, write_plate_csv


def write(tmp_path, text, name="plate.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- parse_hhmmss

@pytest.mark.parametrize("text, hours", [
    ("00:00:00", 0.0),
    ("01:30:00", 1.5),
    ("25:00:00", 25.0),
    ("00:15:00", 0.25),
    ("100:30:00", 100.5),
    ("00:00:01", 1 / 3600),
    ("23:59:59", 23 + 59 / 60 + 59 / 3600),
    (" 01:30:00 ", 1.5),
])
def test_parse_hhmmss_valid(text, hours):
    assert parse_hhmmss(text) == pytest.approx(hours)


@pytest.mark.parametrize("text", [
    "00:75:00",      # minutos > 59
    "00:00:60",      # segundos > 59
    "01:30",         # formato ambíguo (hh:mm)
    "1.02:00:00",    # com dias
    "01:30:00:00",   # campos demais
    "01:3:00",       # minutos com um dígito
    "-01:00:00",     # negativo
    "abc",
    "",
    "12:00:00 pm",
])
def test_parse_hhmmss_invalid(text):
    with pytest.raises(InputError):
        parse_hhmmss(text)


def test_parse_hhmmss_error_shows_value_and_line():
    with pytest.raises(InputError, match=r"linha 5.*00:75:00"):
        parse_hhmmss("00:75:00", line=5)


def test_parse_hhmmss_exact_for_half_hour_steps():
    assert parse_hhmmss("00:30:00") == 0.5
    assert parse_hhmmss("01:30:00") == 1.5


# ---------------------------------------------------------------- leitura de arquivo

def test_read_time_column_converts_to_decimal_hours(tmp_path):
    path = write(tmp_path, "time,A1\n00:00:00,0.1\n01:30:00,0.2\n25:00:00,0.3\n")
    data = read_plate(path)
    assert data.time_source == "time"
    assert list(data.long["time_h"]) == [0.0, 1.5, 25.0]
    assert "time" not in data.long.columns
    assert data.wells == ["A1"]


def test_read_time_h_column(tmp_path):
    path = write(tmp_path, "time_h,A1\n0,0.1\n1.5,0.2\n")
    data = read_plate(path)
    assert data.time_source == "time_h"
    assert list(data.long["time_h"]) == [0.0, 1.5]


def test_time_and_time_h_files_give_identical_long_table(tmp_path):
    plate, _ = default_plate()
    path_h = write_plate_csv(plate, tmp_path / "h.csv", time_format="time_h")
    path_t = write_plate_csv(plate, tmp_path / "t.csv", time_format="time")
    data_h = read_plate(path_h)
    data_t = read_plate(path_t)
    assert data_h.time_source == "time_h"
    assert data_t.time_source == "time"
    assert data_h.wells == data_t.wells
    pd.testing.assert_frame_equal(data_h.long, data_t.long)


def test_invalid_time_in_file_reports_line_and_value(tmp_path):
    path = write(tmp_path, "time,A1\n00:00:00,0.1\n00:75:00,0.2\n")
    with pytest.raises(InputError, match=r"linha 3.*00:75:00"):
        read_plate(path)


def test_ambiguous_hhmm_in_file_is_rejected(tmp_path):
    path = write(tmp_path, "time,A1\n00:00:00,0.1\n01:30,0.2\n")
    with pytest.raises(InputError, match=r"linha 3.*01:30"):
        read_plate(path)


def test_time_with_days_in_file_is_rejected(tmp_path):
    path = write(tmp_path, "time,A1\n00:00:00,0.1\n1.02:00:00,0.2\n")
    with pytest.raises(InputError, match="1.02:00:00"):
        read_plate(path)


def test_empty_time_cell_is_rejected(tmp_path):
    path = write(tmp_path, "time,A1\n00:00:00,0.1\n,0.2\n")
    with pytest.raises(InputError, match="linha 3"):
        read_plate(path)


@pytest.mark.parametrize("times", [
    ["01:00:00", "01:00:00"],              # repetido
    ["02:00:00", "01:00:00"],              # decrescente
    ["00:00:00", "01:00:00", "00:59:59"],  # volta no tempo
])
def test_time_must_be_strictly_increasing_after_conversion(tmp_path, times):
    body = "".join(f"{t},0.1\n" for t in times)
    path = write(tmp_path, "time,A1\n" + body)
    with pytest.raises(InputError, match="estritamente crescente"):
        read_plate(path)


def test_hhmmss_crossing_hour_boundary_is_increasing(tmp_path):
    path = write(tmp_path, "time,A1\n00:59:59,0.1\n01:00:00,0.2\n")
    data = read_plate(path)
    assert data.long["time_h"].is_monotonic_increasing