import numpy as np
import pandas as pd
import pytest

from growthcurves.models import MODELS
from growthcurves.synthetic import (
    DEFAULT_OD0,
    default_plate,
    hours_to_hhmmss,
    make_plate,
    simulate_od,
    time_grid,
    write_metadata_csv,
    write_plate_csv,
)


def test_time_grid_default():
    t = time_grid()
    assert t[0] == 0.0
    assert t[-1] == 24.0
    assert len(t) == 49
    np.testing.assert_allclose(np.diff(t), 0.5)


def test_simulate_od_without_noise_matches_model():
    t = time_grid()
    od = simulate_od("logistic", t, 2.0, 0.5, 3.0, od0=0.1, noise_sd=0)
    expected = 0.1 * np.exp(MODELS["logistic"](t, 2.0, 0.5, 3.0))
    np.testing.assert_allclose(od, expected)


def test_simulate_od_same_seed_is_identical():
    t = time_grid()
    a = simulate_od("gompertz", t, 2.0, 0.4, 4.0, rng=np.random.default_rng(1))
    b = simulate_od("gompertz", t, 2.0, 0.4, 4.0, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_simulate_od_different_seed_differs():
    t = time_grid()
    a = simulate_od("gompertz", t, 2.0, 0.4, 4.0, rng=np.random.default_rng(1))
    b = simulate_od("gompertz", t, 2.0, 0.4, 4.0, rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_simulate_od_noise_sd_is_about_0_01():
    t = np.linspace(0, 24, 20000)
    clean = simulate_od("logistic", t, 2.0, 0.5, 3.0, noise_sd=0)
    noisy = simulate_od("logistic", t, 2.0, 0.5, 3.0, noise_sd=0.01,
                        rng=np.random.default_rng(0))
    assert np.std(noisy - clean) == pytest.approx(0.01, rel=0.05)


def test_simulate_od_unknown_model_raises():
    with pytest.raises(ValueError):
        simulate_od("richards", time_grid(), 2.0, 0.5, 3.0)


@pytest.mark.parametrize("hours, expected", [
    (0, "00:00:00"),
    (1.5, "01:30:00"),
    (25, "25:00:00"),
    (0.25, "00:15:00"),
    (100.5, "100:30:00"),
])
def test_hours_to_hhmmss(hours, expected):
    assert hours_to_hhmmss(hours) == expected


def test_default_plate_structure():
    plate, meta = default_plate()
    assert list(plate.columns) == ["time_h", "A1", "A2", "A3", "B1", "H1", "H2", "H3"]
    assert list(meta.columns) == ["well", "strain", "condition", "is_blank"]
    assert set(meta["well"]) == set(plate.columns) - {"time_h"}
    assert set(meta.loc[meta["is_blank"], "well"]) == {"H1", "H2", "H3"}
    assert plate["time_h"].is_monotonic_increasing


def test_default_plate_is_reproducible():
    p1, m1 = default_plate(seed=42)
    p2, m2 = default_plate(seed=42)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(m1, m2)


def test_blank_wells_sit_around_blank_level():
    plate, _ = default_plate()
    for well in ("H1", "H2", "H3"):
        assert plate[well].mean() == pytest.approx(0.04, abs=0.01)


def test_no_growth_well_stays_flat():
    plate, _ = default_plate()
    assert plate["B1"].max() - plate["B1"].min() < 0.1  # só ruído (sd = 0.01)


def test_growth_well_rises_well_above_blank():
    plate, _ = default_plate()
    assert plate["A1"].iloc[-1] > 5 * plate["H1"].mean()


def test_make_plate_without_blanks():
    plate, meta = make_plate({"A1": {"model": "logistic", "A": 2.0, "mu": 0.5, "lam": 3.0}})
    assert list(plate.columns) == ["time_h", "A1"]
    assert not meta["is_blank"].any()


def test_write_plate_csv_time_h(tmp_path):
    plate, _ = default_plate()
    path = write_plate_csv(plate, tmp_path / "plate.csv", time_format="time_h")
    back = pd.read_csv(path)
    assert "time_h" in back.columns and "time" not in back.columns
    pd.testing.assert_frame_equal(back, plate)


def test_write_plate_csv_time_hhmmss(tmp_path):
    plate, _ = default_plate()
    path = write_plate_csv(plate, tmp_path / "plate.csv", time_format="time")
    back = pd.read_csv(path, dtype={"time": str})
    assert "time" in back.columns and "time_h" not in back.columns
    assert back.columns[0] == "time"
    assert back["time"].iloc[0] == "00:00:00"
    assert back["time"].iloc[3] == "01:30:00"
    assert back["time"].iloc[-1] == "24:00:00"


def test_write_plate_csv_rejects_bad_format(tmp_path):
    plate, _ = default_plate()
    with pytest.raises(ValueError):
        write_plate_csv(plate, tmp_path / "x.csv", time_format="hh:mm")


def test_write_metadata_csv_uses_lowercase_booleans(tmp_path):
    _, meta = default_plate()
    path = write_metadata_csv(meta, tmp_path / "metadata.csv")
    text = path.read_text()
    assert "true" in text and "false" in text
    assert "True" not in text and "False" not in text


def test_default_od0_is_0_2_and_used_by_default():
    assert DEFAULT_OD0 == 0.2
    t = time_grid()
    od = simulate_od("logistic", t, 2.0, 0.5, 3.0, noise_sd=0)
    assert od[0] == pytest.approx(0.2 * np.exp(MODELS["logistic"](0.0, 2.0, 0.5, 3.0)))