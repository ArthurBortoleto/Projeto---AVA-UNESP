import numpy as np
import pandas as pd
import pytest

from growthcurves.io import InputError
from growthcurves.preprocess import (
    N_BASELINE,
    blank_correct,
    log_transform,
    moving_median,
    preprocess,
)
from growthcurves.synthetic import default_plate


def make_long(rows):
    """rows: lista de (poço, tempo, od)."""
    return pd.DataFrame(rows, columns=["well", "time_h", "od"])


def make_meta(wells, blanks=()):
    return pd.DataFrame({
        "well": list(wells),
        "strain": "S1",
        "condition": "LB",
        "is_blank": [w in blanks for w in wells],
    })


def plate_to_long(plate):
    return (plate.melt(id_vars="time_h", var_name="well", value_name="od")
            .dropna(subset=["od"]).reset_index(drop=True)[["well", "time_h", "od"]])


# ---------------------------------------------------------------- correção de branco

def test_blank_correct_subtracts_mean_of_blanks_at_same_time():
    long = make_long([
        ("A1", 0.0, 1.0), ("A1", 1.0, 2.0),
        ("H1", 0.0, 0.1), ("H1", 1.0, 0.2),
        ("H2", 0.0, 0.3), ("H2", 1.0, 0.4),
    ])
    out, dropped, warnings = blank_correct(long, ["H1", "H2"])
    assert list(out["well"]) == ["A1", "A1"]
    np.testing.assert_allclose(out["od"], [0.8, 1.7])
    assert dropped == {}
    assert warnings == []


def test_blank_correct_uses_only_blanks_available_at_each_time():
    long = make_long([
        ("A1", 0.0, 1.0), ("A1", 1.0, 2.0),
        ("H1", 0.0, 0.1), ("H1", 1.0, 0.2),
        ("H2", 0.0, 0.3),                      # H2 sem medida em t = 1
    ])
    out, _, _ = blank_correct(long, ["H1", "H2"])
    np.testing.assert_allclose(out["od"], [0.8, 1.8])   # t=1: só H1 (0.2)


def test_blank_correct_drops_points_without_any_blank_at_that_time():
    long = make_long([
        ("A1", 0.0, 1.0), ("A1", 1.0, 2.0), ("A1", 2.0, 3.0),
        ("H1", 0.0, 0.1), ("H1", 1.0, 0.2),    # sem branco em t = 2
    ])
    out, dropped, warnings = blank_correct(long, ["H1"])
    assert list(out["time_h"]) == [0.0, 1.0]
    assert dropped == {"A1": 1}
    assert len(warnings) == 1 and "sem medida de branco" in warnings[0]


def test_blank_correct_without_blank_wells_warns_and_keeps_od():
    long = make_long([("A1", 0.0, 1.0), ("A1", 1.0, 2.0)])
    out, dropped, warnings = blank_correct(long, [])
    np.testing.assert_array_equal(out["od"], [1.0, 2.0])
    assert dropped == {}
    assert len(warnings) == 1 and "nenhum poço-branco" in warnings[0]


def test_blank_correct_with_blank_wells_that_have_no_measurements():
    long = make_long([("A1", 0.0, 1.0), ("A1", 1.0, 2.0)])
    out, _, warnings = blank_correct(long, ["H1"])
    np.testing.assert_array_equal(out["od"], [1.0, 2.0])
    assert len(warnings) == 1 and "sem nenhuma medida" in warnings[0]


def test_blank_correct_does_not_modify_input():
    long = make_long([("A1", 0.0, 1.0), ("H1", 0.0, 0.1)])
    before = long.copy()
    blank_correct(long, ["H1"])
    pd.testing.assert_frame_equal(long, before)


# ---------------------------------------------------------------- suavização

def test_moving_median_window_1_is_identity_copy():
    values = np.array([1.0, 5.0, 2.0])
    out = moving_median(values, 1)
    np.testing.assert_array_equal(out, values)
    out[0] = 99.0
    assert values[0] == 1.0


def test_moving_median_removes_isolated_spike():
    out = moving_median([1, 1, 10, 1, 1], 3)
    np.testing.assert_array_equal(out, [1, 1, 1, 1, 1])


def test_moving_median_shrinks_window_at_the_edges():
    out = moving_median([1, 5, 2, 3, 4], 3)
    np.testing.assert_allclose(out, [3.0, 2.0, 3.0, 3.0, 3.5])


def test_moving_median_keeps_length_and_handles_empty():
    assert moving_median(np.arange(10.0), 5).shape == (10,)
    assert moving_median([], 3).size == 0


def test_moving_median_does_not_modify_input():
    values = np.array([1.0, 10.0, 1.0])
    moving_median(values, 3)
    np.testing.assert_array_equal(values, [1.0, 10.0, 1.0])


@pytest.mark.parametrize("window", [0, -1, 2, 4, 3.0, "3", True])
def test_invalid_window_raises(window):
    with pytest.raises(InputError, match="janela de suavização inválida"):
        moving_median([1.0, 2.0, 3.0], window)


# ---------------------------------------------------------------- transformação

def test_log_transform_y_is_ln_od_over_od0():
    t = np.arange(6) * 0.5
    od = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.8])
    s = log_transform("A1", t, od)
    assert s.od0 == pytest.approx(0.1)
    np.testing.assert_allclose(s.y, np.log([1, 1, 1, 2, 4, 8]), atol=1e-12)
    assert s.n_points == 6 and s.n_discarded == 0


def test_od0_is_mean_of_first_n_baseline_points():
    assert N_BASELINE == 3
    od = np.array([0.1, 0.2, 0.3, 0.4])
    s = log_transform("A1", np.arange(4.0), od)
    assert s.od0 == pytest.approx(0.2)
    assert s.y[0] == pytest.approx(np.log(0.1 / 0.2))
    s2 = log_transform("A1", np.arange(4.0), od, n_baseline=2)
    assert s2.od0 == pytest.approx(0.15)


def test_nonpositive_od_is_discarded_and_counted():
    t = np.arange(6.0)
    od = np.array([0.1, 0.0, 0.1, -0.05, 0.2, 0.4])
    s = log_transform("A1", t, od)
    assert s.n_nonpositive == 2
    np.testing.assert_array_equal(s.time_h, [0.0, 2.0, 4.0, 5.0])
    assert s.od.shape == s.y.shape == s.time_h.shape
    assert s.od0 == pytest.approx((0.1 + 0.1 + 0.2) / 3)   # só pontos mantidos


def test_all_nonpositive_gives_empty_series():
    s = log_transform("A1", np.arange(3.0), np.array([0.0, -0.1, -0.2]))
    assert s.n_points == 0
    assert s.n_nonpositive == 3
    assert np.isnan(s.od0)
    assert s.y.size == 0


# ---------------------------------------------------------------- pipeline

def test_preprocess_smoothing_happens_before_discarding_nonpositive():
    # Spike negativo isolado: a janela 3 o remove; sem suavização ele é descartado.
    rows = [("A1", float(i), od) for i, od in enumerate([0.5, 0.5, -0.1, 0.5, 0.5, 0.5])]
    meta = make_meta(["A1"])
    smoothed = preprocess(make_long(rows), meta, smooth_window=3).series["A1"]
    raw = preprocess(make_long(rows), meta, smooth_window=1).series["A1"]
    assert smoothed.n_nonpositive == 0 and smoothed.n_points == 6
    assert raw.n_nonpositive == 1 and raw.n_points == 5


def test_preprocess_excludes_blanks_from_series_and_lists_them():
    plate, meta = default_plate()
    result = preprocess(plate_to_long(plate), meta)
    assert list(result.series) == ["A1", "A2", "A3", "B1"]
    assert result.blank_wells == ["H1", "H2", "H3"]
    assert result.warnings == []


def test_preprocess_growth_well_reaches_amplitude():
    plate, meta = default_plate()
    s = preprocess(plate_to_long(plate), meta).series["A1"]   # A = 2.0
    assert s.od0 == pytest.approx(0.2, abs=0.03)
    assert s.y[-1] == pytest.approx(2.0, abs=0.3)
    assert s.n_points == len(plate) and s.n_discarded == 0


def test_preprocess_no_growth_well_stays_low():
    plate, meta = default_plate()
    s = preprocess(plate_to_long(plate), meta).series["B1"]
    assert s.y.max() < 0.5


def test_preprocess_without_blanks_warns():
    plate, meta = default_plate()
    plate = plate.drop(columns=["H1", "H2", "H3"])
    meta = meta[~meta["is_blank"]].reset_index(drop=True)
    result = preprocess(plate_to_long(plate), meta)
    assert result.blank_wells == []
    assert any("nenhum poço-branco" in w for w in result.warnings)
    assert result.series["A1"].n_points == len(plate)


def test_preprocess_well_without_any_valid_point_gets_empty_series():
    long = make_long([("A1", 0.0, 0.1), ("A1", 1.0, 0.2)])
    meta = make_meta(["A1", "Z1"])
    result = preprocess(long, meta)
    assert result.series["Z1"].n_points == 0
    assert np.isnan(result.series["Z1"].od0)


def test_preprocess_reports_points_dropped_for_missing_blank():
    long = make_long([
        ("A1", 0.0, 1.0), ("A1", 1.0, 2.0), ("A1", 2.0, 3.0),
        ("H1", 0.0, 0.1), ("H1", 1.0, 0.1),
    ])
    result = preprocess(long, make_meta(["A1", "H1"], blanks=["H1"]), smooth_window=1)
    assert result.series["A1"].n_no_blank == 1
    assert result.series["A1"].n_points == 2
    assert any("sem medida de branco" in w for w in result.warnings)


def test_preprocess_does_not_modify_inputs():
    plate, meta = default_plate()
    long = plate_to_long(plate)
    before_long, before_meta = long.copy(), meta.copy()
    preprocess(long, meta)
    pd.testing.assert_frame_equal(long, before_long)
    pd.testing.assert_frame_equal(meta, before_meta)


def test_preprocess_rejects_invalid_window():
    plate, meta = default_plate()
    with pytest.raises(InputError):
        preprocess(plate_to_long(plate), meta, smooth_window=2)


def test_preprocess_is_reproducible():
    plate, meta = default_plate()
    long = plate_to_long(plate)
    a = preprocess(long, meta).series["A2"]
    b = preprocess(long, meta).series["A2"]
    np.testing.assert_array_equal(a.y, b.y)