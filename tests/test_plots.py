import numpy as np
import pytest

from growthcurves.fit import FitResult, fit_all, fit_well
from growthcurves.models import MODELS
from growthcurves.plots import (
    annotation_text,
    build_figure,
    plot_all,
    plot_filename,
    plot_well,
    title_text,
)
from growthcurves.preprocess import WellSeries, preprocess
from growthcurves.synthetic import DEFAULT_OD0, default_plate, time_grid

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def clean_series(well="W", model="logistic", A=2.0, mu=0.5, lam=3.0):
    t = time_grid()
    y = MODELS[model](t, A, mu, lam)
    return WellSeries(well, t, DEFAULT_OD0 * np.exp(y), y, DEFAULT_OD0)


def empty_series(well="E"):
    return WellSeries(well, np.array([]), np.array([]), np.array([]), float("nan"))


def flat_series(well="F"):
    t = time_grid()
    return WellSeries(well, t, np.full_like(t, 0.2), np.zeros_like(t), 0.2)


def assert_is_png(path):
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE
    assert len(data) > 1000


# ---------------------------------------------------------------- nomes de arquivo

@pytest.mark.parametrize("well, expected", [
    ("A1", "A1.png"),
    ("H12", "H12.png"),
    ("A/1", "A_1.png"),
    ("A 1", "A_1.png"),
    ("poço-1", "po_o-1.png"),
    ("..\\x", ".._x.png"),
])
def test_plot_filename(well, expected):
    assert plot_filename(well) == expected


def test_plot_filename_never_contains_path_separators():
    name = plot_filename("a/b\\c:d*e?f")
    assert "/" not in name and "\\" not in name and ":" not in name


# ---------------------------------------------------------------- textos

def test_annotation_has_model_and_all_parameters():
    result = fit_well(clean_series())
    text = annotation_text(result)
    assert f"modelo: {result.model}" in text
    assert f"μ = {result.mu_max:.3f} h⁻¹" in text
    assert f"λ = {result.lag_h:.2f} h" in text
    assert f"A = {result.A:.3f}" in text
    assert f"K = {result.K:.3f}" in text
    assert f"t_d = {result.doubling_time_h:.2f} h" in text
    assert "R² = " in text and "AIC = " in text


def test_annotation_is_empty_without_fit():
    assert annotation_text(FitResult("F", "no_growth", reason="x")) == ""
    assert annotation_text(FitResult("F", "fit_failed", reason="x")) == ""


def test_title_for_ok_and_for_problem_wells():
    ok = fit_well(clean_series(well="G1"))
    assert title_text(ok) == f"G1 — ajuste {ok.model}"
    failed = FitResult("S1", "fit_failed", reason="menos de 6 pontos válidos (5)")
    assert title_text(failed) == "S1 — fit_failed: menos de 6 pontos válidos (5)"
    long_reason = FitResult("S1", "fit_failed", reason="x" * 30 + " " + "y" * 80)
    assert "\n" in title_text(long_reason)


# ---------------------------------------------------------------- figura

def test_figure_plots_observed_data_and_fitted_curve():
    series = clean_series()
    result = fit_well(series)
    fig = build_figure(series, result)
    ax = fig.axes[0]
    observed, fitted = ax.lines
    np.testing.assert_array_equal(observed.get_xdata(), series.time_h)
    np.testing.assert_array_equal(observed.get_ydata(), series.y)
    expected = MODELS[result.model](fitted.get_xdata(), result.A, result.mu_max, result.lag_h)
    np.testing.assert_allclose(fitted.get_ydata(), expected)
    assert fitted.get_xdata()[0] == series.time_h[0]
    assert fitted.get_xdata()[-1] == series.time_h[-1]


def test_figure_annotates_parameters_inside_the_axes():
    series = clean_series()
    result = fit_well(series)
    ax = build_figure(series, result).axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert annotation_text(result) in texts


def test_figure_labels_and_title():
    series = clean_series(well="G1")
    result = fit_well(series)
    ax = build_figure(series, result).axes[0]
    assert ax.get_xlabel() == "Tempo (h)"
    assert ax.get_ylabel() == "ln(OD/OD0)"
    assert ax.get_title() == title_text(result)


def test_figure_for_no_growth_has_only_observed_data():
    series = flat_series()
    result = fit_well(series)
    assert result.status == "no_growth"
    ax = build_figure(series, result).axes[0]
    assert len(ax.lines) == 1
    assert len(ax.texts) == 0
    assert "no_growth" in ax.get_title()


def test_figure_for_empty_series_shows_message():
    series = empty_series()
    result = fit_well(series)
    ax = build_figure(series, result).axes[0]
    assert len(ax.lines) == 0
    assert [t.get_text() for t in ax.texts] == ["sem pontos válidos"]
    assert "fit_failed" in ax.get_title()


# ---------------------------------------------------------------- arquivos

@pytest.mark.parametrize("make_series", [clean_series, flat_series, empty_series])
def test_plot_well_writes_a_valid_png_for_every_status(tmp_path, make_series):
    series = make_series()
    path = plot_well(series, fit_well(series), tmp_path / "nested" / "W.png")
    assert path.exists()
    assert_is_png(path)


def test_plot_all_writes_one_png_per_analyzed_well(tmp_path):
    series = {"G": clean_series(well="G"), "F": flat_series(well="F"), "E": empty_series("E")}
    results = fit_all(series)
    paths = plot_all(series, results, tmp_path / "plots")
    assert [p.name for p in paths] == ["G.png", "F.png", "E.png"]
    for p in paths:
        assert p.parent == tmp_path / "plots"
        assert_is_png(p)


def test_plot_all_avoids_overwriting_when_names_collide(tmp_path):
    series = {"A/1": clean_series(well="A/1"), "A_1": clean_series(well="A_1")}
    paths = plot_all(series, fit_all(series), tmp_path)
    assert [p.name for p in paths] == ["A_1.png", "A_1_2.png"]
    assert all(p.exists() for p in paths)


def test_plot_all_on_pipeline_output_skips_blank_wells(tmp_path):
    plate, meta = default_plate()
    long = (plate.melt(id_vars="time_h", var_name="well", value_name="od")
            .dropna(subset=["od"]).reset_index(drop=True)[["well", "time_h", "od"]])
    pre = preprocess(long, meta)
    results = fit_all(pre.series)
    paths = plot_all(pre.series, results, tmp_path / "plots")
    names = sorted(p.name for p in paths)
    assert names == ["A1.png", "A2.png", "A3.png", "B1.png"]
    assert not (tmp_path / "plots" / "H1.png").exists()


def test_plotting_does_not_modify_inputs(tmp_path):
    series = clean_series()
    result = fit_well(series)
    before = (series.time_h.copy(), series.y.copy())
    plot_well(series, result, tmp_path / "W.png")
    np.testing.assert_array_equal(series.time_h, before[0])
    np.testing.assert_array_equal(series.y, before[1])