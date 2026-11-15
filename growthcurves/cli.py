"""Interface de linha de comando.

Uso:
    python -m growthcurves.cli --input data/synthetic/plate.csv \\
                               --metadata data/synthetic/metadata.csv

Códigos de saída: 0 = análise concluída (mesmo que alguns poços tenham falhado);
1 = erro nos dados de entrada, nos parâmetros ou na gravação dos resultados;
2 = uso incorreto da linha de comando (argparse).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .fit import DEFAULT_MIN_GROWTH, DEFAULT_MODELS, MIN_POINTS, fit_all, parse_models
from .io import InputError, align_metadata, read_metadata, read_plate
from .plots import plot_all
from .preprocess import N_BASELINE, preprocess
from .report import (
    build_parameters_table,
    build_qc_report,
    build_run_info,
    write_parameters,
    write_qc_report,
    write_run_info,
)

DEFAULT_OUTPUT_DIR = "results/"
DEFAULT_SMOOTH_WINDOW = 3
# Um ajuste de 3 parâmetros precisa de mais de 3 pontos.
_MIN_ALLOWED_POINTS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m growthcurves.cli",
        description="Análise automatizada de curvas de crescimento microbiano "
                    "a partir de dados de leitor de placas.",
    )
    parser.add_argument("--input", required=True,
                        help="CSV de medidas em formato largo (time_h ou time + uma coluna por poço)")
    parser.add_argument("--metadata", required=True,
                        help="CSV de metadados (well, strain, condition, is_blank)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="pasta de saída (padrão: %(default)s)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="modelos a ajustar, separados por vírgula (padrão: %(default)s)")
    parser.add_argument("--min-growth", type=float, default=DEFAULT_MIN_GROWTH,
                        help="aumento mínimo de ln(OD/OD0) para considerar que houve "
                             "crescimento (padrão: %(default)s)")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                        help="janela da mediana móvel; inteiro ímpar, 1 desativa "
                             "(padrão: %(default)s)")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS,
                        help="mínimo de pontos válidos por poço para ajustar "
                             "(padrão: %(default)s)")
    parser.add_argument("--version", action="version", version=f"growthcurves {__version__}")
    return parser


def run_analysis(input_path, metadata_path, output_dir, models=DEFAULT_MODELS,
                 min_growth=DEFAULT_MIN_GROWTH, smooth_window=DEFAULT_SMOOTH_WINDOW,
                 min_points=MIN_POINTS) -> dict:
    """Roda a análise completa e grava as saídas em ``output_dir``.

    Tudo é calculado antes de gravar qualquer arquivo: se a leitura ou os parâmetros
    forem inválidos (``InputError``), nada é criado na pasta de saída.
    """
    input_path, metadata_path, output_dir = Path(input_path), Path(metadata_path), Path(output_dir)

    models = parse_models(models)
    if min_growth < 0:
        raise InputError(f"--min-growth deve ser maior ou igual a 0 (recebido: {min_growth})")
    if min_points < _MIN_ALLOWED_POINTS:
        raise InputError(
            f"--min-points deve ser pelo menos {_MIN_ALLOWED_POINTS} (recebido: {min_points})"
        )

    data = read_plate(input_path)
    metadata = read_metadata(metadata_path)
    aligned, io_warnings = align_metadata(data.wells, metadata)
    pre = preprocess(data.long, aligned, smooth_window=smooth_window)
    results = fit_all(pre.series, models, min_growth, min_points)

    table = build_parameters_table(results, aligned)
    qc_text = build_qc_report(results, pre, data, io_warnings)
    info = build_run_info(input_path, metadata_path, {
        "models": models,
        "min_growth": min_growth,
        "smooth_window": smooth_window,
        "min_points": min_points,
        "n_baseline": N_BASELINE,
        "output_dir": output_dir,
    })

    write_parameters(table, output_dir / "parameters.csv")
    plot_all(pre.series, results, output_dir / "plots")
    write_qc_report(qc_text, output_dir / "qc_report.md")
    write_run_info(info, output_dir / "run_info.json")

    return {
        "counts": Counter(r.status for r in results),
        "n_analyzed": len(results),
        "n_blank": len(pre.blank_wells),
        "n_warnings": len(io_warnings) + len(pre.warnings),
        "output_dir": output_dir,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_analysis(
            args.input, args.metadata, args.output_dir, models=args.models,
            min_growth=args.min_growth, smooth_window=args.smooth_window,
            min_points=args.min_points,
        )
    except InputError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"erro: não foi possível gravar os resultados: {exc}", file=sys.stderr)
        return 1

    counts = summary["counts"]
    print(
        f"Análise concluída: {summary['n_analyzed']} poços analisados "
        f"({counts['ok']} ok, {counts['no_growth']} sem crescimento, "
        f"{counts['fit_failed']} com falha no ajuste); "
        f"{summary['n_blank']} poços-branco."
    )
    if summary["n_warnings"]:
        print(f"Avisos: {summary['n_warnings']} (veja qc_report.md)")
    print(f"Resultados em: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    # Evita UnicodeEncodeError em consoles que não usam UTF-8 (ex.: cmd do Windows).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main())