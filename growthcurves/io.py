"""Leitura e validação do CSV de leitor de placas (formato largo) e dos metadados."""

"""Erros de formato do arquivo levantam ``InputError``; o CLI captura essa exceção, mostra a mensagem e encerra com código de saída diferente de zero. Problemas que não impedem a análise (ex.: poço sem metadados) viram avisos.
"""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMNS = ("time_h", "time")
METADATA_COLUMNS = ("well", "strain", "condition", "is_blank")

_HHMMSS = re.compile(r"^([0-9]+):([0-5][0-9]):([0-5][0-9])$")
_NUMBER = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_BOOL = {"true": True, "false": False}


class InputError(Exception):
    """Erro nos dados de entrada que interrompe a execução."""


@dataclass(frozen=True)
class PlateData:
    """Resultado da leitura do arquivo de medidas.

    long            : DataFrame (well, time_h, od), sem as células vazias
    wells           : identificadores dos poços, na ordem das colunas do arquivo
    time_source     : coluna de tempo usada no arquivo ("time_h" ou "time")
    n_missing       : total de células vazias descartadas
    missing_by_well : {poço: nº de células vazias}, só para poços com alguma
    """

    long: pd.DataFrame
    wells: list
    time_source: str
    n_missing: int
    missing_by_well: dict


# ---------------------------------------------------------------- tempo e números

def parse_hhmmss(value: str, line: int | None = None) -> float:
    """Converte 'hh:mm:ss' em horas decimais (01:30:00 -> 1.5; horas podem passar de 24)."""
    match = _HHMMSS.match(value.strip())
    if match is None:
        where = f"linha {line}, " if line is not None else ""
        raise InputError(f"{where}valor de tempo fora do formato hh:mm:ss: {value!r}")
    h, m, s = (int(g) for g in match.groups())
    return (3600 * h + 60 * m + s) / 3600


def _to_float(value: str, line: int, column: str) -> float:
    if _NUMBER.match(value):
        x = float(value)
        if math.isfinite(x):
            return x
    raise InputError(f"valor não numérico em linha {line}, coluna '{column}': {value!r}")


# ---------------------------------------------------------------- leitura básica

def _read_csv_rows(path):
    """Lê o CSV com o módulo csv; devolve (cabeçalho, [(nº da linha, campos), ...]).

    Linhas totalmente em branco são ignoradas, mas contadas na numeração.
    """
    path = Path(path)
    header = None
    rows = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                if header is None:
                    header = [c.strip() for c in row]
                else:
                    rows.append((reader.line_num, row))
    except FileNotFoundError:
        raise InputError(f"arquivo não encontrado: {path}") from None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise InputError(f"não foi possível ler {path}: {exc}") from None
    if header is None:
        raise InputError(f"arquivo vazio: {path}")
    return header, rows


def _check_header_names(header, what: str) -> None:
    """Nomes de coluna não vazios e únicos (usa o cabeçalho bruto, não o do pandas)."""
    for position, name in enumerate(header, start=1):
        if name == "":
            raise InputError(f"{what}: nome de coluna vazio na posição {position}")
    duplicated = sorted({h for h in header if header.count(h) > 1})
    if duplicated:
        raise InputError(
            f"{what}: colunas duplicadas no cabeçalho ({', '.join(duplicated)}); "
            "os dados são ambíguos"
        )


def _check_row_lengths(header, rows, what: str) -> None:
    for line, row in rows:
        if len(row) != len(header):
            raise InputError(
                f"{what}: linha {line} tem {len(row)} campos, "
                f"mas o cabeçalho tem {len(header)}"
            )


# ---------------------------------------------------------------- medidas

def read_plate(path) -> PlateData:
    """Lê o CSV largo, valida e converte para o formato longo (well, time_h, od)."""
    header, rows = _read_csv_rows(path)

    if len(header) == 1 and ";" in header[0]:
        raise InputError(
            "o arquivo parece usar ponto e vírgula como separador; use vírgula"
        )
    _check_header_names(header, "arquivo de medidas")

    time_cols = [c for c in TIME_COLUMNS if c in header]
    if len(time_cols) != 1:
        found = ", ".join(time_cols) if time_cols else "nenhuma"
        raise InputError(
            "exatamente uma coluna de tempo (time_h ou time) é obrigatória; "
            f"encontradas: {found}"
        )
    time_name = time_cols[0]
    wells = [h for h in header if h not in TIME_COLUMNS]
    if not wells:
        raise InputError("nenhuma coluna de poço além da coluna de tempo")
    if not rows:
        raise InputError("arquivo de medidas sem nenhuma linha de dados")
    _check_row_lengths(header, rows, "arquivo de medidas")

    lines = [line for line, _ in rows]
    columns = {name: [row[i].strip() for _, row in rows] for i, name in enumerate(header)}

    # --- eixo do tempo (time_h ou time -> horas decimais)
    raw_time = columns[time_name]
    if time_name == "time":
        hours = np.array([parse_hhmmss(v, line) for v, line in zip(raw_time, lines)])
    else:
        hours = np.array([_to_float(v, line, "time_h") for v, line in zip(raw_time, lines)])

    negative = np.flatnonzero(hours < 0)
    if negative.size:
        i = int(negative[0])
        raise InputError(f"tempo negativo na linha {lines[i]}: {raw_time[i]!r}")
    not_increasing = np.flatnonzero(np.diff(hours) <= 0)
    if not_increasing.size:
        i = int(not_increasing[0]) + 1
        raise InputError(
            f"eixo do tempo inválido: tempo repetido ou decrescente na linha {lines[i]} "
            f"({raw_time[i]!r} após {raw_time[i - 1]!r}); "
            "o tempo deve ser estritamente crescente"
        )

    # --- OD por poço (célula vazia = medida ausente)
    wide = {"time_h": hours}
    missing_by_well = {}
    for well in wells:
        values = np.full(len(rows), np.nan)
        for k, (cell, line) in enumerate(zip(columns[well], lines)):
            if cell != "":
                values[k] = _to_float(cell, line, well)
        n_missing = int(np.isnan(values).sum())
        if n_missing:
            missing_by_well[well] = n_missing
        wide[well] = values

    long = (
        pd.DataFrame(wide)
        .melt(id_vars="time_h", var_name="well", value_name="od")
        .dropna(subset=["od"])
        .reset_index(drop=True)[["well", "time_h", "od"]]
    )
    return PlateData(
        long=long,
        wells=wells,
        time_source=time_name,
        n_missing=sum(missing_by_well.values()),
        missing_by_well=missing_by_well,
    )


# ---------------------------------------------------------------- metadados

def read_metadata(path) -> pd.DataFrame:
    """Lê os metadados: colunas well, strain, condition, is_blank (bool)."""
    header, rows = _read_csv_rows(path)
    _check_header_names(header, "metadados")
    absent = [c for c in METADATA_COLUMNS if c not in header]
    if absent:
        raise InputError(f"metadados: colunas obrigatórias ausentes: {', '.join(absent)}")
    _check_row_lengths(header, rows, "metadados")

    idx = {c: header.index(c) for c in METADATA_COLUMNS}
    records = []
    first_line = {}
    for line, row in rows:
        well = row[idx["well"]].strip()
        if well == "":
            raise InputError(f"metadados, linha {line}: identificador de poço vazio")
        if well in first_line:
            raise InputError(
                f"metadados: poço '{well}' repetido (linhas {first_line[well]} e {line})"
            )
        first_line[well] = line
        raw_blank = row[idx["is_blank"]]
        blank = _BOOL.get(raw_blank.strip().lower())
        if blank is None:
            raise InputError(
                f"metadados, linha {line}, coluna 'is_blank': "
                f"esperado true ou false, encontrado {raw_blank!r}"
            )
        records.append({
            "well": well,
            "strain": row[idx["strain"]].strip(),
            "condition": row[idx["condition"]].strip(),
            "is_blank": blank,
        })
    return pd.DataFrame(records, columns=list(METADATA_COLUMNS))


def align_metadata(wells, metadata: pd.DataFrame):
    """Alinha os metadados aos poços do arquivo de medidas.

    Devolve (metadados com uma linha por poço, na ordem do arquivo; lista de avisos).
    Poço sem metadados fica com campos vazios e is_blank = False.
    """
    warnings = []
    by_well = metadata.set_index("well")
    records = []
    for well in wells:
        if well in by_well.index:
            row = by_well.loc[well]
            records.append({"well": well, "strain": row["strain"],
                            "condition": row["condition"], "is_blank": bool(row["is_blank"])})
        else:
            warnings.append(
                f"poço '{well}' sem entrada nos metadados; processado com campos vazios"
            )
            records.append({"well": well, "strain": "", "condition": "", "is_blank": False})
    for well in metadata["well"]:
        if well not in wells:
            warnings.append(f"poço '{well}' presente nos metadados, mas ausente nas medidas")
    return pd.DataFrame(records, columns=list(METADATA_COLUMNS)), warnings