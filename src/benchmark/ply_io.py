from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

TLS2TREES_COLUMNS = ("x", "y", "z", "n_z", "label")
TLS2TREES_DTYPE = np.dtype([(name, "<f8") for name in TLS2TREES_COLUMNS])


@dataclass(frozen=True)
class PlyHeader:
    format: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    header_bytes: int

    @property
    def columns(self) -> list[str]:
        return [name for name, _ in self.properties]


def read_ply_header(path: str | Path) -> PlyHeader:
    ply_path = Path(path).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file does not exist: {ply_path}")

    file_format: str | None = None
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element: str | None = None

    with ply_path.open("rb") as handle:
        first_line = handle.readline()
        if first_line.strip() != b"ply":
            raise ValueError(f"Not a PLY file: {ply_path}")

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"PLY header has no end_header marker: {ply_path}")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"PLY header is not ASCII: {ply_path}") from exc

            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                file_format = parts[1]
            elif parts[0] == "element" and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) != 3 or parts[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {ply_path}: {line}")
                if parts[1] not in _PLY_DTYPES:
                    raise ValueError(f"Unsupported PLY property type {parts[1]!r} in {ply_path}")
                properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                header_bytes = handle.tell()
                break

    if file_format not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported or missing PLY format in {ply_path}: {file_format}")
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex count: {ply_path}")
    if not properties:
        raise ValueError(f"PLY header has no vertex properties: {ply_path}")

    return PlyHeader(
        format=file_format,
        vertex_count=vertex_count,
        properties=tuple(properties),
        header_bytes=header_bytes,
    )


def _structured_dtype(header: PlyHeader) -> np.dtype:
    endian = ">" if header.format == "binary_big_endian" else "<"
    return np.dtype(
        [(name, endian + _PLY_DTYPES[property_type]) for name, property_type in header.properties]
    )


def read_ply_vertices(
    path: str | Path,
    columns: Sequence[str] | None = None,
) -> tuple[PlyHeader, dict[str, np.ndarray]]:
    ply_path = Path(path).expanduser().resolve()
    header = read_ply_header(ply_path)
    requested = list(columns) if columns is not None else header.columns
    missing = sorted(set(requested) - set(header.columns))
    if missing:
        raise ValueError(f"PLY file {ply_path} is missing column(s): {', '.join(missing)}")

    dtype = _structured_dtype(header)
    if header.vertex_count == 0:
        records = np.empty(0, dtype=dtype)
    elif header.format == "ascii":
        records = np.loadtxt(
            ply_path,
            dtype=dtype,
            skiprows=sum(1 for _ in _iter_header_lines(ply_path)),
            max_rows=header.vertex_count,
        )
        records = np.atleast_1d(records)
    else:
        with ply_path.open("rb") as handle:
            handle.seek(header.header_bytes)
            records = np.fromfile(handle, dtype=dtype, count=header.vertex_count)

    if len(records) != header.vertex_count:
        raise ValueError(
            f"PLY vertex count mismatch for {ply_path}: "
            f"header={header.vertex_count}, read={len(records)}"
        )
    return header, {name: np.asarray(records[name]) for name in requested}


def _iter_header_lines(path: Path) -> Iterable[bytes]:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            yield line
            if line.strip() == b"end_header":
                return


def write_tls2trees_ply(
    output_path: str | Path,
    vertex_count: int,
    chunks: Iterable[Mapping[str, np.ndarray]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a binary PLY accepted by TLS2trees without buffer/fn columns."""

    path = Path(output_path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment source tree-seg-benchmark\n"
        f"element vertex {vertex_count}\n"
        "property float64 x\n"
        "property float64 y\n"
        "property float64 z\n"
        "property float64 n_z\n"
        "property float64 label\n"
        "end_header\n"
    ).encode("ascii")

    written = 0
    try:
        with temporary_path.open("wb") as handle:
            handle.write(header)
            for chunk in chunks:
                missing = sorted(set(TLS2TREES_COLUMNS) - set(chunk))
                if missing:
                    raise ValueError(f"PLY chunk is missing column(s): {', '.join(missing)}")
                chunk_size = len(np.asarray(chunk["x"]))
                records = np.empty(chunk_size, dtype=TLS2TREES_DTYPE)
                for name in TLS2TREES_COLUMNS:
                    values = np.asarray(chunk[name])
                    if len(values) != chunk_size:
                        raise ValueError(f"PLY chunk column {name!r} has inconsistent length")
                    records[name] = values
                handle.write(records.tobytes())
                written += chunk_size

        if written != vertex_count:
            raise ValueError(f"PLY point count mismatch: expected {vertex_count}, wrote {written}")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return path
