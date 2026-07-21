"""Validate the isolated historical TLS2trees runtime on Barkla.

The published method code pins a 2021 PyTorch/PyG stack that cannot coexist
with the benchmark's Python 3.12 utility environment.  This validator keeps
that compatibility environment explicit and, on a GPU node, executes the
compiled PyG operators and bundled FSCT model before a benchmark is allowed to
use it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


EXPECTED_UPSTREAM_COMMIT = "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
EXPECTED_MODEL_SHA256 = (
    "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
)
EXPECTED_PYTHON = (3, 9)
EXPECTED_MARKER_SCHEMA = 2
EXPECTED_DETERMINISM_POLICY = (
    "seeded_best_effort_upstream_compatible_nondeterministic_cuda_scatter"
)
EXPECTED_CONDA_PACKAGES = {
    "cudatoolkit": {
        "version": "11.1.1",
        "build": "h6406543_8",
        "build_number": 8,
        "subdir": "linux-64",
        "md5": "4851e7f19b684e517dc8e6b5b375dda0",
    },
}
EXPECTED_CUDA_RUNTIME_LIBRARIES = (
    "libcublas.so.11",
    "libcublasLt.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.11",
    "libnvToolsExt.so.1",
)
EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.21.1",
    "pandas": "1.2.5",
    "scipy": "1.7.1",
    "scikit-learn": "0.24.2",
    "networkx": "2.6.2",
    "matplotlib": "3.4.3",
    "tqdm": "4.62.1",
    "laspy": "2.0.2",
    "PyYAML": "6.0.1",
    "torch": "1.9.0+cu111",
    "torch-geometric": "1.7.2",
    "torch-cluster": "1.5.9",
    "torch-scatter": "2.0.8",
    "torch-sparse": "0.6.11",
    "torch-spline-conv": "1.2.1",
}
REQUIRED_IMPORTS = (
    "laspy",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "torch_cluster",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_spline_conv",
    "tqdm",
    "yaml",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in EXPECTED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def version_contract_errors(
    installed: Mapping[str, Optional[str]],
    python_version: Sequence[int],
) -> list[str]:
    errors: list[str] = []
    if tuple(python_version[:2]) != EXPECTED_PYTHON:
        errors.append(
            "Python version mismatch: expected "
            f"{EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, found "
            f"{python_version[0]}.{python_version[1]}"
        )
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        actual = installed.get(name)
        if actual != expected:
            errors.append(
                f"Distribution mismatch for {name}: expected {expected}, found {actual}"
            )
    return errors


def conda_package_records(prefix: Path) -> dict[str, dict[str, Any]]:
    conda_meta = prefix.expanduser().resolve() / "conda-meta"
    if not conda_meta.is_dir():
        raise FileNotFoundError(f"Conda metadata directory does not exist: {conda_meta}")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(conda_meta.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = str(record.get("name", ""))
        if name in EXPECTED_CONDA_PACKAGES:
            records[name] = {
                "version": str(record.get("version", "")),
                "build": str(record.get("build", "")),
                "build_number": record.get("build_number"),
                "subdir": str(record.get("subdir", "")),
                "md5": str(record.get("md5", "")),
                "url": str(record.get("url", "")),
            }
    return records


def conda_package_contract_errors(
    installed: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for name, expected in EXPECTED_CONDA_PACKAGES.items():
        actual = installed.get(name)
        if not isinstance(actual, Mapping):
            errors.append(f"Conda package is missing: {name}")
            continue
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                errors.append(
                    f"Conda package mismatch for {name} {field}: "
                    f"expected {expected_value}, found {actual.get(field)}"
                )
    return errors


def cuda_runtime_library_evidence(prefix: Path) -> dict[str, Any]:
    library_dir = (prefix.expanduser().resolve() / "lib").resolve()
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    search_paths = {
        Path(entry).expanduser().resolve()
        for entry in ld_library_path.split(os.pathsep)
        if entry
    }
    if library_dir not in search_paths:
        raise RuntimeError(
            f"TLS2trees Conda library directory is missing from LD_LIBRARY_PATH: "
            f"{library_dir}"
        )
    libraries: dict[str, str] = {}
    for name in EXPECTED_CUDA_RUNTIME_LIBRARIES:
        candidate = library_dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Pinned CUDA runtime library is missing: {candidate}")
        libraries[name] = str(candidate.resolve())
    return {
        "library_dir": str(library_dir),
        "ld_library_path_contains_prefix_lib": True,
        "libraries": libraries,
    }


def setup_marker_errors(
    marker: Mapping[str, Any],
    expected_python_executable: Path,
) -> list[str]:
    errors: list[str] = []
    expected_fields = {
        "schema_version": EXPECTED_MARKER_SCHEMA,
        "status": "passed",
        "python_user_site_disabled": True,
        "tls2trees_commit": EXPECTED_UPSTREAM_COMMIT,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "cuda_required": True,
        "cpu_model_load": "passed",
        "compatibility_classification": (
            "historical_runtime_reproduction_not_for_instance_parameter_tuning"
        ),
    }
    for field, expected in expected_fields.items():
        if marker.get(field) != expected:
            errors.append(
                f"Setup marker has unexpected {field}: {marker.get(field)!r}"
            )

    recorded_python_version = str(marker.get("python_version", ""))
    try:
        recorded_major_minor = tuple(
            int(value) for value in recorded_python_version.split(".")[:2]
        )
    except ValueError:
        recorded_major_minor = ()
    if recorded_major_minor != EXPECTED_PYTHON:
        errors.append(
            "Setup marker has unexpected Python version: "
            f"{recorded_python_version!r}"
        )
    if not str(marker.get("validated_at_utc", "")).strip():
        errors.append("Setup marker validation timestamp is missing")

    recorded_python = Path(str(marker.get("python_executable", ""))).expanduser()
    if not recorded_python.is_absolute() or (
        recorded_python.resolve() != expected_python_executable.resolve()
    ):
        errors.append(
            "Setup marker Python does not match this environment: "
            f"recorded={recorded_python}, current={expected_python_executable}"
        )

    distributions = marker.get("distributions")
    if not isinstance(distributions, Mapping):
        errors.append("Setup marker distributions are missing")
    else:
        errors.extend(version_contract_errors(distributions, EXPECTED_PYTHON))

    conda_packages = marker.get("conda_packages")
    if not isinstance(conda_packages, Mapping):
        errors.append("Setup marker Conda package evidence is missing")
    else:
        errors.extend(conda_package_contract_errors(conda_packages))

    cuda_runtime = marker.get("cuda_runtime")
    if not isinstance(cuda_runtime, Mapping):
        errors.append("Setup marker CUDA runtime evidence is missing")
    else:
        expected_library_dir = str(
            (expected_python_executable.resolve().parent.parent / "lib").resolve()
        )
        if cuda_runtime.get("library_dir") != expected_library_dir:
            errors.append(
                "Setup marker has unexpected CUDA runtime library directory: "
                f"{cuda_runtime.get('library_dir')!r}"
            )
        if cuda_runtime.get("ld_library_path_contains_prefix_lib") is not True:
            errors.append("Setup marker did not validate the CUDA runtime search path")
        libraries = cuda_runtime.get("libraries")
        if not isinstance(libraries, Mapping):
            errors.append("Setup marker CUDA runtime libraries are missing")
        else:
            for name in EXPECTED_CUDA_RUNTIME_LIBRARIES:
                if not str(libraries.get(name, "")).strip():
                    errors.append(f"Setup marker CUDA runtime library is missing: {name}")

    cuda = marker.get("cuda")
    if not isinstance(cuda, Mapping):
        errors.append("Setup marker CUDA evidence is missing")
    else:
        expected_cuda = {
            "available": True,
            "compiled_pyg_operations": "passed",
            "bundled_model_forward": "passed",
            "deterministic_algorithms_enabled": False,
            "determinism_policy": EXPECTED_DETERMINISM_POLICY,
            "repeat_forward_all_finite": True,
            "seed": 42,
            "torch_cuda_build": "11.1",
            "model_missing_keys": [],
            "model_unexpected_keys": [],
        }
        for field, expected in expected_cuda.items():
            if cuda.get(field) != expected:
                errors.append(
                    f"Setup marker CUDA evidence has unexpected {field}: "
                    f"{cuda.get(field)!r}"
                )
        if not isinstance(cuda.get("repeat_forward_exact_equal"), bool):
            errors.append(
                "Setup marker CUDA evidence has invalid repeat_forward_exact_equal"
            )
        repeat_delta = cuda.get("repeat_forward_max_abs_delta")
        if (
            isinstance(repeat_delta, bool)
            or not isinstance(repeat_delta, (int, float))
            or not math.isfinite(repeat_delta)
            or repeat_delta < 0
        ):
            errors.append(
                "Setup marker CUDA evidence has invalid repeat_forward_max_abs_delta"
            )
        if not str(cuda.get("device_name", "")).strip():
            errors.append("Setup marker does not identify the validated GPU")
    return errors


def load_and_validate_setup_marker(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"TLS2trees setup marker does not exist: {resolved}")
    marker = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("TLS2trees setup marker must contain a JSON object")
    errors = setup_marker_errors(marker, Path(sys.executable))
    if errors:
        raise RuntimeError("; ".join(errors))
    return marker


def git_value(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Git command failed in {repo}: {' '.join(arguments)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_upstream(repo: Path) -> tuple[Path, str]:
    resolved = repo.expanduser().resolve()
    if not (resolved / ".git").is_dir():
        raise FileNotFoundError(f"TLS2trees Git checkout does not exist: {resolved}")
    commit = git_value(resolved, "rev-parse", "HEAD")
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"TLS2trees commit mismatch: expected {EXPECTED_UPSTREAM_COMMIT}, "
            f"found {commit}"
        )
    if git_value(resolved, "status", "--porcelain"):
        raise RuntimeError(f"TLS2trees checkout is dirty: {resolved}")
    model = resolved / "tls2trees" / "fsct" / "model" / "model.pth"
    if not model.is_file():
        raise FileNotFoundError(f"Bundled FSCT model does not exist: {model}")
    model_sha256 = sha256_file(model)
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"Bundled FSCT model mismatch: expected {EXPECTED_MODEL_SHA256}, "
            f"found {model_sha256}"
        )
    return model, model_sha256


def import_required_modules() -> dict[str, str]:
    imported: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        module = importlib.import_module(name)
        imported[name] = str(getattr(module, "__version__", "installed"))
    from torch_geometric.nn import (  # noqa: F401
        PointConv,
        fps,
        global_max_pool,
        knn_interpolate,
        radius,
    )

    del PointConv, fps, global_max_pool, knn_interpolate, radius
    return imported


def load_bundled_model(repo: Path, model_path: Path, device: Any) -> tuple[Any, list[str], list[str]]:
    for entry in (repo, repo / "tls2trees"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import torch
    from fsct.model import Net

    model = Net(num_classes=4).to(device)
    state = torch.load(str(model_path), map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Bundled FSCT state dict does not exactly match the pinned architecture: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model, list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def validate_cuda_stack(repo: Path, model_path: Path) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import fps, global_max_pool, knn_interpolate, radius

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable to PyTorch")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # TLS2trees/PyG 1.7.2 uses CUDA scatter_add, for which this historical
    # PyTorch stack has no deterministic implementation. Enforcing global
    # deterministic algorithms would reject the published method itself.
    torch.use_deterministic_algorithms(False)

    point_count = 4096
    positions = torch.rand((point_count, 3), dtype=torch.float32, device=device)
    features = torch.rand((point_count, 8), dtype=torch.float32, device=device)
    batch = torch.zeros(point_count, dtype=torch.long, device=device)
    sampled = fps(positions, batch, ratio=0.25, random_start=False)
    row, column = radius(
        positions,
        positions[sampled],
        0.25,
        batch,
        batch[sampled],
        max_num_neighbors=64,
    )
    interpolated = knn_interpolate(
        features,
        positions,
        positions[sampled[:128]],
        batch,
        batch[sampled[:128]],
        k=3,
    )
    pooled = global_max_pool(features, batch)
    if not len(sampled) or not len(row) or not len(column):
        raise RuntimeError("CUDA PyG sampling or radius operations returned no evidence")
    if interpolated.shape != (128, 8) or pooled.shape != (1, 8):
        raise RuntimeError("CUDA PyG interpolation or pooling returned an invalid shape")

    model, missing_keys, unexpected_keys = load_bundled_model(
        repo, model_path, device
    )
    data = Data(
        pos=torch.rand((point_count, 3), dtype=torch.float32, device=device),
        x=None,
        batch=batch,
    )
    with torch.no_grad():
        output = model(data)
        repeated_output = model(data)
    torch.cuda.synchronize()
    expected_output_shape = (1, 4, point_count)
    if (
        tuple(output.shape) != expected_output_shape
        or tuple(repeated_output.shape) != expected_output_shape
    ):
        raise RuntimeError(
            "Bundled FSCT repeat forward pass returned unexpected shapes: "
            f"first={tuple(output.shape)}, repeated={tuple(repeated_output.shape)}"
        )
    repeat_forward_all_finite = bool(
        torch.isfinite(output).all().item()
        and torch.isfinite(repeated_output).all().item()
    )
    if not repeat_forward_all_finite:
        raise RuntimeError("Bundled FSCT repeat forward pass returned non-finite values")
    repeat_forward_max_abs_delta = float(
        (output - repeated_output).abs().max().item()
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_capability": [properties.major, properties.minor],
        "torch_cuda_build": torch.version.cuda,
        "torch_arch_list": list(torch.cuda.get_arch_list()),
        "sampled_point_count": int(len(sampled)),
        "radius_edge_count": int(len(row)),
        "model_output_shape": list(output.shape),
        "model_missing_keys": missing_keys,
        "model_unexpected_keys": unexpected_keys,
        "compiled_pyg_operations": "passed",
        "bundled_model_forward": "passed",
        "deterministic_algorithms_enabled": False,
        "determinism_policy": EXPECTED_DETERMINISM_POLICY,
        "seed": 42,
        "repeat_forward_all_finite": repeat_forward_all_finite,
        "repeat_forward_exact_equal": bool(torch.equal(output, repeated_output)),
        "repeat_forward_max_abs_delta": repeat_forward_max_abs_delta,
    }


def validate_environment(
    repo: Path,
    require_cuda: bool,
    skip_model_load: bool = False,
) -> dict[str, Any]:
    if require_cuda and skip_model_load:
        raise ValueError("--skip-model-load cannot be combined with --require-cuda")
    os.environ["PYTHONNOUSERSITE"] = "1"
    model, model_sha256 = verify_upstream(repo)
    versions = distribution_versions()
    errors = version_contract_errors(versions, sys.version_info)
    conda_packages = conda_package_records(Path(sys.prefix))
    errors.extend(conda_package_contract_errors(conda_packages))
    if errors:
        raise RuntimeError("; ".join(errors))
    cuda_runtime = cuda_runtime_library_evidence(Path(sys.prefix))
    imports = import_required_modules()

    resolved_repo = repo.expanduser().resolve()
    import torch

    if skip_model_load:
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        model_load_status = "skipped_for_login_node_static_preflight"
    else:
        cpu_model, missing_keys, unexpected_keys = load_bundled_model(
            resolved_repo, model, torch.device("cpu")
        )
        del cpu_model
        model_load_status = "passed"
    cuda = (
        validate_cuda_stack(resolved_repo, model)
        if require_cuda
        else {
            "available": bool(torch.cuda.is_available()),
            "required": False,
            "torch_cuda_build": torch.version.cuda,
            "compiled_pyg_operations": "not_run_without_allocated_gpu",
            "bundled_model_forward": "not_run_without_allocated_gpu",
            "model_missing_keys": missing_keys,
            "model_unexpected_keys": unexpected_keys,
        }
    )
    return {
        "schema_version": EXPECTED_MARKER_SCHEMA,
        "status": "passed",
        "validated_at_utc": utc_now(),
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_user_site_disabled": os.environ.get("PYTHONNOUSERSITE") == "1",
        "expected_python": f"{EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}",
        "distributions": versions,
        "conda_packages": conda_packages,
        "cuda_runtime": cuda_runtime,
        "imports": imports,
        "tls2trees_repo": str(resolved_repo),
        "tls2trees_commit": EXPECTED_UPSTREAM_COMMIT,
        "model_path": str(model),
        "model_sha256": model_sha256,
        "cuda_required": require_cuda,
        "cuda": cuda,
        "cpu_model_load": model_load_status,
        "compatibility_classification": (
            "historical_runtime_reproduction_not_for_instance_parameter_tuning"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the isolated pinned TLS2trees runtime."
    )
    parser.add_argument("--tls2trees-repo", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--setup-marker-json")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = validate_environment(
            Path(args.tls2trees_repo),
            require_cuda=args.require_cuda,
            skip_model_load=args.skip_model_load,
        )
        if args.setup_marker_json:
            marker_path = Path(args.setup_marker_json)
            marker = load_and_validate_setup_marker(marker_path)
            payload["setup_marker_validation"] = {
                "status": "passed",
                "path": str(marker_path.expanduser().resolve()),
                "sha256": sha256_file(marker_path.expanduser().resolve()),
                "validated_at_utc": marker["validated_at_utc"],
                "gpu_name": marker["cuda"]["device_name"],
            }
        if args.output_json:
            output = Path(args.output_json).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"validation_json={output}")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("status=TLS2TREES_ENVIRONMENT_VALIDATED")
    print(f"python={payload['python_version']}")
    print(f"torch={payload['distributions']['torch']}")
    print(f"torch_geometric={payload['distributions']['torch-geometric']}")
    print(f"cuda_required={str(payload['cuda_required']).lower()}")
    print(f"cuda_available={str(payload['cuda']['available']).lower()}")
    if payload["cuda_required"]:
        print(f"gpu_name={payload['cuda']['device_name']}")
        print("compiled_pyg_operations=passed")
        print("bundled_model_forward=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
