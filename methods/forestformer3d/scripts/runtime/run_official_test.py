"""Run pinned upstream tools/test.py without modifying the upstream checkout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SOURCE_COMMIT = "6a75c3735e4a4108d02ee944a8b93177f2360a4f"
TEST_SHA256 = "e05fd7a449d4fd4f2c1bb091e29d09fba21e2942b64790aed63b49f0bba51c96"
CONFIG_SHA256 = "cdf6bff5269dfbd73f4b4c7fe30deffdbfed7fd73668ca2f0bc5bb792f04ec1f"
CHECKPOINT_SHA256 = (
    "01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"
)
MODEL_SOURCE_SHA256 = (
    "aad9cb8d5b2cd2405fe3b4499d88bd8e02dd95241a93b8af2a10e59135160462"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_root: Path, config: Path) -> Path:
    source_root = source_root.resolve()
    test_py = source_root / "tools/test.py"
    if subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT:
        raise ValueError("ForestFormer3D source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("ForestFormer3D source checkout is dirty")
    if sha256_file(test_py) != TEST_SHA256:
        raise ValueError("Official tools/test.py SHA-256 mismatch")
    if sha256_file(config.resolve()) != CONFIG_SHA256:
        raise ValueError("Official config SHA-256 mismatch")
    return test_py


def audit_effective_predict(source_root: Path, output_dir: Path) -> dict[str, object]:
    """Fail closed on the manually audited label-use structure."""

    model_source = source_root / "oneformer3d/oneformer3d.py"
    observed_sha256 = sha256_file(model_source)
    if observed_sha256 != MODEL_SOURCE_SHA256:
        raise ValueError("Official model source SHA-256 mismatch")
    tree = ast.parse(model_source.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ForAINetV2OneFormer3D_XAwarequery"
    ]
    if len(classes) != 1:
        raise ValueError("Could not identify the audited ForestFormer3D class")
    predicts = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "predict"
    ]
    if len(predicts) != 1:
        raise ValueError("Could not identify the effective predict method")
    predict = predicts[0]

    load_lines: dict[str, list[int]] = {}
    for name in ("pts_semantic_gt", "pts_instance_gt", "all_instance_labels"):
        load_lines[name] = sorted(
            node.lineno
            for node in ast.walk(predict)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        )
    expected_load_lines = {
        "pts_semantic_gt": [2574],
        "pts_instance_gt": [2290, 2574],
        "all_instance_labels": [],
    }
    if load_lines != expected_load_lines:
        raise ValueError(
            "Effective predict label-use structure differs from the audited source"
        )

    evidence: dict[str, object] = {
        "schema": "forestformer3d_effective_predict_audit_v1",
        "status": "passed",
        "model_source": str(model_source),
        "model_source_sha256": observed_sha256,
        "class": classes[0].name,
        "predict_first_line": predict.lineno,
        "label_load_lines": load_lines,
        "finding": (
            "Ground-truth arrays are loaded for output serialization only; "
            "the instance-label unique set is assigned but never consumed."
        ),
        "prediction_uses_ground_truth": False,
    }
    (output_dir / "effective_predict_audit.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def install_model_input_fingerprint(output_dir: Path) -> None:
    """Hash the exact point tensor presented to the effective predict method."""

    import numpy as np
    from oneformer3d.oneformer3d import ForAINetV2OneFormer3D_XAwarequery

    original_predict = ForAINetV2OneFormer3D_XAwarequery.predict

    def audited_predict(
        self: object,
        batch_inputs_dict: dict[str, object],
        batch_data_samples: list[object],
        **kwargs: object,
    ) -> object:
        points = batch_inputs_dict.get("points")
        if not isinstance(points, list) or len(points) != 1:
            raise ValueError("Expected one model-facing point tensor")
        tensor = points[0]
        array = tensor.detach().cpu().contiguous().numpy()
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"Unexpected model-facing point shape: {array.shape}")
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
        evidence = {
            "schema": "forestformer3d_model_input_fingerprint_v1",
            "batch_input_keys": sorted(batch_inputs_dict),
            "point_count": int(array.shape[0]),
            "point_dimensions": int(array.shape[1]),
            "dtype": str(array.dtype),
            "point_tensor_sha256": digest.hexdigest(),
        }
        (output_dir / "model_input_fingerprint.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return original_predict(
            self, batch_inputs_dict, batch_data_samples, **kwargs
        )

    ForAINetV2OneFormer3D_XAwarequery.predict = audited_predict


def prepare_entrypoint_checkpoint(checkpoint_path: Path, output_dir: Path) -> Path:
    """Precondition the already-fixed archive for upstream's mandatory fix.

    The published archive contains RSKC sparse weights, which spconv 2.3.6
    accepts directly. Pinned tools/test.py unconditionally applies
    ``permute(1, 2, 3, 4, 0)`` as though it had received an unfixed checkpoint.
    We supply the inverse layout so that upstream's unchanged operation
    reconstructs every original tensor exactly before MMEngine loads it.
    """

    import torch

    checkpoint_path = checkpoint_path.resolve()
    observed_sha256 = sha256_file(checkpoint_path)
    if observed_sha256 != CHECKPOINT_SHA256:
        raise ValueError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, found {observed_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Official checkpoint does not contain a state_dict mapping")

    converted = 0
    for key in list(checkpoint["state_dict"]):
        tensor = checkpoint["state_dict"][key]
        if (
            (key.startswith("unet") or key.startswith("input_conv"))
            and key.endswith("weight")
            and getattr(tensor, "ndim", None) == 5
        ):
            preconditioned = tensor.permute(4, 0, 1, 2, 3).contiguous()
            restored = preconditioned.permute(1, 2, 3, 4, 0)
            if not torch.equal(restored, tensor):
                raise ValueError(f"Checkpoint precondition is not lossless: {key}")
            checkpoint["state_dict"][key] = preconditioned
            converted += 1
    if converted != 49:
        raise ValueError(f"Expected 49 sparse weights, found {converted}")

    handle = tempfile.NamedTemporaryFile(
        prefix="ff3d_entrypoint_preconditioned_",
        suffix=".pth",
        delete=False,
    )
    handle.close()
    temporary_path = Path(handle.name)
    torch.save(checkpoint, temporary_path)
    evidence = {
        "schema": "forestformer3d_checkpoint_entrypoint_adapter_v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": observed_sha256,
        "preconditioned_checkpoint_sha256": sha256_file(temporary_path),
        "sparse_tensor_count": converted,
        "precondition_operation": "permute(4,0,1,2,3)",
        "upstream_operation": "permute(1,2,3,4,0)",
        "round_trip_exact": True,
        "reason": "published archive is already fixed; pinned test.py is unconditional",
    }
    (output_dir / "checkpoint_entrypoint_adapter.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return temporary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_py = verify_source(args.source_root, args.config)
    for path in (args.checkpoint, args.data_root / args.ann_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.work_dir.mkdir(parents=True, exist_ok=False)

    sys.path.insert(0, str(args.source_root.resolve()))
    audit_effective_predict(args.source_root.resolve(), args.work_dir)
    install_model_input_fingerprint(args.work_dir)

    # The pinned official file uses torch.load/torch.save but omits import torch.
    # Supplying the missing global preserves the exact upstream file and model
    # path; no model logic or source file is patched.
    import torch
    from mmengine.evaluator import BaseMetric
    from mmdet3d.registry import METRICS

    @METRICS.register_module(force=True)
    class ForestFormer3DSmokeNoOpMetric(BaseMetric):
        """Leave raw whole-plot validation to the benchmark adapter."""

        def __init__(self, **_kwargs: object) -> None:
            super().__init__()

        def process(self, _data_batch: object, _data_samples: object) -> None:
            return None

        def compute_metrics(self, _results: list[object]) -> dict[str, float]:
            return {}

    entrypoint_checkpoint = prepare_entrypoint_checkpoint(
        args.checkpoint, args.work_dir
    )
    sys.argv = [
        str(test_py),
        str(args.config.resolve()),
        str(entrypoint_checkpoint),
        "--work-dir",
        str(args.work_dir.resolve()),
        "--cfg-options",
        f"test_dataloader.dataset.data_root={args.data_root.resolve()}",
        f"test_dataloader.dataset.ann_file={args.ann_file}",
        "test_dataloader.batch_size=1",
        "test_dataloader.num_workers=0",
        "test_dataloader.persistent_workers=False",
        "test_evaluator.type=ForestFormer3DSmokeNoOpMetric",
        "randomness.seed=3407",
        "randomness.deterministic=False",
    ]
    os.chdir(args.source_root.resolve())
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        runpy.run_path(
            str(test_py), run_name="__main__", init_globals={"torch": torch}
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        resource_evidence = {
            "schema": "forestformer3d_case_resource_usage_v1",
            "wall_seconds": time.monotonic() - started,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
        }
        (args.work_dir / "resource_usage.json").write_text(
            json.dumps(resource_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entrypoint_checkpoint.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
