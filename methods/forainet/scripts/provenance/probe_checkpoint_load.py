"""Instantiate the pinned ForAINet model and audit checkpoint tensor coverage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")

    pointcloud_root = args.upstream_root / "PointCloudSegmentation"
    if not pointcloud_root.is_dir():
        raise FileNotFoundError(pointcloud_root)
    os.chdir(pointcloud_root)
    sys.path.insert(0, str(pointcloud_root))

    import numpy
    import torch
    from omegaconf import OmegaConf
    from torch_points3d.datasets.base_dataset import BaseDataset
    from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

    archive = torch.load(args.checkpoint, map_location="cpu")
    expected = archive["models"]["latest"]
    run_config = OmegaConf.create(archive["run_config"])
    run_config.data.fold = []
    checkpoint = ModelCheckpoint(
        str(args.checkpoint.parent),
        args.checkpoint.stem,
        "latest",
        run_config=run_config,
        resume=False,
        strict=True,
    )
    dataset_properties = checkpoint.dataset_properties
    if not dataset_properties:
        raise RuntimeError("checkpoint does not retain dataset properties")
    dataset = OmegaConf.create(dataset_properties)
    model = checkpoint.create_model(dataset, weight_name="latest")
    BaseDataset.set_transform(model, checkpoint.data_config)
    actual = model.state_dict()

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = sorted(
        key
        for key in expected_keys & actual_keys
        if tuple(expected[key].shape) != tuple(actual[key].shape)
    )
    compatible = not missing and not unexpected and not shape_mismatches
    payload = {
        "schema": "forainet_checkpoint_load_probe_v1",
        "status": "verified" if compatible else "incompatible",
        "loader": "official_model_checkpoint_with_saved_dataset_properties",
        "data_fold_override": [],
        "weight_name": "latest",
        "dataset_properties": dataset_properties,
        "checkpoint_tensor_count": len(expected_keys),
        "model_tensor_count": len(actual_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "compatible_fraction": (
            (len(expected_keys) - len(missing) - len(shape_mismatches))
            / len(expected_keys)
            if expected_keys
            else 0.0
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "torch-geometric": package_version("torch-geometric"),
            "MinkowskiEngine": package_version("MinkowskiEngine"),
            "torchsparse": package_version("torchsparse"),
            "hdbscan": package_version("hdbscan"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not compatible:
        raise RuntimeError("checkpoint does not completely match the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
