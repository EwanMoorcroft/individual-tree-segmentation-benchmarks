"""Prepare a pinned SegmentAnyTree tracker that retains aligned instance labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHOR = '                    print("writing evaluation txt")'
EMPTY_PREDICTION_ANCHOR = "                    has_prediction = test_area_i.ins_pre != -1"
EMPTY_PREDICTION_GUARD = """                    has_prediction = test_area_i.ins_pre != -1
                    if not bool(has_prediction.any()):
                        _sat_os = __import__("os")
                        _sat_np = __import__("numpy")
                        _sat_output_dir = _sat_os.environ.get("SEGMENTANYTREE_ALIGNED_OUTPUT_DIR", ".")
                        self._dataset.to_eval_ply(
                            test_area_i.pos,
                            _sat_np.full(test_area_i.pos.shape[0], -1, dtype=_sat_np.int64),
                            test_area_i.instance_labels,
                            _sat_os.path.join(
                                _sat_output_dir,
                                "Instance_results_forEval_{}.ply".format(i),
                            ),
                        )
                        continue"""
INSERTION = """                    _sat_os = __import__("os")
                    _sat_output_dir = _sat_os.environ.get("SEGMENTANYTREE_ALIGNED_OUTPUT_DIR", ".")
                    _sat_full_ins_pred = (
                        full_ins_pred.detach().cpu().numpy()
                        if hasattr(full_ins_pred, "detach")
                        else full_ins_pred
                    )
                    self._dataset.to_eval_ply(
                        test_area_i.pos,
                        _sat_full_ins_pred,
                        test_area_i.instance_labels,
                        _sat_os.path.join(
                            _sat_output_dir,
                            "Instance_results_forEval_{}.ply".format(i),
                        ),
                    )

"""


def patch_source(source: str) -> str:
    if source.count(ANCHOR) != 1:
        raise ValueError("Expected one final evaluation output anchor")
    if source.count(EMPTY_PREDICTION_ANCHOR) != 1:
        raise ValueError("Expected one empty-prediction guard anchor")
    if "Instance_results_forEval_{}.ply" in source:
        raise ValueError("Aligned instance evaluation output is already enabled")
    patched = source.replace(EMPTY_PREDICTION_ANCHOR, EMPTY_PREDICTION_GUARD)
    patched = patched.replace(ANCHOR, INSERTION + ANCHOR)
    compile(patched, "panoptic_tracker_pointgroup_treeins.py", "exec")
    return patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the pinned SegmentAnyTree PointGroup tracker and enable its "
            "row-aligned instance prediction/ground-truth PLY output."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Upstream tracker does not exist: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    patched = patch_source(source_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")

    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "output": str(output_path),
                    "changes": [
                        (
                            "write full-resolution aligned instance predictions "
                            "and ground truth before export merging"
                        )
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
