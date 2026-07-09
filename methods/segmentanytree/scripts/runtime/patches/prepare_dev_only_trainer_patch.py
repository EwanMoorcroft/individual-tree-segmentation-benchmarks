"""Prepare a development-only SegmentAnyTree trainer with safe checkpoint resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEST_EVALUATION_BLOCK = """            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")
"""
DISABLED_TEST_EVALUATION_BLOCK = """            # Test evaluation is run separately after model selection.
            if False:
                self._test_epoch(epoch, "test")
"""
INCOMPLETE_RESUME_BLOCK = """            #train need
            #self._model.instantiate_optimizers(self._cfg, "cuda" in device)
"""
RESUME_SCALER_BLOCK = """            # The checkpoint restores optimizer state but not the AMP scaler.
            self._model._grad_scale = torch.cuda.amp.GradScaler(
                enabled=self._model.is_mixed_precision()
            )
"""
IMPORT_BLOCK = """import os
import copy
"""
DIAGNOSTIC_IMPORT_BLOCK = """import os
import copy
import faulthandler
import json

_diagnostic_stack_seconds = int(
    os.environ.get("SEGMENTANYTREE_DIAGNOSTIC_STACK_SECONDS", "0")
)
if _diagnostic_stack_seconds > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        _diagnostic_stack_seconds,
        repeat=False,
    )
"""
PRETRAINED_LOAD_ANCHOR = """            self._model.set_pretrained_weights()
"""
PRETRAINED_LOAD_VALIDATION = PRETRAINED_LOAD_ANCHOR + """            if os.environ.get("SEGMENTANYTREE_REQUIRE_PRETRAINED_LOAD") == "1":
                pretrained_path = os.environ["SEGMENTANYTREE_PRETRAINED_PATH"]
                weight_name = os.environ.get(
                    "SEGMENTANYTREE_PRETRAINED_WEIGHT_NAME",
                    "latest",
                )
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                models = checkpoint.get("models", {})
                if weight_name not in models:
                    raise ValueError(
                        "Pretrained checkpoint does not contain weight set {}. "
                        "Available: {}".format(weight_name, sorted(models))
                    )
                weights = models[weight_name]
                model_state = self._model.state_dict()
                compatible = {
                    key: value
                    for key, value in weights.items()
                    if key in model_state and value.size() == model_state[key].size()
                }
                total_numel = sum(value.numel() for value in model_state.values())
                compatible_numel = sum(value.numel() for value in compatible.values())
                compatible_fraction = (
                    compatible_numel / total_numel if total_numel else 0.0
                )
                minimum_fraction = float(
                    os.environ.get(
                        "SEGMENTANYTREE_PRETRAINED_MIN_COMPATIBLE_FRACTION",
                        "0.95",
                    )
                )
                if compatible_fraction < minimum_fraction:
                    raise ValueError(
                        "Pretrained checkpoint compatibility {:.6f} is below "
                        "required {:.6f}.".format(
                            compatible_fraction,
                            minimum_fraction,
                        )
                    )
                self._model.load_state_dict_with_same_shape(weights, strict=False)
                validation_path = os.environ.get(
                    "SEGMENTANYTREE_PRETRAINED_VALIDATION_OUTPUT"
                )
                if validation_path:
                    with open(validation_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "checkpoint": pretrained_path,
                                "weight_name": weight_name,
                                "checkpoint_state_keys": len(weights),
                                "compatible_state_keys": len(compatible),
                                "compatible_numel": compatible_numel,
                                "model_numel": total_numel,
                                "compatible_fraction": compatible_fraction,
                                "minimum_fraction": minimum_fraction,
                            },
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.write("\\n")
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def patch_source(source: str) -> str:
    count = source.count(TEST_EVALUATION_BLOCK)
    if count != 2:
        raise ValueError(
            "Expected two automatic test-evaluation blocks in the pinned "
            f"trainer, found {count}"
        )
    if source.count(INCOMPLETE_RESUME_BLOCK) != 1:
        raise ValueError("Expected the pinned trainer's incomplete resume block.")
    if source.count(IMPORT_BLOCK) != 1:
        raise ValueError("Expected the pinned trainer import block.")
    if source.count(PRETRAINED_LOAD_ANCHOR) != 1:
        raise ValueError("Expected the pinned trainer pretrained-load anchor.")
    patched = source.replace(
        TEST_EVALUATION_BLOCK,
        DISABLED_TEST_EVALUATION_BLOCK,
    )
    patched = patched.replace(INCOMPLETE_RESUME_BLOCK, RESUME_SCALER_BLOCK)
    patched = patched.replace(
        PRETRAINED_LOAD_ANCHOR,
        PRETRAINED_LOAD_VALIDATION,
    )
    return patched.replace(IMPORT_BLOCK, DIAGNOSTIC_IMPORT_BLOCK)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the pinned SegmentAnyTree trainer and disable automatic test "
            "evaluation during development-only training."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    source = source_path.read_text(encoding="utf-8")
    patched = patch_source(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")

    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "source_sha256": sha256_text(source),
                    "output": str(output_path),
                    "output_sha256": sha256_text(patched),
                    "automatic_test_evaluation_disabled": True,
                    "validation_evaluation_preserved": True,
                    "resume_gradient_scaler_initialized": True,
                    "pretrained_weight_compatibility_check_supported": True,
                    "diagnostic_stack_dump_supported": True,
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
