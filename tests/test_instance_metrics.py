from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.instance_metrics import (
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)


def test_matching_maximizes_valid_pair_count_instead_of_greedy_iou() -> None:
    matrix = np.array([[0.90, 0.80], [0.85, 0.00]])

    matches = maximum_cardinality_threshold_matching(matrix, 0.5)

    assert matches == [(0, 1), (1, 0)]


def test_matching_handles_empty_matrices() -> None:
    assert maximum_cardinality_threshold_matching(np.zeros((3, 0)), 0.5) == []
    assert maximum_cardinality_threshold_matching(np.zeros((0, 3)), 0.5) == []


def test_matching_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        maximum_cardinality_threshold_matching(np.zeros(3), 0.5)
    with pytest.raises(ValueError, match="interval"):
        maximum_cardinality_threshold_matching(np.zeros((1, 1)), 0.0)
    with pytest.raises(ValueError, match="finite"):
        maximum_cardinality_threshold_matching(np.array([[np.nan]]), 0.5)


def test_precision_recall_f1_uses_count_identity() -> None:
    assert precision_recall_f1(2, 1, 2) == (2 / 3, 0.5, 4 / 7)
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        precision_recall_f1(1, -1, 0)


def test_package_star_import_exposes_only_existing_modules() -> None:
    namespace: dict[str, object] = {}
    exec("from benchmark import *", namespace)
    assert namespace["instance_metrics"].__name__ == "benchmark.instance_metrics"
    assert namespace["ply_io"].__name__ == "benchmark.ply_io"
