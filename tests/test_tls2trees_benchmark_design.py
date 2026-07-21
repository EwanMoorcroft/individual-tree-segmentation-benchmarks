from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "methods/tls2trees/configs"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_tls2trees_split_contract_is_exact_and_isolated() -> None:
    config = load_yaml("for_instance_benchmark.yml")
    contract = config["dataset"]["split_contract"]
    development = contract["development"]
    test = contract["test"]

    assert config["project"]["status"] == (
        "development_tuned_completed_published_default_test_ready"
    )
    assert config["dataset"]["split_metadata"] == "data_split_metadata.csv"
    assert config["dataset"]["split_metadata_sha256"] == (
        "dd64aa338681f8f4166f8d175879a2b0b0158ecf222497ec6f7f0b23bc4fce94"
    )
    assert development["plot_count"] == 21
    assert development["point_count"] == 101_769_037
    assert development["reference_tree_count"] == 807
    assert test["plot_count"] == 11
    assert test["point_count"] == 49_709_922
    assert test["reference_tree_count"] == 323
    assert len(development["relative_paths"]) == development["plot_count"]
    assert len(test["relative_paths"]) == test["plot_count"]
    assert not set(development["relative_paths"]) & set(test["relative_paths"])
    assert config["split_control"]["held_out_test_accessed_during_tuning"] is False
    assert config["split_control"]["repeat_test_for_setting_selection_permitted"] is False


def test_tls2trees_leaf_targets_are_explicit_and_separate() -> None:
    config = load_yaml("for_instance_benchmark.yml")
    leaf_off = config["targets"]["leaf_off"]
    leaf_on = config["targets"]["leaf_on"]

    assert leaf_off["included_semantic_classes"] == [4, 6]
    assert leaf_off["excluded_semantic_classes"] == [0, 1, 2, 5]
    assert leaf_off["ignored_evaluation_semantic_classes"] == [3]
    assert leaf_off["prediction_pattern"] == "*.leafoff.ply"
    assert leaf_on["included_semantic_classes"] == [4, 5, 6]
    assert leaf_on["excluded_semantic_classes"] == [0, 1, 2]
    assert leaf_on["ignored_evaluation_semantic_classes"] == [3]
    assert leaf_on["prediction_pattern"] == "*.leafon.ply"
    assert leaf_off["ignored_instance_labels"] == [0, -1]
    assert leaf_on["ignored_instance_labels"] == [0, -1]
    assert leaf_off["prediction_pattern"] != leaf_on["prediction_pattern"]
    assert config["evaluation"]["protocol"] == (
        "for_instance_tls2trees_source_row_class3_ignore"
    )
    assert config["evaluation"]["inference_only_semantic_classes"] == [3]


def test_published_default_matches_paper_values_and_records_ambiguity() -> None:
    config = load_yaml("for_instance_published_default.yml")
    method = config["method"]
    preprocessing = config["published_preprocessing"]
    hidden_semantic = config["semantic_parameters"]["hidden_fixed_values"]
    instance = config["instance_parameters"]

    assert config["project"]["status"] == (
        "development_smoke_implemented_execution_pending"
    )
    assert method["variant"] == "published_default"
    assert method["training_mode"] == "external_training_only"
    assert method["publication_release"]["commit"] == (
        "216100ed2dade15d1bd6f09c287787e55085102a"
    )
    assert method["executable_pin"]["commit"] == (
        "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
    )
    environment = config["runtime_environment"]
    assert environment["classification"] == (
        "historical_runtime_compatibility_not_parameter_tuning"
    )
    assert environment["default_prefix"] == "~/fastscratch/venvs/tls2trees"
    assert environment["python"] == "3.9"
    assert environment["user_site_packages_enabled"] is False
    assert environment["primary_packages"]["torch"] == "1.9.0+cu111"
    assert environment["primary_packages"]["torch_geometric"] == "1.7.2"
    assert environment["barkla_validation_status"] == (
        "execution_pending_l40s_compatibility_unproven"
    )
    assert preprocessing["tile_edge_length_m"] == 10.0
    assert preprocessing["downsample_voxel_length_m"] == 0.02
    assert config["semantic_parameters"]["use_reference_semantic_labels_as_method_input"] is False
    assert config["semantic_parameters"]["buffer_m"] == 5.0
    assert hidden_semantic["max_distance_between_tiles_m"] == float("inf")
    assert hidden_semantic["inactive_in_current_preprocessing_and_inference_route"] == {
        "noise_class": 4,
        "slice_thickness_m": 0.2,
        "slice_increment_m": 0.05,
        "min_tree_cylinders": 10,
    }
    assert instance["n_tiles"] == 5
    assert "paper_permitting_3_5_or_7" in instance["n_tiles_provenance"]
    assert instance["slice_thickness_m"] == 0.5
    assert instance["find_stems_boundary_m"] == [2.0, 2.5]
    assert instance["find_stems_min_radius_m"] == 0.025
    assert instance["find_stems_min_points"] == 200
    assert instance["graph_edge_length_m"] == 2.0
    assert instance["graph_maximum_cumulative_gap_m"] == 3.0
    assert instance["min_points_per_tree"] == 200
    assert instance["add_leaves"] is True
    assert instance["add_leaves_voxel_length_m"] == 0.5
    assert instance["add_leaves_edge_length_m"] == 1.0
    assert config["reproducibility_controls"]["numpy_random_seed"] == 42
    compatibility_names = {
        item["name"] for item in config["compatibility_modifications"]
    }
    assert "cap_impossible_small_wood_graph_neighbour_count" in compatibility_names
    assert "reconcile_cross_tile_prediction_ownership" in compatibility_names
    assert config["reproducibility_controls"]["stochastic_realisation_note"]
    run_gate = config["run_gate"]
    assert run_gate["runnable"] is True
    assert run_gate["scope"] == "published_default_development_smoke_only"
    assert run_gate["full_development_runnable"] is False
    assert run_gate["development_search_runnable"] is False
    assert run_gate["held_out_test_runnable"] is False


def test_development_search_is_bounded_and_never_uses_test_metrics() -> None:
    config = load_yaml("for_instance_search_space.yml")
    design = config["design"]

    assert config["project"]["status"] == (
        "development_search_completed_configuration_frozen"
    )
    assert config["method"]["variant"] == "development_tuned"
    assert config["method"]["training_mode"] == "external_training_only"
    assert config["dataset"]["allowed_split"] == "development"
    assert config["dataset"]["held_out_test_accessed"] is False
    assert design["stage_0"]["selection_uses_accuracy_metrics"] is False
    assert design["stage_1"]["maximum_configurations_including_baseline"] == 12
    assert design["stage_2"]["maximum_candidates_per_target"] == 3
    assert design["stage_2"]["maximum_refinement_rounds"] == 1
    assert design["stage_3"]["user_review_required_before_tuned_test"] is True
    assert design["reproducibility_status"] == (
        "stage1_candidates_frozen_before_development_metrics"
    )
    probe = config["compatibility_probe"]
    assert probe["runnable"] is True
    assert probe["candidate_count"] == 6
    assert probe["selection_uses_accuracy_metrics"] is False
    assert probe["reference_labels_accessed"] is False
    assert probe["held_out_test_accessed"] is False
    assert probe["full_development_runnable"] is False
    stage1 = config["stage1_execution"]
    assert stage1["runnable"] is True
    assert stage1["promoted_candidate_count"] == 4
    assert stage1["stage0_plot_count"] == 5
    assert stage1["expected_metric_count"] == 40
    assert stage1["final_configuration_selected"] is False
    assert stage1["full_development_runnable"] is False
    assert stage1["held_out_test_runnable"] is False
    assert config["selection"]["test_metrics_permitted"] is False


def test_legacy_oracle_pilot_cannot_be_mistaken_for_target_row() -> None:
    legacy = load_yaml("for_instance_accuracy.yml")
    assert legacy["project"]["status"] == "legacy_diagnostic_not_target_row"
    assert legacy["project"]["mode"] == (
        "legacy_for_instance_tls2trees_leaf_off_oracle_semantic_pilot"
    )
    note = legacy["method"]["compatibility_note"]
    assert "oracle-semantic" in note
    assert "not a published_default or development_tuned benchmark row" in note


def test_tls2trees_variant_terminology_is_not_model_training_terminology() -> None:
    paths = [
        CONFIG_ROOT / "for_instance_benchmark.yml",
        CONFIG_ROOT / "for_instance_published_default.yml",
        CONFIG_ROOT / "for_instance_search_space.yml",
        ROOT / "methods/tls2trees/docs/for_instance_benchmark.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").casefold()
        assert "published_pretrained" not in text, path
        assert "fine_tuned" not in text, path
