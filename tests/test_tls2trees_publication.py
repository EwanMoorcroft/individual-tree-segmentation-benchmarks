from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/tls2trees_publication.py"
)
SPEC = importlib.util.spec_from_file_location("tls2trees_publication_tested", MODULE_PATH)
assert SPEC and SPEC.loader
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def test_project_root_cannot_be_a_publication_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="cannot be the project root"):
        publication.publication_path(project, project)


def test_publication_accepts_equivalent_barkla_mount_alias(tmp_path: Path) -> None:
    physical_parent = tmp_path / "mnt/scratch/user"
    physical_project = physical_parent / "tree-seg-benchmark"
    physical_project.mkdir(parents=True)
    alias_parent = tmp_path / "users/user/scratch"
    alias_parent.parent.mkdir(parents=True)
    alias_parent.symlink_to(physical_parent, target_is_directory=True)
    aliased_target = (
        alias_parent
        / "tree-seg-benchmark/results/metadata/tls2trees/receipt.json"
    )

    assert publication.publication_path(
        aliased_target, physical_project
    ) == publication.lexical_absolute(aliased_target)


def test_text_publication_rejects_target_symlink_without_external_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    target = project / "methods/tls2trees/examples/result.csv"
    external = tmp_path / "external.csv"
    target.parent.mkdir(parents=True)
    external.write_text("external remains unchanged\n", encoding="utf-8")
    target.symlink_to(external)

    with pytest.raises(ValueError, match="Publication target is a symlink"):
        publication.publish_text_bundle(
            {target: "new public result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
        )

    assert target.is_symlink()
    assert external.read_text(encoding="utf-8") == "external remains unchanged\n"


def test_text_publication_rejects_staging_symlink_without_external_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    target = project / "methods/tls2trees/examples/result.csv"
    temporary = target.with_name(
        f".{target.name}.tls2trees-test-finalisation.tmp"
    )
    external = tmp_path / "external.csv"
    target.parent.mkdir(parents=True)
    external.write_text("external remains unchanged\n", encoding="utf-8")
    temporary.symlink_to(external)

    with pytest.raises(ValueError, match="Publication staging file is a symlink"):
        publication.publish_text_bundle(
            {target: "new public result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
        )

    assert not target.exists()
    assert temporary.is_symlink()
    assert external.read_text(encoding="utf-8") == "external remains unchanged\n"


def test_publication_lock_rejects_a_concurrent_publisher(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    with publication.publication_lock(project):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    f"sys.path.insert(0, {str(MODULE_PATH.parent)!r}); "
                    "from tls2trees_publication import publication_lock; "
                    f"\nwith publication_lock(Path({str(project)!r})):\n    pass"
                ),
            ],
            capture_output=True,
            text=True,
        )

    assert completed.returncode != 0
    assert "Another TLS2trees public-result finalizer" in completed.stderr


def test_publication_lock_rejects_a_hard_linked_lock_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lock_path = project / publication.PUBLICATION_LOCK_RELATIVE
    external = tmp_path / "external-lock"
    lock_path.parent.mkdir(parents=True)
    external.write_text("external lock\n", encoding="utf-8")
    os.link(external, lock_path)

    with pytest.raises(ValueError, match="multiple hard links"):
        with publication.publication_lock(project):
            pass

    assert external.read_text(encoding="utf-8") == "external lock\n"


def test_locked_git_gate_rechecks_clean_and_exact_recovery_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    allowed = project / "methods/tls2trees/examples/result.csv"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    allowed.write_text("recoverable\n", encoding="utf-8")
    with publication.publication_lock(project):
        with pytest.raises(RuntimeError, match="dirty worktree"):
            publication.validate_git_worktree(
                project,
                recovery_confirmed=False,
                recovery_paths=set(),
                expected_head=head,
            )
        publication.validate_git_worktree(
            project,
            recovery_confirmed=True,
            recovery_paths={"methods/tls2trees/examples/result.csv"},
            expected_head=head,
        )

    unrelated = project / "unrelated.txt"
    unrelated.write_text("not permitted\n", encoding="utf-8")
    with publication.publication_lock(project):
        with pytest.raises(RuntimeError, match="unrelated path"):
            publication.validate_git_worktree(
                project,
                recovery_confirmed=True,
                recovery_paths={"methods/tls2trees/examples/result.csv"},
                expected_head=head,
            )


def test_conflicting_preexisting_stage_is_preserved(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / "methods/tls2trees/examples/result.csv"
    temporary = target.with_name(
        f".{target.name}.tls2trees-test-finalisation.tmp"
    )
    temporary.parent.mkdir(parents=True)
    temporary.write_text("conflict evidence\n", encoding="utf-8")

    with pytest.raises(ValueError, match="staging content conflicts"):
        publication.publish_text_bundle(
            {target: "rendered result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
        )

    assert temporary.read_text(encoding="utf-8") == "conflict evidence\n"
    assert not target.exists()


def test_publication_does_not_overwrite_target_changed_between_replacements(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    first = project / "methods/tls2trees/examples/first.csv"
    second = project / "methods/tls2trees/examples/second.csv"
    first.parent.mkdir(parents=True)
    first.write_text("first baseline\n", encoding="utf-8")
    second.write_text("second baseline\n", encoding="utf-8")
    replacements = 0

    def mutate_second_after_first(source: Path, destination: Path) -> None:
        nonlocal replacements
        os.replace(source, destination)
        replacements += 1
        if replacements == 1:
            second.write_text("concurrent manual edit\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed before replacement"):
        publication.publish_text_bundle(
            {first: "first result\n", second: "second result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
            replace=mutate_second_after_first,
        )

    assert second.read_text(encoding="utf-8") == "concurrent manual edit\n"


def test_publication_verifies_the_complete_bundle_before_returning(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    first = project / "methods/tls2trees/examples/first.csv"
    second = project / "methods/tls2trees/examples/second.csv"
    first.parent.mkdir(parents=True)
    replacements = 0

    def mutate_first_after_last_replace(source: Path, destination: Path) -> None:
        nonlocal replacements
        os.replace(source, destination)
        replacements += 1
        if replacements == 2:
            first.write_text("changed after publication\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="bundle verification failed"):
        publication.publish_text_bundle(
            {first: "first result\n", second: "second result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
            replace=mutate_first_after_last_replace,
        )


def test_text_publication_rejects_hard_linked_target(tmp_path: Path) -> None:
    project = tmp_path / "project"
    target = project / "methods/tls2trees/examples/result.csv"
    external = tmp_path / "external.csv"
    target.parent.mkdir(parents=True)
    external.write_text("external remains unchanged\n", encoding="utf-8")
    os.link(external, target)

    with pytest.raises(ValueError, match="multiple hard links"):
        publication.publish_text_bundle(
            {target: "rendered result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
        )

    assert external.read_text(encoding="utf-8") == "external remains unchanged\n"


def test_recovery_refuses_content_that_is_neither_head_nor_rendered(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    target = project / "methods/tls2trees/examples/result.csv"
    target.parent.mkdir(parents=True)
    target.write_text("Git baseline\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    target.write_text("unrelated manual edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from both"):
        publication.publish_text_bundle(
            {target: "rendered result\n"},
            project_root=project,
            staging_suffix=".tls2trees-test-finalisation.tmp",
            enforce_head_baseline=True,
        )

    assert target.read_text(encoding="utf-8") == "unrelated manual edit\n"
