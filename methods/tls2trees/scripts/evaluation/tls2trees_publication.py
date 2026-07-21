"""Shared fail-closed primitives for TLS2trees public-result publication."""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Collection, Iterator, Mapping


PUBLICATION_LOCK_RELATIVE = Path(
    "results/metadata/tls2trees/for_instance/.public_result_publication.lock"
)


def lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without following its final symlink."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def publication_path(path: Path, project_root: Path) -> Path:
    """Validate a lexical publication path and every existing parent directory."""

    root = lexical_absolute(project_root)
    candidate = lexical_absolute(path)
    root_resolved = root.resolve(strict=True)
    # Resolve parent aliases for containment, but deliberately do not follow
    # the final component: require_regular_or_missing must still see and
    # reject a publication target that is itself a symlink.
    candidate_resolved = candidate.parent.resolve(strict=False) / candidate.name
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Publication target is outside the project root: {candidate}"
        ) from exc
    if candidate_resolved == root_resolved:
        raise ValueError("Publication target cannot be the project root")

    parent = candidate.parent
    while parent.resolve(strict=False) != root_resolved:
        if parent.is_symlink():
            raise ValueError(f"Publication parent is a symlink: {parent}")
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Publication parent is not a directory: {parent}")
        if parent == parent.parent:
            raise ValueError(
                f"Publication target is outside the project root: {candidate}"
            )
        parent = parent.parent
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Invalid publication project root: {root}")
    return candidate


def require_regular_or_missing(path: Path, *, label: str) -> None:
    """Reject symlinks and non-regular filesystem objects at an exact path."""

    if path.is_symlink():
        raise ValueError(f"{label} is a symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    if path.exists() and path.stat().st_nlink != 1:
        raise ValueError(f"{label} has multiple hard links: {path}")


def require_git_head(project_root: Path, expected_head: str) -> None:
    completed = subprocess.run(
        ["git", "-C", os.fspath(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != expected_head:
        raise RuntimeError(
            "TLS2trees publication HEAD changed after finalisation submission"
        )


def preflight_text_target(
    path: Path,
    *,
    project_root: Path,
    staging_suffix: str,
) -> tuple[Path, Path]:
    target = publication_path(path, project_root)
    temporary = target.with_name(f".{target.name}{staging_suffix}")
    require_regular_or_missing(target, label="Publication target")
    require_regular_or_missing(temporary, label="Publication staging file")
    return target, temporary


def validate_git_worktree(
    project_root: Path,
    *,
    recovery_confirmed: bool,
    recovery_paths: Collection[str],
    expected_head: str,
) -> None:
    """Recheck the publication worktree while the shared lock is held."""

    root = lexical_absolute(project_root)
    require_git_head(root, expected_head)
    completed = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        ],
        check=True,
        capture_output=True,
    )
    lock_entry = os.fsencode(PUBLICATION_LOCK_RELATIVE.as_posix())
    entries = [
        entry
        for entry in completed.stdout.split(b"\0")
        if entry and entry[3:] != lock_entry
    ]
    if not recovery_confirmed:
        if entries:
            raise RuntimeError(
                "Refusing TLS2trees public-result finalisation from a dirty worktree"
            )
        return

    allowed = set(recovery_paths)
    for entry in entries:
        if len(entry) < 4 or entry[2:3] != b" ":
            raise RuntimeError("Malformed Git worktree status during publication")
        status = os.fsdecode(entry[:2])
        changed_path = os.fsdecode(entry[3:])
        if status not in {" M", "??"}:
            raise RuntimeError(
                f"Publication recovery rejects Git status {status!r}: {changed_path}"
            )
        if changed_path not in allowed:
            raise RuntimeError(
                f"Publication recovery rejects unrelated path: {changed_path}"
            )
        candidate = publication_path(root / changed_path, root)
        if candidate.is_symlink():
            raise RuntimeError(
                f"Publication recovery rejects symbolic link: {changed_path}"
            )


@contextmanager
def publication_lock(project_root: Path) -> Iterator[Path]:
    """Hold the one cross-finalizer advisory lock for a complete publication.

    The lock is deliberately non-blocking. A concurrent publisher must stop and
    be retried rather than render registry updates from stale pre-lock content.
    """

    root = lexical_absolute(project_root)
    lock_path = publication_path(root / PUBLICATION_LOCK_RELATIVE, root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    require_regular_or_missing(lock_path, label="TLS2trees publication lock")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ValueError(
                "TLS2trees publication lock is not a unique regular file: "
                f"{lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another TLS2trees public-result finalizer holds the publication lock"
            ) from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _write_exclusive_regular(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _head_blob(project_root: Path, path: Path) -> bytes | None:
    relative = path.relative_to(project_root).as_posix()
    completed = subprocess.run(
        ["git", "-C", os.fspath(project_root), "show", f"HEAD:{relative}"],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return completed.stdout
    return None


def publish_text_bundle(
    writes: Mapping[Path, str],
    *,
    project_root: Path,
    staging_suffix: str,
    replace: Callable[[Path, Path], None] = os.replace,
    enforce_head_baseline: bool = False,
    expected_head: str | None = None,
) -> None:
    """Stage and recover a deterministic multi-file text publication safely.

    Callers must hold :func:`publication_lock` before they read mutable public
    registries and keep it held through this commit. Existing targets and
    staging files are accepted only when regular and byte-identical.
    """

    if not staging_suffix.startswith(".tls2trees-") or not staging_suffix.endswith(
        ".tmp"
    ):
        raise ValueError("Invalid TLS2trees publication staging suffix")

    root = lexical_absolute(project_root)
    normalized: dict[Path, str] = {}
    for raw_path, text in writes.items():
        path = publication_path(raw_path, root)
        if path in normalized:
            raise ValueError(f"Publication contains duplicate target path: {path}")
        if not isinstance(text, str):
            raise TypeError(f"Publication content is not text: {path}")
        normalized[path] = text

    # Validate every existing public and staging path before creating anything.
    paths: list[tuple[Path, Path, str, bytes | None]] = []
    for path in sorted(normalized, key=lambda item: item.as_posix()):
        path, temporary = preflight_text_target(
            path,
            project_root=root,
            staging_suffix=staging_suffix,
        )
        current = path.read_bytes() if path.exists() else None
        if enforce_head_baseline:
            rendered = normalized[path].encode("utf-8")
            if current != rendered:
                baseline = _head_blob(root, path)
                if current != baseline:
                    raise ValueError(
                        "Publication target differs from both the Git HEAD "
                        f"baseline and rendered recovery content: {path}"
                    )
        paths.append((path, temporary, normalized[path], current))

    staged: list[tuple[Path, Path, bytes | None]] = []
    created_staging: set[Path] = set()
    staging = True
    try:
        for path, temporary, text, original in paths:
            if path.exists() and path.read_text(encoding="utf-8") == text:
                if temporary.exists():
                    if temporary.read_text(encoding="utf-8") != text:
                        raise ValueError(
                            f"Publication staging file conflicts with target: {temporary}"
                        )
                    temporary.unlink()
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            staged.append((path, temporary, original))
            if temporary.exists():
                if temporary.read_text(encoding="utf-8") != text:
                    raise ValueError(
                        f"Publication staging content conflicts: {temporary}"
                    )
            else:
                _write_exclusive_regular(temporary, text)
                created_staging.add(temporary)
            require_regular_or_missing(
                temporary, label="Publication staging file"
            )
            if temporary.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"Staged publication content changed: {temporary}")

        if expected_head is not None:
            require_git_head(root, expected_head)
        for path, _, _, original in paths:
            current = path.read_bytes() if path.exists() else None
            if current != original:
                raise RuntimeError(
                    f"Publication target changed during staging: {path}"
                )
        staging = False
        for path, temporary, original in staged:
            # os.replace replaces an exact symlink pathname rather than following
            # it, but re-check to fail closed on unexpected concurrent changes.
            if expected_head is not None:
                require_git_head(root, expected_head)
            require_regular_or_missing(path, label="Publication target")
            require_regular_or_missing(temporary, label="Publication staging file")
            current = path.read_bytes() if path.exists() else None
            if current != original:
                raise RuntimeError(
                    f"Publication target changed before replacement: {path}"
                )
            replace(temporary, path)
            require_regular_or_missing(path, label="Published target")
            if path.read_text(encoding="utf-8") != normalized[path]:
                raise RuntimeError(f"Published target differs after replacement: {path}")
            if expected_head is not None:
                require_git_head(root, expected_head)
        for path, _, text, _ in paths:
            require_regular_or_missing(path, label="Published target")
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"Published bundle verification failed: {path}")
        if expected_head is not None:
            require_git_head(root, expected_head)
    except BaseException:
        if staging:
            for temporary in created_staging:
                if temporary.is_symlink():
                    continue
                if temporary.is_file():
                    temporary.unlink()
        raise
