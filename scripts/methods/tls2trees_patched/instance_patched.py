"""Run pinned TLS2trees instance.py with the pandas 3 groupby.apply fix.

The upstream repository remains unchanged. This wrapper loads its instance
script, inserts the missing grouping key as ``clstr`` when pandas omits grouping
columns from ``DataFrameGroupBy.apply``, then executes the patched source.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_SCRIPT = PROJECT_ROOT / "external" / "TLS2trees" / "tls2trees" / "instance.py"
PATCH_TARGET = "    chull = chull.reset_index(drop=True)"
PATCH_REPLACEMENT = """\
    # pandas 3 excludes grouping columns from DataFrameGroupBy.apply results.
    if 'clstr' not in chull.columns:
        if chull.index.nlevels < 2:
            raise RuntimeError('Cannot restore clstr: groupby.apply did not return a grouped index')
        chull = chull.copy()
        chull.insert(0, 'clstr', chull.index.get_level_values(0).to_numpy())
    chull = chull.reset_index(drop=True)"""


def patched_source(source: str) -> str:
    matches = source.count(PATCH_TARGET)
    if matches != 1:
        raise RuntimeError(
            f"Expected one TLS2trees patch target, found {matches}; "
            "verify the pinned upstream commit before running"
        )
    return source.replace(PATCH_TARGET, PATCH_REPLACEMENT, 1)


def main() -> None:
    if not UPSTREAM_SCRIPT.is_file():
        raise FileNotFoundError(f"Upstream TLS2trees instance script not found: {UPSTREAM_SCRIPT}")
    source = patched_source(UPSTREAM_SCRIPT.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(UPSTREAM_SCRIPT),
        "__package__": None,
    }
    exec(compile(source, str(UPSTREAM_SCRIPT), "exec"), namespace)


if __name__ == "__main__":
    main()
