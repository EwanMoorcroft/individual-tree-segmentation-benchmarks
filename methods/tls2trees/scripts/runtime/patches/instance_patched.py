"""Run pinned TLS2trees instance.py with documented compatibility fixes.

The upstream repository remains unchanged. This wrapper loads its instance
script, restores the grouping key omitted by newer pandas, honours the parsed
leaf graph edge distance, seeds stochastic operations, then executes it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PATCH_TARGET = "    chull = chull.reset_index(drop=True)"
PATCH_REPLACEMENT = """\
    # pandas 3 excludes grouping columns from DataFrameGroupBy.apply results.
    # An empty apply result is a valid no-predictions outcome, not a schema error.
    if chull.empty:
        os.makedirs(params.odir, exist_ok=True)
        reason_path = os.path.join(
            params.odir,
            f'.{params.n}.tls2trees_no_predictions.txt',
        )
        with open(reason_path, 'x', encoding='utf-8') as handle:
            handle.write('no_clustered_wood_convex_hulls\\n')
        if params.verbose:
            print('no clustered wood convex hulls; no instance predictions written')
        raise SystemExit(0)
    if 'clstr' not in chull.columns:
        if chull.index.nlevels < 2:
            raise RuntimeError('Cannot restore clstr: groupby.apply did not return a grouped index')
        chull = chull.copy()
        chull.insert(0, 'clstr', chull.index.get_level_values(0).to_numpy())
    chull = chull.reset_index(drop=True)"""
LEAF_EDGE_TARGET = "max_length=1, # i.e. any leaves which are separated by greater are ignored"
LEAF_EDGE_REPLACEMENT = "max_length=params.add_leaves_edge_length, # configured leaf graph edge length"
EMPTY_ORIGINS_TARGET = (
    "    origins = [s for s in origins if s in edges.source.values]" + " \n\n"
    "    # compute graph"
)
EMPTY_ORIGINS_REPLACEMENT = """\
    origins = [s for s in origins if s in edges.source.values]
    if not origins:
        return pd.DataFrame(
            columns=['clstr', 'distance', 't_clstr', 'is_tip']
        )

    # compute graph"""
EMPTY_WOOD_PATH_TARGET = (
    "    wood_paths = generate_path(chull," + " \n"
    "                               skeleton.loc[skeleton.dbh_node].clstr," + " \n"
    "                               n_neighbours=200," + " \n"
    "                               max_length=params.graph_edge_length)\n\n"
    "    # removes paths that are longer for same clstr"
)
EMPTY_WOOD_PATH_REPLACEMENT = """\
    wood_paths = generate_path(chull,
                               skeleton.loc[skeleton.dbh_node].clstr,
                               n_neighbours=200,
                               max_length=params.graph_edge_length)
    if wood_paths.empty:
        os.makedirs(params.odir, exist_ok=True)
        reason_path = os.path.join(
            params.odir,
            f'.{params.n}.tls2trees_no_predictions.txt',
        )
        with open(reason_path, 'x', encoding='utf-8') as handle:
            handle.write('no_graph_connected_stem_bases\\n')
        if params.verbose:
            print('no graph-connected stem bases; no instance predictions written')
        raise SystemExit(0)

    # removes paths that are longer for same clstr"""
SMALL_GRAPH_TARGET = """\
def generate_path(samples, origins, n_neighbours=200, max_length=0):

    # compute nearest neighbours for each vertex in cluster convex hull
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])"""
SMALL_GRAPH_REPLACEMENT = """\
def generate_path(samples, origins, n_neighbours=200, max_length=0):

    # sklearn self-query adds one neighbour before removing each sample itself.
    # Preserve the published value for large graphs and use every other sample
    # when the graph is smaller than that fixed neighbourhood.
    sample_count = len(samples)
    if sample_count < 2:
        return pd.DataFrame(
            columns=['clstr', 'distance', 't_clstr', 'is_tip']
        )
    n_neighbours = min(n_neighbours, sample_count - 1)

    # compute nearest neighbours for each vertex in cluster convex hull
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])"""
NO_STEMS_TARGET = "    if params.add_leaves:"
NO_STEMS_REPLACEMENT = """\
    # No stem file was emitted for the central tile. Adding leaves cannot create
    # a valid instance without a stem owner, so record an explicit empty result.
    if not params.base_I:
        os.makedirs(params.odir, exist_ok=True)
        reason_path = os.path.join(
            params.odir,
            f'.{params.n}.tls2trees_no_predictions.txt',
        )
        with open(reason_path, 'x', encoding='utf-8') as handle:
            handle.write('no_in_tile_stem_predictions\\n')
        if params.verbose:
            print('no in-tile stem predictions; no instance predictions written')
        raise SystemExit(0)

    if params.add_leaves:"""
EMPTY_LEAF_TIPS_TARGET = """\
        chull = chull.loc[(chull.is_tip) & (chull.n_z > params.find_stems_boundary[0])]
        chull.loc[:, 'xlabel'] = 2"""
EMPTY_LEAF_TIPS_REPLACEMENT = """\
        chull = chull.loc[(chull.is_tip) & (chull.n_z > params.find_stems_boundary[0])]
        if chull.empty:
            # Valid stem instances exist, but there are no eligible branch tips
            # from which to attach leaves. Preserve those instances for the
            # leaf-on target without inventing leaf ownership.
            for lv in in_tile_stem_nodes:
                I = params.base_I[lv]
                wood_fn = glob.glob(os.path.join(
                    params.odir,
                    '*' if params.save_diameter_class else '',
                    f'{params.n}_T{I}.leafoff.ply',
                ))[0]
                leafon_fn = wood_fn.replace('leafoff', 'leafon')
                with open(wood_fn, 'rb') as source_handle:
                    with open(leafon_fn, 'xb') as target_handle:
                        target_handle.write(source_handle.read())
            if params.verbose:
                print('no eligible leaf-attachment tips; stem-only leaf-on predictions written')
            raise SystemExit(0)
        chull.loc[:, 'xlabel'] = 2"""


def upstream_script() -> Path:
    value = os.environ.get("TLS2TREES_REPO")
    repo = (
        Path(value).expanduser().resolve()
        if value
        else PROJECT_ROOT / "external" / "TLS2trees"
    )
    return repo / "tls2trees" / "instance.py"


def patched_source(
    source: str,
    *,
    require_leaf_edge: bool = False,
    require_empty_graph_guard: bool = False,
    require_small_graph_guard: bool = False,
    require_no_stems_guard: bool = False,
    require_empty_leaf_tips_guard: bool = False,
) -> str:
    matches = source.count(PATCH_TARGET)
    if matches != 1:
        raise RuntimeError(
            f"Expected one TLS2trees patch target, found {matches}; "
            "verify the pinned upstream commit before running"
        )
    source = source.replace(PATCH_TARGET, PATCH_REPLACEMENT, 1)
    leaf_matches = source.count(LEAF_EDGE_TARGET)
    if require_leaf_edge and leaf_matches != 1:
        raise RuntimeError(
            f"Expected one TLS2trees leaf-edge patch target, found {leaf_matches}; "
            "verify the pinned upstream commit before running"
        )
    if leaf_matches == 1:
        source = source.replace(LEAF_EDGE_TARGET, LEAF_EDGE_REPLACEMENT, 1)

    empty_origins_matches = source.count(EMPTY_ORIGINS_TARGET)
    empty_wood_path_matches = source.count(EMPTY_WOOD_PATH_TARGET)
    if require_empty_graph_guard and (
        empty_origins_matches != 1 or empty_wood_path_matches != 1
    ):
        raise RuntimeError(
            "Expected one TLS2trees empty-graph patch target for origins and "
            "wood paths; verify the pinned upstream commit before running"
        )
    if empty_origins_matches == 1:
        source = source.replace(
            EMPTY_ORIGINS_TARGET, EMPTY_ORIGINS_REPLACEMENT, 1
        )
    if empty_wood_path_matches == 1:
        source = source.replace(
            EMPTY_WOOD_PATH_TARGET, EMPTY_WOOD_PATH_REPLACEMENT, 1
        )
    small_graph_matches = source.count(SMALL_GRAPH_TARGET)
    if require_small_graph_guard and small_graph_matches != 1:
        raise RuntimeError(
            "Expected one TLS2trees small-graph neighbour patch target; "
            "verify the pinned upstream commit before running"
        )
    if small_graph_matches == 1:
        source = source.replace(SMALL_GRAPH_TARGET, SMALL_GRAPH_REPLACEMENT, 1)
    no_stems_matches = source.count(NO_STEMS_TARGET)
    if require_no_stems_guard and no_stems_matches != 1:
        raise RuntimeError(
            "Expected one TLS2trees no-stems patch target; verify the pinned "
            "upstream commit before running"
        )
    if no_stems_matches == 1:
        source = source.replace(NO_STEMS_TARGET, NO_STEMS_REPLACEMENT, 1)
    empty_leaf_tips_matches = source.count(EMPTY_LEAF_TIPS_TARGET)
    if require_empty_leaf_tips_guard and empty_leaf_tips_matches != 1:
        raise RuntimeError(
            "Expected one TLS2trees empty-leaf-tips patch target; verify the "
            "pinned upstream commit before running"
        )
    if empty_leaf_tips_matches == 1:
        source = source.replace(
            EMPTY_LEAF_TIPS_TARGET,
            EMPTY_LEAF_TIPS_REPLACEMENT,
            1,
        )
    return source


def main() -> None:
    script = upstream_script()
    if not script.is_file():
        raise FileNotFoundError(f"Upstream TLS2trees instance script not found: {script}")
    seed = int(os.environ.get("TLS2TREES_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    source = patched_source(
        script.read_text(encoding="utf-8"),
        require_leaf_edge=True,
        require_empty_graph_guard=True,
        require_small_graph_guard=True,
        require_no_stems_guard=True,
        require_empty_leaf_tips_guard=True,
    )
    namespace = {
        "__name__": "__main__",
        "__file__": str(script),
        "__package__": None,
    }
    exec(compile(source, str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
