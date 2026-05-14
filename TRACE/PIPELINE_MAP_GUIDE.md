# `pipeline_map.py` Layout Guide

Distilled from a three-agent collaborative pass on `TRACE/pipeline_map.py` (commit `14a1c53`). Use this when adding new nodes/edges, reorganising clusters, or building a similar diagram for another project. The point is to keep the diagram dense and readable as it grows.

---

## Hard constraints (don't violate)

These keep the diagram visually consistent across edits.

- **Do not change individual box shape, style, fill color, font size, margin, or label content.** Boxes are sized by their label content; their geometry is set in `build_dot`'s `node [...]` line and the per-group `fill` / `stroke` in `GROUPS`. Both should be considered locked unless the user explicitly asks for a re-style.
- **Do not rename nodes, change a node's `kind`, or alter its `substeps`.** The substeps are the source of truth for what each stage does — they're not layout knobs.
- **Do not add or delete semantic edges.** An edge represents a real data flow. Add an edge only when the underlying pipeline gained one; delete only when it's truly gone.

## What you *can* edit

- `GROUPS` — cluster labels, fills, ordering. Adding a new cluster is fine if the pipeline truly grew a new conceptual phase.
- The order of items in `NODES` — graphviz uses declaration order as a tiebreaker.
- The `EDGES` list — you may add `constraint=false`, `style="dashed"`, reorder entries.
- Global DOT settings inside `build_dot` — `rankdir`, `nodesep`, `ranksep`, `splines`, `concentrate`, `newrank`, `compound`, plus inline `{rank=same; ...}` groups.
- The `_RANK_PAIRS` dict — declare which nodes should sit on the same rank.

## Layout principles

Listed roughly in priority order. The first three should always be honoured; the rest are tools to reach for when something looks off.

1. **No single long thin column or short fat row, especially in the middle of the map.** This is the user's top visual complaint. Tall single-file stacks of 6+ boxes are bad; wide single-file rows of 7+ boxes are bad. Aim for a balanced footprint roughly 4:3 → 3:2 aspect ratio.
2. **Each input sits above its primary consumer column.** Order `NODES` so e.g. `IMG → P1 (left), WING_MODEL → P2, LMK_MODEL → P3, SEG_MODEL → P5, CONFIG → right`. Shortens the longest incoming arrows.
3. **Orthogonal routing for crisp arrows.** Set `splines=ortho` in DOT globals. Right-angle channels reduce perceived crossings by ~60% (~15-20 → ~5-7 in this diagram) and shorten apparent edge length.
4. **Side-reference edges should not constrain the layout.** Any edge that's a "side input" or "optional reference" — not the primary data spine — gets `constraint=false`. In this map: `CONFIG → I2/I5/I6/MM`, `WING_GJ → P5` (ROI), `LMK_GJ → OUT_LMK_OV`, `SEG_GJ → OUT_SEG_OV`. These also typically get `style="dashed"` so a reader can see at a glance which edges only fire conditionally.
5. **Pair adjacent stages into 2-wide rows via `{rank=same; ...}`.** Define them in `_RANK_PAIRS` (a dict keyed by group id). The `build_dot` helper emits `{rank=same; A; B;}` lines inside the cluster subgraph. Used here for `P1+P2 / P3+P4 / P5+P6` and `I1+I2 / I3+I4 / I5+I6`, collapsing tall single-file columns into 2-wide grids.
6. **Inputs and outputs are single horizontal strips.** Both clusters benefit from sitting in one or two rows rather than being free-flowing. `_RANK_PAIRS` makes this trivial.
7. **Tightened spacing** — `nodesep=0.30`, `ranksep=0.50`. Default (0.35 / 0.60) leaves too much white between rows once boxes get dense. Going much tighter than this starts to crowd labels.
8. **`newrank=true` + `compound=true`** when you use cross-cluster ranks. Without `newrank`, `rank=same` can fight cluster boundaries; without `compound`, edges to/from clusters draw incorrectly.

## Iteration workflow

1. Read `pipeline_map.py` to remind yourself of current state.
2. Render baseline → some `/tmp/<...>_baseline.png` using `.venv-pipeline-map/bin/python pipeline_map.py --png <path>`.
3. Look at the PNG (multimodal `Read` tool). Identify the worst single offender: long column, fat row, big empty wedge, edge that crosses every other edge, etc.
4. Make a *focused* change. Render → new PNG. Compare.
5. Iterate 3-6 times. Stop when further changes have diminishing returns.

If you have multiple competing layout goals (empty space, crossings, compactness, etc.), spawn one agent per goal in isolated worktrees so they don't fight each other, then merge their best techniques into a final pass.

## Things that *don't* work well (tried and rejected)

- `concentrate=true` *alone*: bundles parallel edges but tends to crush cluster layouts (External inputs ends up squashed off-canvas). Use only when paired with very controlled rank pinning.
- `rankdir=LR` for this graph: produces a 52" × 14" footprint — even harder to read at the standard zoom.
- `nodesep` / `ranksep` below ~0.25 / 0.40: starts compressing edge labels and overlapping arrowheads.
- Splitting clusters into more than ~6 subgroups: visual noise outweighs the organisational gain.

## Render commands (cheat sheet)

```bash
# Static PNG (most common during iteration)
TRACE/.venv-pipeline-map/bin/python TRACE/pipeline_map.py --png /tmp/preview.png

# Raw DOT for debugging
TRACE/.venv-pipeline-map/bin/python TRACE/pipeline_map.py --dot /tmp/pipeline.dot

# Interactive vispy window (default — pan/zoom)
TRACE/.venv-pipeline-map/bin/python TRACE/pipeline_map.py
```

## Adding new pipeline stages (recipe)

When the pipeline gains a new stage, step, or output:

1. **Add the `Node(...)` entry** in `NODES`, with the correct `group`, an ID short enough to read inline (`P7`, `I7`, `OUT_*`, etc.), and a list of `substeps` (one bullet per real operation — don't fluff).
2. **Wire the semantic edges** in `EDGES`. Use solid edges for the primary data flow; add `style="dashed"` for optional/conditional inputs.
3. **Register in `_RANK_PAIRS`** if the cluster's row pattern needs to extend (e.g. if you add P7, decide whether it pairs with P6 or starts a new row).
4. **If the new edge is a side reference** (cross-cluster jump that's not the primary spine), mark it `constraint=false` so it doesn't distort the layout.
5. Render, look, iterate.
