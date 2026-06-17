"""Interactive pipeline map for the TRACE + identifyFeatures workflow.

The layout (node positions, sizes, edge splines) is precomputed by
graphviz **once** and cached in ``pipeline_layout.plain``. At runtime the
script reads that file and renders with vispy — no `dot` invocation
needed. To regenerate after a NODES/EDGES change, run with
``--regenerate-layout`` (requires the graphviz binary).

Run with the venv at ``.venv-pipeline-map/bin/python pipeline_map.py``.

Flags:
    --png PATH               Render a static PNG to PATH and exit (no window).
    --dot PATH               Write the generated DOT source to PATH.
    --regenerate-layout      Recompute the cached layout via `dot -Tplain`.
    --dry-run                Print a layout summary and exit.

Before editing this file — especially before adding nodes, edges, or
adjusting the layout — read ``PIPELINE_MAP_GUIDE.md`` (same directory). It
records the layout constraints, techniques (rank pairs, ortho routing,
side-reference ``constraint=false``, …), and what's been tried and
rejected. The diagram looks balanced today only because of those
choices, and they're not obvious from the surrounding code alone.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pipeline graph definition
# ---------------------------------------------------------------------------

# Groups drive fill colors and cluster backgrounds.
GROUPS = {
    # User-provided input (the only thing the user must supply).
    "user_input": {"label": "User input", "fill": "#fde6c4", "stroke": "#c8862a", "cluster": True},
    # Inputs that ship with TRACE — DL models + pipeline configuration.
    "input": {"label": "Bundled with TRACE", "fill": "#e8eef7", "stroke": "#4a6fa5", "cluster": True},
    "preproc": {"label": "TRACE Stage 1 — preprocessing", "fill": "#e6f4ea", "stroke": "#3c8c4f", "cluster": True},
    "artifact": {"label": "Intermediate artifacts", "fill": "#fff7e0", "stroke": "#b38a1b", "cluster": True},
    "ifeat": {"label": "TRACE Stage 2 — identifyFeatures", "fill": "#f4e6f4", "stroke": "#8c3c8c", "cluster": True},
    "result": {"label": "WingResult", "fill": "#fde7e7", "stroke": "#b33c3c", "cluster": True},
    "output": {"label": "Outputs", "fill": "#e0f0f7", "stroke": "#3078a0", "cluster": True},
}


@dataclass
class Node:
    id: str
    title: str
    group: str
    substeps: list[str] = field(default_factory=list)
    kind: str = "process"  # "process" or "data"


NODES: list[Node] = [
    # --- User input (the only thing the user must provide) ---
    Node(
        "IMG",
        "Wing image",
        "user_input",
        [
            "Standard: tif / tiff / bmp / png / jpg / jpeg",
            "Adobe: psd / psb",
            "Modern: heic / heif / svg",
            "Camera RAW: dng / nef / cr2 / cr3 / arw / raf / orf / pef / rw2 / srw / raw",
            "Microscopy: czi / nd2 / lif / lsm (→ OME-TIFF)",
        ],
        "data",
    ),
    # --- Bundled with TRACE (ship with the project) ---
    # Ordered so each model sits above its primary consumer column:
    # WING_MODEL → P2, LMK_MODEL → P3, SEG_MODEL → P5, CONFIG → ifeat (right).
    Node("WING_MODEL", "Wing isolation model dir", "input", ["weights + metadata.json (optional)"], "data"),
    Node("LMK_MODEL", "Landmark model", "input", ["ResNet18 U-Net checkpoint (.pt)"], "data"),
    Node("SEG_MODEL", "Segmentation model dir", "input", ["weights + metadata.json"], "data"),
    Node("CONFIG", "PipelineConfig", "input", ["JSON / GUI"], "data"),
    # --- Preprocessing stages (renumbered to start at 1) ---
    Node(
        "P1",
        "Stage 1: Resolution adjust",
        "preproc",
        [
            "Compare input µm/px to model's training µm/px",
            "Skip if ratio is inside tolerance band",
            "Otherwise rescale image toward target µm/px",
            "Geometry is inverse-rescaled after Stage 2 (identifyFeatures)",
        ],
    ),
    Node(
        "P2",
        "Stage 2: Wing isolation (optional)",
        "preproc",
        [
            "Run wing-isolation model → wing polygon",
            "Buffer polygon by wing_expand_fraction",
            "Mask non-wing pixels to 0",
            "Write isolated image + wing.geojson",
        ],
    ),
    Node(
        "P3",
        "Stage 3: Landmark detection",
        "preproc",
        [
            "Load LandmarkPredictor (cached)",
            "Heatmap regression inference",
            "Extract peak coordinates",
            "Map names → GeoJSON schema",
            "Write landmarks.geojson",
        ],
    ),
    Node(
        "P4",
        "Stage 4: Hinge chop",
        "preproc",
        [
            "Load landmarks from Stage 3",
            "Build hinge polyline from distal-margin landmarks",
            "Build proximal mask",
            "Black out hinge pixels (in place, no translation)",
            "Write chopped image (temp)",
        ],
    ),
    Node(
        "P5",
        "Stage 5: Segmentation",
        "preproc",
        [
            "Load seg model + metadata (cached)",
            "Read RGB uint8",
            "Tiled inference w/ center-crop stitch",
            "Optional ROI from wing.geojson (skips background tiles)",
            "Per-channel normalization",
            "Gaussian smooth probabilities",
            "Argmax → class mask",
            "Polygonize mask → features",
            "Save detection.geojson",
        ],
    ),
    Node(
        "P6",
        "Stage 6: Wing rotation (optional)",
        "preproc",
        [
            "Fit affine from reliable landmarks",
            "Rotate un-masked image to canonical orientation",
            "Apply same affine to every produced GeoJSON (in-place)",
            "Optional mirror-correct for opposite-chirality wings",
            "→ rotated GeoJSONs feed identifyFeatures + overlay renders",
        ],
    ),
    # --- Preprocessing artifacts ---
    Node("WING_GJ", "wing.geojson", "artifact", ["single 'wing' feature (Stage 2)"], "data"),
    Node("LMK_GJ", "landmarks.geojson", "artifact", ["Point features"], "data"),
    Node("CHOPPED", "chopped image", "artifact", ["temp; deleted unless --keep-intermediates"], "data"),
    Node("SEG_GJ", "detection.geojson", "artifact", ["vein + intervein polygons"], "data"),
    # --- identifyFeatures steps ---
    Node(
        "I1",
        "Step 1: Parse inputs",
        "ifeat",
        [
            "Load vein / intervein polys",
            "Snap raw landmarks",
            "Compute wing outline (union)",
            "Estimate image_shape",
        ],
    ),
    Node(
        "I2",
        "Step 2: Build skeleton graph",
        "ifeat",
        [
            "Rasterize vein polys → vein_mask",
            "Boundary smoothing (optional)",
            "Skeletonize (RIDGE / medial-axis / …)",
            "Prune (distance-map / multi-scale)",
            "Collinear edge merge",
            "Gap bridging — pass 1 / 2 / 3",
            "Compute median_vein_width_px",
        ],
    ),
    Node(
        "I3",
        "Step 3: Anchor landmarks",
        "ifeat",
        [
            "Snap each landmark to nearest node",
            "Junction vs endpoint preference",
            "Store snap_distance",
        ],
    ),
    Node(
        "I4",
        "Step 4: Compute wing axis",
        "ifeat",
        [
            "Proximal / distal from landmarks",
            "Unit vector + length",
        ],
    ),
    Node(
        "I5",
        "Step 5: Call veins",
        "ifeat",
        [
            "Merge through crossvein junctions",
            "Detect costa (margin band)",
            "Propagate labels through deg-2",
            "Extend to distal landmarks",
            "Detect L6",
            "Detect crossveins (ACV / PCV)",
            "Label ectopic veins (EV*)",
            "Assign tissue polys (buffer vw)",
            "h-maxima split of intervein polys",
        ],
    ),
    Node(
        "I6",
        "Step 6: Call intervein regions",
        "ifeat",
        [
            "Buffer vein centerlines",
            "Match bounding-vein sets",
            "Tie-break by wing-axis position",
            "Absorb tiny fragments",
        ],
    ),
    # --- Result + outputs ---
    Node(
        "WR",
        "WingResult",
        "result",
        [
            "veins: list[VeinIdentification]",
            "intervein_regions: list[InterveinRegion]",
            "landmarks, wing_outline, warnings",
            "Inverse-rescale (Stage 1) back to original-pixel space",
        ],
        "data",
    ),
    Node(
        "MM",
        "measurementMaker: custom distances",
        "ifeat",
        [
            "User-defined landmark-pair distances",
            "Augments batch CSV with custom_<label>_um columns",
            "Fast path: emits CSV without running identifyFeatures",
        ],
    ),
    Node("OUT_GJ", "per-wing GeoJSON", "output", ["{stem}_output.geojson"], "data"),
    Node("OUT_OVERLAY", "vein + intervein overlay", "output", ["{stem}_overlay.png"], "data"),
    Node(
        "OUT_LMK_OV",
        "landmarks output",
        "output",
        [
            "{stem}_landmarks_overlay.png (rendered points)",
            "{stem}_landmarks.geojson (raw points, optional)",
        ],
        "data",
    ),
    Node(
        "OUT_SEG_OV",
        "segmentation output",
        "output",
        [
            "{stem}_segmentation_overlay.png (vein/intervein classes)",
            "{stem}_segmentation.geojson (raw polygons, optional)",
        ],
        "data",
    ),
    Node(
        "OUT_ISO",
        "isolated wing image",
        "output",
        ["{stem}_isolated.tif", "Masked single-wing image (Stage 2 artifact kept as output)"],
        "data",
    ),
    Node(
        "OUT_CHOP",
        "chopped image",
        "output",
        ["{stem}_chopped.tif", "Hinge-removed image (Stage 4 artifact kept as output)"],
        "data",
    ),
    Node("OUT_AP_OV", "AP compartment overlay", "output", ["{stem}_ap_overlay.png"], "data"),
    Node("OUT_CV_OV", "CV ratio overlay", "output", ["{stem}_cv_ratio_overlay.png"], "data"),
    Node(
        "OUT_CSV",
        "measurements csv",
        "output",
        [
            "area (wing, intervein regions, A/P compartments)",
            "length (wing, veins)",
            "custom measurements (from measurementMaker)",
        ],
        "data",
    ),
]

# edges: (src, dst, optional edge label, optional attrs dict)
# attrs is a dict of dot edge attributes; the two we use are:
#   constraint=false → edge drawn but doesn't pull endpoints into the same column
#   style=dashed     → visually indicates a weak/side reference
EDGES: list[tuple[str, str, str | None, dict[str, str] | None]] = [
    # Inputs → preprocessing
    ("IMG", "P1", None, None),
    # Stage 1 → Stage 2 (wing isolation reads the rescaled image)
    ("P1", "P2", None, None),
    ("WING_MODEL", "P2", None, None),
    ("P2", "WING_GJ", None, None),
    # Stage 2 (or Stage 1 when isolation off) → Stage 3 (landmarks)
    ("P2", "P3", "isolated image", None),
    ("P1", "P3", "(when isolation off)", {"style": "dashed"}),
    ("LMK_MODEL", "P3", None, None),
    ("P3", "LMK_GJ", None, None),
    # Landmarks → Stage 4 (hinge chop)
    ("LMK_GJ", "P4", None, None),
    ("P2", "P4", None, None),
    ("P4", "CHOPPED", None, None),
    # Stage 4 → Stage 5 (segmentation)
    ("CHOPPED", "P5", None, None),
    ("SEG_MODEL", "P5", None, None),
    ("WING_GJ", "P5", "ROI", {"style": "dashed", "constraint": "false"}),
    ("P5", "SEG_GJ", None, None),
    # Stage 6: rotation (optional) applies the same affine to image + every GeoJSON
    ("LMK_GJ", "P6", None, None),
    ("SEG_GJ", "P6", None, None),
    ("WING_GJ", "P6", None, None),
    # Preprocessing → identifyFeatures
    ("SEG_GJ", "I1", None, None),
    ("LMK_GJ", "I1", None, None),
    ("WING_GJ", "I1", None, {"style": "dashed"}),
    # CONFIG is placed in its own cluster adjacent to ifeat so these edges
    # stay short and don't create long diagonal crossings.
    ("CONFIG", "I2", None, {"constraint": "false"}),
    ("CONFIG", "I5", None, {"constraint": "false"}),
    ("CONFIG", "I6", None, {"constraint": "false"}),
    # identifyFeatures internal
    ("I1", "I2", "vein_polys", None),
    ("I1", "I3", "landmarks", None),
    ("I2", "I3", "SkeletonGraph", None),
    ("I3", "I4", "anchored", None),
    ("I1", "I5", "wing_outline", None),
    ("I2", "I5", "skeleton", None),
    ("I3", "I5", "landmarks", None),
    ("I4", "I5", "wing_axis", None),
    ("I5", "I6", "split polys + veins", None),
    # Results
    ("I5", "WR", "veins", None),
    ("I6", "WR", "regions", None),
    ("WR", "OUT_GJ", None, None),
    ("WR", "OUT_OVERLAY", None, None),
    ("WR", "OUT_AP_OV", None, None),
    ("WR", "OUT_CV_OV", None, None),
    ("WR", "OUT_CSV", None, None),
    # Landmarks output (overlay PNG + raw GeoJSON) derives from LMK_GJ.
    # constraint=false so dot routes this as a short side reference instead
    # of stretching the artifact row down toward outputs.
    ("LMK_GJ", "OUT_LMK_OV", None, {"constraint": "false"}),
    # Segmentation output (overlay PNG + raw GeoJSON) derives from SEG_GJ —
    # same treatment.
    ("SEG_GJ", "OUT_SEG_OV", None, {"constraint": "false"}),
    # Isolated wing image (Stage 2 byproduct kept as output when requested).
    # The image itself isn't a separate artifact in the map — it's the
    # "isolated image" data already flowing P2 → P3. constraint=false to
    # keep this as a side reference rather than reshaping the preproc spine.
    ("P2", "OUT_ISO", None, {"constraint": "false"}),
    # Chopped image (Stage 4 byproduct kept as output when requested).
    ("CHOPPED", "OUT_CHOP", None, {"constraint": "false"}),
    # measurementMaker: post-CSV augmentation with user-defined landmark distances.
    # Also handles the fast path where identifyFeatures is skipped entirely and
    # the CSV is built directly from landmarks.
    ("LMK_GJ", "MM", None, None),
    ("CONFIG", "MM", None, {"style": "dashed", "constraint": "false"}),
    ("MM", "OUT_CSV", "custom distances", None),
]


# ---------------------------------------------------------------------------
# DOT generation + layout
# ---------------------------------------------------------------------------


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_label(s: str) -> str:
    """Label-specific escape: preserves `\\l` / `\\n` / `\\r` directives."""
    return s.replace('"', '\\"')


# Pairs of node ids that should sit on the same rank inside their cluster, to
# break tall single-file columns into 2-wide rows. Each pair is rendered as a
# `{rank=same; A; B;}` subgroup.
_RANK_PAIRS: dict[str, list[list[str]]] = {
    "preproc": [["P1", "P2"], ["P3", "P4"], ["P5", "P6"]],
    "ifeat": [["I1", "I2"], ["I3", "I4"], ["I5", "I6"], ["MM"]],
    "artifact": [["WING_GJ", "LMK_GJ", "CHOPPED", "SEG_GJ"]],
    "output": [
        ["OUT_GJ", "OUT_OVERLAY", "OUT_LMK_OV", "OUT_SEG_OV"],
        ["OUT_ISO", "OUT_CHOP", "OUT_AP_OV", "OUT_CV_OV", "OUT_CSV"],
    ],
    # Bundled inputs (no IMG — that lives in user_input now). Pack into one
    # 4-wide row so the cluster reads as a single horizontal strip.
    "input": [["WING_MODEL", "LMK_MODEL", "SEG_MODEL", "CONFIG"]],
}


# Per-node minimum width (inches) for boxes whose bold vispy title is wider
# than graphviz's substep-based sizing would compute. Forces graphviz to
# reserve a wider box.
_NODE_MIN_WIDTH: dict[str, float] = {
    "P3": 4.6,  # Stage 3: Landmark detection
    "I4": 3.5,  # Step 4: Compute wing axis
    "I6": 7.5,  # Step 6: Call intervein regions
}


def build_dot() -> str:
    # fontsize 13 + margin "0.30,0.18" on nodes leaves enough slack inside the
    # graphviz-sized box that vispy's text renderer (different font metrics)
    # still lands comfortably inside the rectangle.
    lines: list[str] = [
        "digraph G {",
        "  rankdir=TB;",
        "  nodesep=0.30;",
        "  ranksep=0.50;",
        "  newrank=true;",
        "  compound=true;",
        "  splines=ortho;",
        '  graph [fontname="Helvetica"];',
        # fontsize is intentionally inflated (vs the 10/7.5pt vispy uses for
        # title/substeps) so graphviz computes box widths with extra slack —
        # vispy's text rendering (especially the bold title) is wider per
        # character than graphviz's font metrics anticipate.
        '  node  [fontname="Helvetica", fontsize=30, shape=box, ' 'style="rounded,filled", margin="0.30,0.18"];',
        '  edge  [fontname="Helvetica", fontsize=10];',
    ]

    by_group: dict[str, list[Node]] = {g: [] for g in GROUPS}
    for n in NODES:
        by_group.setdefault(n.group, []).append(n)

    def _emit_node(n: Node, meta: dict, indent: str) -> str:
        label_lines = [n.title] + ([""] + [f"• {s}" for s in n.substeps] if n.substeps else [])
        label = "\\l".join(label_lines) + "\\l"
        shape = "ellipse" if n.kind == "data" else "box"
        min_width = _NODE_MIN_WIDTH.get(n.id)
        width_attr = f", width={min_width}" if min_width else ""
        return (
            f'{indent}{n.id} [label="{_escape_label(label)}", '
            f'shape={shape}, fillcolor="{meta["fill"]}", color="{meta["stroke"]}"{width_attr}];'
        )

    for group_id, members in by_group.items():
        if not members:
            continue
        meta = GROUPS[group_id]
        if not meta.get("cluster", True):
            # Non-cluster group: emit nodes at top level (no subgraph wrapper).
            for n in members:
                lines.append(_emit_node(n, meta, "  "))
            continue
        lines.append(f"  subgraph cluster_{group_id} {{")
        lines.append(f'    label="{_escape(meta["label"])}";')
        lines.append(f'    style="rounded,filled";')
        lines.append(f'    color="{meta["stroke"]}";')
        lines.append(f'    fillcolor="#ffffff00";')
        lines.append(f"    fontsize=13;")
        # Cluster margin pads inside the cluster border so adjacent clusters
        # (e.g. user_input vs input) don't collide.
        lines.append(f"    margin=20;")
        for n in members:
            lines.append(_emit_node(n, meta, "    "))
        # Pair adjacent stages into 2-wide rows to avoid tall single-file columns.
        rank_pairs = _RANK_PAIRS.get(group_id, [])
        for pair in rank_pairs:
            ids_in = [nid for nid in pair if nid in {m.id for m in members}]
            if len(ids_in) >= 2:
                lines.append(f"    {{rank=same; {'; '.join(ids_in)};}}")
        lines.append("  }")

    for src, dst, lbl, attrs in EDGES:
        attr_parts: list[str] = []
        if lbl:
            attr_parts.append(f'label="{_escape(lbl)}"')
        if attrs:
            for k, v in attrs.items():
                attr_parts.append(f'{k}="{_escape(v)}"')
        if attr_parts:
            lines.append(f'  {src} -> {dst} [{", ".join(attr_parts)}];')
        else:
            lines.append(f"  {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)


@dataclass
class LaidOutNode:
    id: str
    x: float
    y: float
    w: float
    h: float
    label: str
    shape: str


@dataclass
class LaidOutEdge:
    src: str
    dst: str
    spline: list[tuple[float, float]]
    label: str | None = None
    label_pos: tuple[float, float] | None = None


@dataclass
class Layout:
    width: float
    height: float
    nodes: dict[str, LaidOutNode]
    edges: list[LaidOutEdge]


_PLAIN_NODE_RE = re.compile(r'^node\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"((?:[^"\\]|\\.)*)"')


def run_dot_plain(dot_src: str) -> str:
    result = subprocess.run(
        ["dot", "-Tplain"],
        input=dot_src,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_plain(plain: str) -> Layout:
    """Parse `dot -Tplain` output.

    Grammar (from man dot):
        graph scale width height
        node name x y w h label style shape color fillcolor
        edge tail head n x1 y1 x2 y2 ... xn yn [label xl yl] style color
        stop
    """
    nodes: dict[str, LaidOutNode] = {}
    edges: list[LaidOutEdge] = []
    width = height = 0.0

    def split_tokens(line: str) -> list[str]:
        # Tokens are whitespace-separated; quoted strings can contain spaces & escaped quotes.
        out = []
        i = 0
        while i < len(line):
            c = line[i]
            if c.isspace():
                i += 1
                continue
            if c == '"':
                j = i + 1
                buf = []
                while j < len(line):
                    if line[j] == "\\" and j + 1 < len(line):
                        buf.append(line[j + 1])
                        j += 2
                        continue
                    if line[j] == '"':
                        break
                    buf.append(line[j])
                    j += 1
                out.append("".join(buf))
                i = j + 1
            else:
                j = i
                while j < len(line) and not line[j].isspace():
                    j += 1
                out.append(line[i:j])
                i = j
        return out

    for raw in plain.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = split_tokens(line)
        kind = toks[0]
        if kind == "graph":
            # graph scale width height
            width = float(toks[2])
            height = float(toks[3])
        elif kind == "node":
            name = toks[1]
            x = float(toks[2])
            y = float(toks[3])
            w = float(toks[4])
            h = float(toks[5])
            label = toks[6]
            shape = toks[8] if len(toks) > 8 else "box"
            nodes[name] = LaidOutNode(name, x, y, w, h, label, shape)
        elif kind == "edge":
            # edge tail head n x1 y1 ... xn yn [label xl yl] style color
            tail = toks[1]
            head = toks[2]
            n_pts = int(toks[3])
            spline = []
            idx = 4
            for _ in range(n_pts):
                spline.append((float(toks[idx]), float(toks[idx + 1])))
                idx += 2
            label = None
            label_pos = None
            # Optional: "label" xl yl
            if idx < len(toks):
                try:
                    float(toks[idx])  # style? or label?
                    # style token — no label present
                except ValueError:
                    label = toks[idx]
                    idx += 1
                    if idx + 1 < len(toks):
                        try:
                            label_pos = (float(toks[idx]), float(toks[idx + 1]))
                            idx += 2
                        except ValueError:
                            pass
            edges.append(LaidOutEdge(tail, head, spline, label, label_pos))
        elif kind == "stop":
            break

    return Layout(width=width, height=height, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Rendering (vispy)
# ---------------------------------------------------------------------------


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    if len(h) == 8:  # RRGGBBAA
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        return (r / 255, g / 255, b / 255, a / 255)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255, g / 255, b / 255, alpha)


def _add_bridge_turns_into_node_rects(layout: "Layout") -> None:
    """Some edges, after spreading + snap, have their endpoint clamped onto
    one side of the dst rect but still outside on the perpendicular axis —
    e.g. WING_GJ → P5 ends at P5's left edge but 0.1 in below its bottom.
    Append a perpendicular Bezier segment so the line turns 90° and meets
    the rect boundary. Same logic for spline starts that don't quite enter
    the src rect.
    """
    for e in layout.edges:
        if not e.spline or len(e.spline) < 4:
            continue
        spline = [tuple(p) for p in e.spline]

        # --- end side
        dst = layout.nodes.get(e.dst)
        if dst is not None:
            dx0, dx1 = dst.x - dst.w / 2, dst.x + dst.w / 2
            dy0, dy1 = dst.y - dst.h / 2, dst.y + dst.h / 2
            ex, ey = spline[-1]
            inside_x = dx0 - 0.01 <= ex <= dx1 + 0.01
            inside_y = dy0 - 0.01 <= ey <= dy1 + 0.01
            if not (inside_x and inside_y):
                last4 = spline[-4:]
                xs = [p[0] for p in last4]
                ys = [p[1] for p in last4]
                is_h = max(ys) - min(ys) < 0.05 and max(xs) - min(xs) > 0.05
                is_v = max(xs) - min(xs) < 0.05 and max(ys) - min(ys) > 0.05
                target = None
                if is_h and not inside_y:
                    target = (ex, dy0 if ey < dy0 else dy1)
                elif is_v and not inside_x:
                    target = (dx0 if ex < dx0 else dx1, ey)
                if target is not None and target != (ex, ey):
                    p0 = (ex, ey)
                    p3 = target
                    p1 = (p0[0] + (p3[0] - p0[0]) / 3.0, p0[1] + (p3[1] - p0[1]) / 3.0)
                    p2 = (p0[0] + 2 * (p3[0] - p0[0]) / 3.0, p0[1] + 2 * (p3[1] - p0[1]) / 3.0)
                    spline.extend([p1, p2, p3])

        # --- start side
        src = layout.nodes.get(e.src)
        if src is not None:
            sx0, sx1 = src.x - src.w / 2, src.x + src.w / 2
            sy0, sy1 = src.y - src.h / 2, src.y + src.h / 2
            sx, sy = spline[0]
            inside_x = sx0 - 0.01 <= sx <= sx1 + 0.01
            inside_y = sy0 - 0.01 <= sy <= sy1 + 0.01
            if not (inside_x and inside_y):
                first4 = spline[:4]
                xs = [p[0] for p in first4]
                ys = [p[1] for p in first4]
                is_h = max(ys) - min(ys) < 0.05 and max(xs) - min(xs) > 0.05
                is_v = max(xs) - min(xs) < 0.05 and max(ys) - min(ys) > 0.05
                target = None
                if is_h and not inside_y:
                    target = (sx, sy0 if sy < sy0 else sy1)
                elif is_v and not inside_x:
                    target = (sx0 if sx < sx0 else sx1, sy)
                if target is not None and target != (sx, sy):
                    p3 = (sx, sy)
                    p0 = target
                    p1 = (p0[0] + (p3[0] - p0[0]) / 3.0, p0[1] + (p3[1] - p0[1]) / 3.0)
                    p2 = (p0[0] + 2 * (p3[0] - p0[0]) / 3.0, p0[1] + 2 * (p3[1] - p0[1]) / 3.0)
                    spline = [p0, p1, p2] + spline

        e.spline = spline


def _snap_endpoints_to_node_rects(layout: "Layout") -> None:
    """Pull each edge's spline[0] and spline[-1] onto the boundary of their
    src/dst node rect so the arrow tip actually meets the box.

    `dot -Tplain` reserves ~0.15 in for an arrowhead by ending the spline
    that far outside the target. Our vispy arrowhead is only ~0.10 in, so
    the tip floats in empty space and the line visually appears to leave
    the box from a corner — looks like an awkward angle even though the
    underlying geometry is a perfect L.

    The snap clamps the endpoint to the node rect while preserving the
    direction of the adjacent Bezier segment:
      - if the last 4 control points share x (vertical), only y is clamped
      - if they share y (horizontal), only x is clamped
      - otherwise (rare) both are clamped — the segment wasn't pure ortho
        to start with, so we can't make it worse.
    No diagonals are ever introduced.
    """

    def _seg_dir(ctrl):
        xs = [p[0] for p in ctrl]
        ys = [p[1] for p in ctrl]
        if max(xs) - min(xs) < 0.05:
            return "vertical"
        if max(ys) - min(ys) < 0.05:
            return "horizontal"
        return None

    def _reinterp(p0, p3):
        return (
            (p0[0] + (p3[0] - p0[0]) / 3.0, p0[1] + (p3[1] - p0[1]) / 3.0),
            (p0[0] + 2 * (p3[0] - p0[0]) / 3.0, p0[1] + 2 * (p3[1] - p0[1]) / 3.0),
        )

    for e in layout.edges:
        if not e.spline or len(e.spline) < 4:
            continue
        spline = [tuple(p) for p in e.spline]
        # start: clamp to source rect along adjacent segment's axis
        src = layout.nodes.get(e.src)
        if src is not None:
            x, y = spline[0]
            sx0, sx1 = src.x - src.w / 2, src.x + src.w / 2
            sy0, sy1 = src.y - src.h / 2, src.y + src.h / 2
            direction = _seg_dir(spline[:4])
            new_x, new_y = x, y
            if direction == "vertical":
                if y < sy0:
                    new_y = sy0
                elif y > sy1:
                    new_y = sy1
            elif direction == "horizontal":
                if x < sx0:
                    new_x = sx0
                elif x > sx1:
                    new_x = sx1
            else:
                new_x = max(sx0, min(sx1, x))
                new_y = max(sy0, min(sy1, y))
            if (new_x, new_y) != (x, y):
                spline[0] = (new_x, new_y)
                p1, p2 = _reinterp(spline[0], spline[3])
                spline[1] = p1
                spline[2] = p2
        # end: clamp to dst rect along adjacent segment's axis
        dst = layout.nodes.get(e.dst)
        if dst is not None:
            x, y = spline[-1]
            dx0, dx1 = dst.x - dst.w / 2, dst.x + dst.w / 2
            dy0, dy1 = dst.y - dst.h / 2, dst.y + dst.h / 2
            direction = _seg_dir(spline[-4:])
            new_x, new_y = x, y
            if direction == "vertical":
                if y < dy0:
                    new_y = dy0
                elif y > dy1:
                    new_y = dy1
            elif direction == "horizontal":
                if x < dx0:
                    new_x = dx0
                elif x > dx1:
                    new_x = dx1
            else:
                new_x = max(dx0, min(dx1, x))
                new_y = max(dy0, min(dy1, y))
            if (new_x, new_y) != (x, y):
                spline[-1] = (new_x, new_y)
                p1, p2 = _reinterp(spline[-4], spline[-1])
                spline[-3] = p1
                spline[-2] = p2
        e.spline = spline


def _spread_parallel_horizontal_segments(
    layout: "Layout",
    *,
    x_overlap_min: float = 0.5,
    y_proximity_max: float = 0.45,
    min_separation: float = 0.50,
) -> None:
    """Find bundles of near-horizontal Bezier segments running parallel & close in y,
    then push them apart so each line is clearly distinct.

    `dot -Tplain` with `splines=ortho` often crams 3+ parallel horizontals into
    the narrow gap between two ranks. We:
      1. Collect every "long horizontal" Bezier segment (all 4 control points
         share y within ±0.05 in, x-span > 0.5 in).
      2. Bundle segments that overlap in x and are within `y_proximity_max` of
         each other.
      3. Spread each bundle around its mean y so consecutive lines are at least
         `min_separation` apart.
      4. After shifting a horizontal segment, re-interpolate the inner control
         points of the adjacent (vertical) Bezier segments so each ortho turn
         stays a clean straight line rather than wiggling.
      5. Move the edge's label_pos along with its segment so the label sticks
         with its line.
    """
    Y_TOL = 0.05  # tolerance for "same y" (horizontal segment)
    X_MIN_SPAN = 0.5  # minimum x-span to qualify as a long horizontal

    # Step 1 — collect horizontal Bezier segments.
    segs = []
    for edge_idx, e in enumerate(layout.edges):
        spline = list(e.spline)
        n_segs = (len(spline) - 1) // 3
        for k in range(n_segs):
            ctrl = [spline[3 * k + j] for j in range(4)]
            ys = [p[1] for p in ctrl]
            xs = [p[0] for p in ctrl]
            if max(ys) - min(ys) > Y_TOL:
                continue
            x_span = max(xs) - min(xs)
            if x_span < X_MIN_SPAN:
                continue
            segs.append(
                {
                    "edge_idx": edge_idx,
                    "seg_idx": k,
                    "n_segs": n_segs,
                    "x0": min(xs),
                    "x1": max(xs),
                    "y": sum(ys) / 4.0,
                }
            )

    # Step 2 — group by x-overlap + y-proximity (transitive closure).
    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if abs(segs[i]["y"] - segs[j]["y"]) > y_proximity_max:
                continue
            x_overlap = min(segs[i]["x1"], segs[j]["x1"]) - max(segs[i]["x0"], segs[j]["x0"])
            if x_overlap < x_overlap_min:
                continue
            union(i, j)

    bundles: dict[int, list[int]] = {}
    for i in range(len(segs)):
        bundles.setdefault(find(i), []).append(i)

    # Step 3 — for bundles of 2+, spread along y around the bundle mean,
    # clamped to the y-channel between source bottoms and target tops so
    # horizontals never end up overlapping their source or target nodes.
    SAFETY = 0.10  # inches between a horizontal and the nearest node edge

    def _channel(members):
        """Allowed (y_min, y_max) for this bundle's horizontals."""
        floor = float("-inf")
        ceiling = float("inf")
        for idx in members:
            e = layout.edges[segs[idx]["edge_idx"]]
            src = layout.nodes.get(e.src)
            dst = layout.nodes.get(e.dst)
            if src is None or dst is None:
                continue
            if src.y > dst.y:  # downward edge
                ceiling = min(ceiling, src.y - src.h / 2)
                floor = max(floor, dst.y + dst.h / 2)
            else:  # upward
                ceiling = min(ceiling, dst.y - dst.h / 2)
                floor = max(floor, src.y + src.h / 2)
        return floor + SAFETY, ceiling - SAFETY

    shifts: list[tuple[int, int, float]] = []  # (edge_idx, seg_idx, dy)
    for members in bundles.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda idx: segs[idx]["y"])
        ys = [segs[idx]["y"] for idx in members]
        mean_y = sum(ys) / len(ys)
        n = len(members)
        y_min_allowed, y_max_allowed = _channel(members)
        available = y_max_allowed - y_min_allowed
        # If the channel is too narrow for the requested spread, shrink it.
        effective_sep = min(min_separation, available / max(n - 1, 1)) if available > 0 else min_separation
        proposed = [mean_y + (k - (n - 1) / 2.0) * effective_sep for k in range(n)]
        # Clamp the bundle as a whole (preserving the spread) so the top/bottom
        # don't poke into the source/target node rectangles.
        if proposed[-1] > y_max_allowed:
            shift = y_max_allowed - proposed[-1]
            proposed = [y + shift for y in proposed]
        if proposed[0] < y_min_allowed:
            shift = y_min_allowed - proposed[0]
            proposed = [y + shift for y in proposed]
        for rank, idx in enumerate(members):
            dy = proposed[rank] - segs[idx]["y"]
            if abs(dy) > 1e-4:
                shifts.append((segs[idx]["edge_idx"], segs[idx]["seg_idx"], dy))

    # Step 4 — apply shifts. We mutate edge.spline (a list of tuples) in place
    # by rebuilding it. Then reinterpolate adjacent vertical segments so they
    # stay straight after the corner moves.
    def _reinterp_inner(p0, p3):
        return (
            (p0[0] + (p3[0] - p0[0]) / 3.0, p0[1] + (p3[1] - p0[1]) / 3.0),
            (p0[0] + 2 * (p3[0] - p0[0]) / 3.0, p0[1] + 2 * (p3[1] - p0[1]) / 3.0),
        )

    # Group shifts by edge so a multi-segment edge gets a single rebuild.
    per_edge: dict[int, list[tuple[int, float]]] = {}
    for edge_idx, seg_idx, dy in shifts:
        per_edge.setdefault(edge_idx, []).append((seg_idx, dy))

    for edge_idx, sh in per_edge.items():
        spline = [tuple(p) for p in layout.edges[edge_idx].spline]
        n_segs = (len(spline) - 1) // 3
        # Apply each shift: bump y of the 4 control points of seg_idx by dy.
        for seg_idx, dy in sh:
            for j in range(4):
                px, py = spline[3 * seg_idx + j]
                spline[3 * seg_idx + j] = (px, py + dy)
            # Adjacent segments: rebuild their inner control points so the
            # straight-line direction is preserved (no wiggle).
            if seg_idx > 0:
                p0 = spline[3 * (seg_idx - 1)]
                p3 = spline[3 * seg_idx]  # already shifted
                p1, p2 = _reinterp_inner(p0, p3)
                spline[3 * (seg_idx - 1) + 1] = p1
                spline[3 * (seg_idx - 1) + 2] = p2
            if seg_idx < n_segs - 1:
                p0 = spline[3 * (seg_idx + 1)]  # already shifted
                p3 = spline[3 * (seg_idx + 1) + 3]
                p1, p2 = _reinterp_inner(p0, p3)
                spline[3 * (seg_idx + 1) + 1] = p1
                spline[3 * (seg_idx + 1) + 2] = p2

            # Step 5 — drag the edge label with its segment when applicable.
            e = layout.edges[edge_idx]
            if e.label_pos is not None:
                lx, ly = e.label_pos
                # Was the label on this segment (in x range and near old y)?
                seg_info = next(
                    (s for s in segs if s["edge_idx"] == edge_idx and s["seg_idx"] == seg_idx),
                    None,
                )
                if seg_info and seg_info["x0"] - 0.1 <= lx <= seg_info["x1"] + 0.1:
                    if abs(ly - seg_info["y"]) < 0.2:
                        e.label_pos = (lx, ly + dy)

        layout.edges[edge_idx].spline = spline


def _reseat_fallback_edge_labels(layout: "Layout") -> None:
    """`dot -Tplain` falls back to label_pos=(0, 0.11111) when it can't find
    a good slot. When multiple edges all hit this fallback, their labels
    stack in the bottom-left corner of the canvas, completely illegible.

    Replace any fallback position with the geometric midpoint of the edge's
    spline. The next displacement pass will then nudge it off any node it
    happens to land inside.
    """
    for e in layout.edges:
        if e.label is None or e.label_pos is None:
            continue
        lx, ly = e.label_pos
        if abs(lx) < 0.01 and abs(ly - 0.11111) < 0.02:
            spline = list(e.spline)
            if not spline:
                continue
            mid = spline[len(spline) // 2]
            e.label_pos = (float(mid[0]), float(mid[1]))


def _displace_labels_off_obstacles(
    layout: "Layout",
    *,
    safety: float = 0.10,
    max_iters: int = 4,
) -> None:
    """If an edge label's bounding box overlaps any node OR any other edge
    label, shift the label along whichever axis (up/down/left/right) clears
    every overlap with the smallest displacement. Iterated since moving one
    label may unblock another's preferred destination.

    `dot` places some edge labels on the rank between two node rows, which
    works when the rank-gap is wide enough; when it isn't (e.g. ``wing_outline``
    landing in the narrow gap between I4 and I3, or ``anchored`` sitting
    on MM), the label ends up tucked under a box. Other labels (e.g.
    ``wing_axis`` next to ``split polys + veins``) can also land on top of
    each other in the same rank-gap.
    """
    SCALE = 72.0
    PX_TO_INCH = 3.27 / SCALE
    EDGE_FONT_PX = 6.5
    CW_EDGE = 0.55
    LH = EDGE_FONT_PX * PX_TO_INCH

    nodes = list(layout.nodes.values())

    def label_w(label):
        return len(label) * EDGE_FONT_PX * CW_EDGE * PX_TO_INCH

    def has_overlap(cx, cy, lw, current_edge_idx):
        lx0, lx1 = cx - lw / 2, cx + lw / 2
        ly0, ly1 = cy - LH / 2, cy + LH / 2
        for n in nodes:
            nx0 = n.x - n.w / 2 - safety
            nx1 = n.x + n.w / 2 + safety
            ny0 = n.y - n.h / 2 - safety
            ny1 = n.y + n.h / 2 + safety
            if lx0 < nx1 and lx1 > nx0 and ly0 < ny1 and ly1 > ny0:
                return True
        for ei, e in enumerate(layout.edges):
            if ei == current_edge_idx:
                continue
            if not e.label or e.label_pos is None:
                continue
            ox, oy = e.label_pos
            ow = label_w(e.label)
            ox0 = ox - ow / 2 - safety
            ox1 = ox + ow / 2 + safety
            oy0 = oy - LH / 2 - safety
            oy1 = oy + LH / 2 + safety
            if lx0 < ox1 and lx1 > ox0 and ly0 < oy1 and ly1 > oy0:
                return True
        return False

    for _ in range(max_iters):
        any_moved = False
        for ei, e in enumerate(layout.edges):
            if not e.label or e.label_pos is None:
                continue
            lx, ly = e.label_pos
            lw = label_w(e.label)
            if not has_overlap(lx, ly, lw, ei):
                continue
            best = None  # (distance, new_x, new_y)
            for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
                for steps in range(1, 80):
                    s = steps * 0.1
                    tx, ty = lx + dx * s, ly + dy * s
                    if not has_overlap(tx, ty, lw, ei):
                        if best is None or s < best[0]:
                            best = (s, tx, ty)
                        break
            if best:
                e.label_pos = (best[1], best[2])
                any_moved = True
        if not any_moved:
            break


def _lift_horizontals_above_crossed_labels(
    layout: "Layout",
    *,
    safety: float = 0.08,
    max_iters: int = 3,
) -> None:
    """For each horizontal Bezier segment that crosses a label box (cluster or
    edge) belonging to a DIFFERENT edge, shift the segment upward so it clears
    the label. The shift is clamped to stay just below the source node's bottom
    edge — we can't realistically push a horizontal above the box it exits.

    Constants mirror the render() values: cluster labels = 12 px bold, edge
    labels = 6.5 px italic. The scene-units → pixels ratio at the default
    frame-fit zoom is ~3.27, so a pixel of text ≈ 0.045 graphviz-inches.

    Runs up to `max_iters` passes; an earlier shift can introduce a new
    crossing that the next pass cleans up.
    """
    SCALE = 72.0
    SCENE_PER_PIXEL = 3.27
    PX_TO_INCH = SCENE_PER_PIXEL / SCALE
    CLUSTER_FONT_PX = 12.0
    EDGE_FONT_PX = 6.5
    CW_EDGE = 0.55
    CW_CLUSTER = 0.62
    CLUSTER_BASELINE_OFFSET = 17.6 / SCALE  # render code treats 17.6 scene units

    nodes_meta = {n.id: n for n in NODES}

    def _cluster_label_box(label, members):
        xs = [n.x for n in members]
        ws = [n.w for n in members]
        ys_ = [n.y for n in members]
        hs = [n.h for n in members]
        x0 = min(xs[i] - ws[i] / 2 for i in range(len(members))) - 0.20
        x1 = max(xs[i] + ws[i] / 2 for i in range(len(members))) + 0.20
        y1_inches = max(ys_[i] + hs[i] / 2 for i in range(len(members))) + 0.50
        baseline_inches = y1_inches - CLUSTER_BASELINE_OFFSET
        ascent_inches = CLUSTER_FONT_PX * 0.80 * PX_TO_INCH
        descent_inches = CLUSTER_FONT_PX * 0.20 * PX_TO_INCH
        w = len(label) * CLUSTER_FONT_PX * CW_CLUSTER * PX_TO_INCH
        cx = (x0 + x1) / 2
        return cx - w / 2, cx + w / 2, baseline_inches - descent_inches, baseline_inches + ascent_inches

    def _edge_label_box(label, cx, cy):
        w = len(label) * EDGE_FONT_PX * CW_EDGE * PX_TO_INCH
        h = EDGE_FONT_PX * PX_TO_INCH
        return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2

    def _build_label_boxes():
        boxes = []  # (kind, owner_id_or_edge_idx, x0, x1, y0, y1)
        clusters_grouped = {}
        for nid, ln in layout.nodes.items():
            g = nodes_meta[nid].group
            if not GROUPS[g].get("cluster", True):
                continue
            clusters_grouped.setdefault(g, []).append(ln)
        for g, members in clusters_grouped.items():
            label = GROUPS[g].get("label")
            if not label:
                continue
            boxes.append(("cluster", g, *_cluster_label_box(label, members)))
        for ei, e in enumerate(layout.edges):
            if e.label and e.label_pos:
                boxes.append(("edge", ei, *_edge_label_box(e.label, e.label_pos[0], e.label_pos[1])))
        return boxes

    def _horizontal_segs():
        out = []
        for ei, e in enumerate(layout.edges):
            spline = list(e.spline)
            n_segs = (len(spline) - 1) // 3
            for k in range(n_segs):
                ctrl = [spline[3 * k + j] for j in range(4)]
                ys = [p[1] for p in ctrl]
                xs = [p[0] for p in ctrl]
                if max(ys) - min(ys) > 0.05:
                    continue
                if max(xs) - min(xs) < 0.5:
                    continue
                out.append((ei, k, n_segs, sum(ys) / 4, min(xs), max(xs)))
        return out

    def _reinterp_inner(p0, p3):
        return (
            (p0[0] + (p3[0] - p0[0]) / 3.0, p0[1] + (p3[1] - p0[1]) / 3.0),
            (p0[0] + 2 * (p3[0] - p0[0]) / 3.0, p0[1] + 2 * (p3[1] - p0[1]) / 3.0),
        )

    STAGGER = 0.30  # additional separation between two segments lifted above the same label

    for _ in range(max_iters):
        boxes = _build_label_boxes()
        segs = _horizontal_segs()
        # First pass — compute everyone's proposed target_y.
        proposals = []  # (ei, seg_idx, n_segs, sy, sx0, sx1, base_target_y)
        for ei, seg_idx, n_segs, sy, sx0, sx1 in segs:
            crossed_label_tops = []
            for kind, owner, lx0, lx1, ly0, ly1 in boxes:
                if kind == "edge" and owner == ei:
                    continue
                if ly0 <= sy <= ly1 and sx0 < lx1 and sx1 > lx0:
                    crossed_label_tops.append(ly1)
            if not crossed_label_tops:
                continue
            target_y = max(crossed_label_tops) + safety
            proposals.append([ei, seg_idx, n_segs, sy, sx0, sx1, target_y])
        # Stagger groups of proposals whose target_y is the same (e.g. multiple
        # horizontals all lifted above the same cluster label). Sort each
        # group by current sy so the one originally lower stays lower.
        proposals.sort(key=lambda p: (round(p[6], 1), p[3]))
        i = 0
        while i < len(proposals):
            j = i
            while j < len(proposals) and abs(proposals[j][6] - proposals[i][6]) < 0.05:
                # Same target_y AND x-range overlap with anyone in [i:j+1]?
                if j > i:
                    overlaps_any = any(
                        min(proposals[j][5], proposals[k][5]) - max(proposals[j][4], proposals[k][4]) > 0.5
                        for k in range(i, j)
                    )
                    if not overlaps_any:
                        break
                j += 1
            # Group is proposals[i:j]; stagger upward.
            for k_off, idx in enumerate(range(i, j)):
                proposals[idx][6] += k_off * STAGGER
            i = j
        # Second pass — apply (with source-bottom cap).
        any_change = False
        for ei, seg_idx, n_segs, sy, sx0, sx1, target_y in proposals:
            e = layout.edges[ei]
            src = layout.nodes.get(e.src)
            dst = layout.nodes.get(e.dst)
            if src is not None:
                if src.y > (dst.y if dst else float("-inf")):
                    cap = src.y - src.h / 2 - safety
                    if target_y > cap:
                        target_y = cap
            dy = target_y - sy
            if abs(dy) < 0.01:
                continue
            spline = [tuple(p) for p in e.spline]
            for j in range(4):
                px, py = spline[3 * seg_idx + j]
                spline[3 * seg_idx + j] = (px, py + dy)
            if seg_idx > 0:
                p0 = spline[3 * (seg_idx - 1)]
                p3 = spline[3 * seg_idx]
                p1, p2 = _reinterp_inner(p0, p3)
                spline[3 * (seg_idx - 1) + 1] = p1
                spline[3 * (seg_idx - 1) + 2] = p2
            if seg_idx < n_segs - 1:
                p0 = spline[3 * (seg_idx + 1)]
                p3 = spline[3 * (seg_idx + 1) + 3]
                p1, p2 = _reinterp_inner(p0, p3)
                spline[3 * (seg_idx + 1) + 1] = p1
                spline[3 * (seg_idx + 1) + 2] = p2
            e.spline = spline
            # Drag the edge's own label with its segment if applicable.
            if e.label_pos is not None and sx0 - 0.1 <= e.label_pos[0] <= sx1 + 0.1:
                if abs(e.label_pos[1] - (sy + 0.18)) < 0.30:  # label was lifted 0.18 above sy
                    e.label_pos = (e.label_pos[0], e.label_pos[1] + dy)
            any_change = True
        if not any_change:
            break


def _lift_labels_off_horizontal_segments(layout: "Layout", lift: float = 0.18) -> None:
    """For each labeled edge whose label sits on a roughly horizontal Bezier
    segment, nudge the label `lift` inches upward so the line doesn't
    visually pass through the italic text.

    A segment qualifies as "horizontal" if its 4 control points share a y
    value within a small tolerance. We only lift when the label's x is within
    the segment's x-span and its y is within 0.2 in of the segment's y.
    Labels on vertical segments or already off-axis are left alone.
    """
    Y_TOL = 0.05
    for e in layout.edges:
        if not e.label or e.label_pos is None:
            continue
        lx, ly = e.label_pos
        spline = list(e.spline)
        n_segs = (len(spline) - 1) // 3
        for k in range(n_segs):
            ctrl = [spline[3 * k + j] for j in range(4)]
            ys = [p[1] for p in ctrl]
            xs = [p[0] for p in ctrl]
            if max(ys) - min(ys) > Y_TOL:
                continue
            seg_y = sum(ys) / 4
            if not (min(xs) - 0.1 <= lx <= max(xs) + 0.1):
                continue
            if abs(ly - seg_y) > 0.2:
                continue
            e.label_pos = (lx, ly + lift)
            break


def _clip_polyline_to_node_rects(
    pts: np.ndarray,
    src_rect: tuple[float, float, float, float] | None,
    dst_rect: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Trim leading/trailing polyline points that lie inside the src/dst node rects.

    `dot -Tplain` sometimes emits an edge whose first Bezier control point sits
    inside its source node (graphviz keeps it for spline-shape continuity but
    visually the line then leaks into the box). Same issue at the target end.
    We walk in from each end, find where the polyline first crosses the rect
    boundary, and splice in the boundary-crossing point so the rendered line
    starts/ends exactly at the node edge.
    """

    def inside(p, rect):
        xmin, ymin, xmax, ymax = rect
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    def segment_rect_exit(p_in, p_out, rect):
        # Smallest t in (0, 1] where the segment from p_in (inside) to p_out
        # (outside) crosses the rect boundary.
        xmin, ymin, xmax, ymax = rect
        dx = p_out[0] - p_in[0]
        dy = p_out[1] - p_in[1]
        ts = []
        if dx > 0:
            ts.append((xmax - p_in[0]) / dx)
        elif dx < 0:
            ts.append((xmin - p_in[0]) / dx)
        if dy > 0:
            ts.append((ymax - p_in[1]) / dy)
        elif dy < 0:
            ts.append((ymin - p_in[1]) / dy)
        ts = [t for t in ts if 0 < t <= 1.0 + 1e-9]
        if not ts:
            return (float(p_out[0]), float(p_out[1]))
        t = min(ts)
        return (float(p_in[0] + t * dx), float(p_in[1] + t * dy))

    pts_list = [tuple(map(float, p)) for p in pts]

    if src_rect is not None and len(pts_list) >= 2 and inside(pts_list[0], src_rect):
        first_out = next((i for i in range(len(pts_list)) if not inside(pts_list[i], src_rect)), None)
        if first_out is not None and first_out > 0:
            cross = segment_rect_exit(pts_list[first_out - 1], pts_list[first_out], src_rect)
            pts_list = [cross] + pts_list[first_out:]

    if dst_rect is not None and len(pts_list) >= 2 and inside(pts_list[-1], dst_rect):
        # Walk from the end.
        last_out = None
        for i in range(len(pts_list) - 1, -1, -1):
            if not inside(pts_list[i], dst_rect):
                last_out = i
                break
        if last_out is not None and last_out < len(pts_list) - 1:
            cross = segment_rect_exit(pts_list[last_out + 1], pts_list[last_out], dst_rect)
            pts_list = pts_list[: last_out + 1] + [cross]

    return np.array(pts_list, dtype=np.float32)


def _catmull_rom_to_lines(spline: list[tuple[float, float]], samples_per_seg: int = 16):
    """Convert a dot B-spline control polygon to a dense polyline.

    `dot -Tplain` emits edge splines as sequences of Bezier control points —
    the first point is the start, then groups of 3 define each cubic segment.
    """
    if len(spline) < 4:
        return np.array(spline, dtype=np.float32)

    pts: list[tuple[float, float]] = [spline[0]]
    i = 0
    while i + 3 < len(spline):
        p0 = np.array(spline[i])
        p1 = np.array(spline[i + 1])
        p2 = np.array(spline[i + 2])
        p3 = np.array(spline[i + 3])
        for s in range(1, samples_per_seg + 1):
            t = s / samples_per_seg
            b = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            pts.append((float(b[0]), float(b[1])))
        i += 3
    return np.array(pts, dtype=np.float32)


def render(layout: Layout, nodes_meta: dict[str, Node], png_path: Path | None = None) -> None:
    import vispy

    vispy.use(app="glfw")
    from vispy import scene
    from vispy.scene import visuals

    canvas = scene.SceneCanvas(
        title="TRACE pipeline map",
        keys="interactive",
        size=(2400, 1800),
        bgcolor="white",
        show=png_path is None,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)

    # vispy's Text visual uses pixel-based font_size that does NOT scale with
    # the camera. We collect every Text visual we create here along with its
    # "base" font size (the size that looks right at the default frame-all
    # view), then hook the camera's transform_change event to rescale font_size
    # proportionally to zoom. See the _rescale_text() callback at the end of
    # this function for the actual scaling logic.
    scaling_texts: list = []

    # Coordinate scaling: dot units are inches; convert to a pixelly scale.
    SCALE = 72.0

    # Cluster backgrounds ------------------------------------------------------
    by_group: dict[str, list[LaidOutNode]] = {}
    for ln in layout.nodes.values():
        g = nodes_meta[ln.id].group
        by_group.setdefault(g, []).append(ln)

    for group_id, members in by_group.items():
        if not GROUPS[group_id].get("cluster", True):
            continue  # non-cluster groups (e.g. "input") don't get a box
        xs = [n.x for n in members]
        ys = [n.y for n in members]
        ws = [n.w for n in members]
        hs = [n.h for n in members]
        x0 = (min(xs[i] - ws[i] / 2 for i in range(len(members))) - 0.20) * SCALE
        x1 = (max(xs[i] + ws[i] / 2 for i in range(len(members))) + 0.20) * SCALE
        # Just enough bottom padding to clear the lowest node; extra headroom
        # at the top for the cluster label to sit above the tallest node.
        y0 = (min(ys[i] - hs[i] / 2 for i in range(len(members))) - 0.22) * SCALE
        y1 = (max(ys[i] + hs[i] / 2 for i in range(len(members))) + 0.50) * SCALE

        stroke = _hex_to_rgba(GROUPS[group_id]["stroke"], 0.35)
        fill = _hex_to_rgba(GROUPS[group_id]["fill"], 0.25)
        visuals.Rectangle(
            center=((x0 + x1) / 2, (y0 + y1) / 2),
            width=(x1 - x0),
            height=(y1 - y0),
            color=fill,
            border_color=stroke,
            border_width=2,
            radius=10,
            parent=view.scene,
        )
        # Cluster label sits in the headroom at the top of the cluster box,
        # between the cluster top edge and the topmost contained node.
        # Same baseline trick as node titles: place the baseline so the
        # capital-letter top sits ``CLUSTER_PAD`` px below the cluster top.
        label_text = GROUPS[group_id].get("label")
        if label_text:
            CLUSTER_LABEL_FONT = 12.0
            CLUSTER_PAD = 8.0
            cluster_label_ascent = CLUSTER_LABEL_FONT * 0.80
            baseline_y = y1 - CLUSTER_PAD - cluster_label_ascent
            _t = visuals.Text(
                label_text,
                pos=((x0 + x1) / 2, baseline_y),
                color=_hex_to_rgba(GROUPS[group_id]["stroke"]),
                font_size=CLUSTER_LABEL_FONT,
                bold=True,
                anchor_x="center",
                anchor_y="baseline",
                parent=view.scene,
            )
            scaling_texts.append((_t, CLUSTER_LABEL_FONT))

    # Edges --------------------------------------------------------------------
    def _rect_for(node_id: str):
        n = layout.nodes.get(node_id)
        if n is None:
            return None
        return (n.x - n.w / 2, n.y - n.h / 2, n.x + n.w / 2, n.y + n.h / 2)

    # Click-to-highlight state — populated as we create edge visuals so the
    # mouse_release handler below can recolor the lines/arrows belonging to
    # the clicked node's outgoing edges.
    EDGE_BASE_LINE_COLOR = (0.30, 0.33, 0.40, 0.85)
    EDGE_BASE_LINE_WIDTH = 1.6
    EDGE_BASE_ARROW_COLOR = (0.30, 0.33, 0.40, 1.0)
    EDGE_HIGHLIGHT_COLOR = (0.95, 0.45, 0.10, 1.0)
    EDGE_HIGHLIGHT_WIDTH = 3.2
    edge_visuals_by_src: dict = {}  # src_id -> list of (line, arrow_or_None)
    all_edge_visuals: list = []

    for e in layout.edges:
        pts_raw = _catmull_rom_to_lines(e.spline)
        pts_raw = _clip_polyline_to_node_rects(pts_raw, _rect_for(e.src), _rect_for(e.dst))
        pts = pts_raw * SCALE
        line_vis = visuals.Line(
            pos=pts,
            color=EDGE_BASE_LINE_COLOR,
            width=EDGE_BASE_LINE_WIDTH,
            method="gl",
            parent=view.scene,
        )
        arrow_vis = None
        # Arrowhead at end
        if len(pts) >= 2:
            p_end = pts[-1]
            p_prev = pts[-2]
            d = p_end - p_prev
            n = np.linalg.norm(d)
            if n > 1e-6:
                d = d / n
                perp = np.array([-d[1], d[0]])
                size = 7.0
                tri = np.array(
                    [
                        p_end,
                        p_end - d * size + perp * size * 0.5,
                        p_end - d * size - perp * size * 0.5,
                    ],
                    dtype=np.float32,
                )
                arrow_vis = visuals.Polygon(
                    pos=tri,
                    color=EDGE_BASE_ARROW_COLOR,
                    border_color=EDGE_BASE_ARROW_COLOR,
                    parent=view.scene,
                )
        edge_visuals_by_src.setdefault(e.src, []).append((line_vis, arrow_vis))
        all_edge_visuals.append((line_vis, arrow_vis))
        if e.label and e.label_pos:
            EDGE_LABEL_FONT = 6.5
            _t = visuals.Text(
                e.label,
                pos=(e.label_pos[0] * SCALE, e.label_pos[1] * SCALE),
                color=(0.25, 0.25, 0.35, 1.0),
                font_size=EDGE_LABEL_FONT,
                italic=True,
                anchor_x="center",
                anchor_y="center",
                parent=view.scene,
            )
            scaling_texts.append((_t, EDGE_LABEL_FONT))

    # Nodes --------------------------------------------------------------------
    # vispy Text with anchor_y="baseline" places the glyph baseline exactly at
    # pos.y — that's the only anchor whose meaning is unambiguous, so we use
    # it for everything and compute baselines ourselves.
    #
    # Font metrics (approximate for the default OpenSans face):
    #   ascent  ≈ 0.80 * font_size   (top of capital letters above baseline)
    #   descent ≈ 0.20 * font_size   (descenders below baseline)
    #   line_height = 1.1 * font_size (vispy default for multi-line Text)
    # Reduced from 12/9 so the initial-framing render (whole diagram in
    # the window) doesn't overflow box borders. These sizes are the values
    # that look right at the frame-all zoom; _rescale_text() (defined after
    # the initial set_range() below) scales them up/down with the camera
    # so text stays proportional to the boxes as the user zooms.
    TITLE_FONT = 8.0
    SUB_FONT = 6.0
    ASCENT_TITLE = TITLE_FONT * 0.80
    LINE_H_TITLE = TITLE_FONT * 1.10
    LINE_H_SUB = SUB_FONT * 1.10
    ASCENT_SUB = SUB_FONT * 0.80
    PAD_TOP = 30.0  # distance from box top edge to title cap-top — shifts title and bullets down
    PAD_LEFT = 10.0  # left inset for substeps
    GAP_TITLE_SUB = 22.0  # vertical gap between title baseline-descent and first bullet ascent

    for ln in layout.nodes.values():
        meta = nodes_meta[ln.id]
        group = GROUPS[meta.group]
        fill = _hex_to_rgba(group["fill"])
        stroke = _hex_to_rgba(group["stroke"])
        cx = ln.x * SCALE
        cy = ln.y * SCALE
        w = ln.w * SCALE
        h = ln.h * SCALE

        visuals.Rectangle(
            center=(cx, cy),
            width=w,
            height=h,
            color=fill,
            border_color=stroke,
            border_width=1.8,
            radius=8 if meta.kind == "process" else 14,
            parent=view.scene,
        )

        top_y = cy + h / 2
        # Title baseline: top of box − padding − ascent, so the top of the
        # capital letters sits PAD_TOP below the box top edge.
        title_baseline = top_y - PAD_TOP - ASCENT_TITLE
        _t = visuals.Text(
            meta.title,
            pos=(cx, title_baseline),
            color=_hex_to_rgba(group["stroke"]),
            font_size=TITLE_FONT,
            bold=True,
            anchor_x="center",
            anchor_y="baseline",
            parent=view.scene,
        )
        scaling_texts.append((_t, TITLE_FONT))
        if meta.substeps:
            # First substep baseline sits below the title by (title descent
            # + gap + substep ascent).
            sub_first_baseline = (
                title_baseline - (TITLE_FONT - ASCENT_TITLE) - GAP_TITLE_SUB - ASCENT_SUB  # title descent
            )
            sub_text = "\n".join(f"• {s}" for s in meta.substeps)
            _t = visuals.Text(
                sub_text,
                pos=(cx - w / 2 + PAD_LEFT, sub_first_baseline),
                color=(0.18, 0.20, 0.25, 1.0),
                font_size=SUB_FONT,
                anchor_x="left",
                anchor_y="baseline",
                parent=view.scene,
            )
            scaling_texts.append((_t, SUB_FONT))

    # Frame the view to fit everything ----------------------------------------
    all_x: list[float] = []
    all_y: list[float] = []
    for ln in layout.nodes.values():
        all_x += [(ln.x - ln.w / 2 - 0.5) * SCALE, (ln.x + ln.w / 2 + 0.5) * SCALE]
        all_y += [(ln.y - ln.h / 2 - 0.5) * SCALE, (ln.y + ln.h / 2 + 0.9) * SCALE]
    view.camera.set_range(x=(min(all_x), max(all_x)), y=(min(all_y), max(all_y)), margin=0.05)

    # Snapshot the framing-zoom rect width as the "1.0×" reference. Every
    # font_size in scaling_texts is the value picked to look right at this
    # zoom — see the comment near TITLE_FONT/SUB_FONT for the framing-fit
    # rationale. _rescale_text() below scales each visual's font_size by
    # base_rect_width / current_rect_width so text grows linearly as the
    # camera zooms in.
    base_rect_width = float(view.camera.rect.width) or 1.0

    def _rescale_text():
        cur_w = float(view.camera.rect.width)
        if cur_w <= 0:
            return
        ratio = base_rect_width / cur_w
        for tv, base_size in scaling_texts:
            new_size = max(1.0, base_size * ratio)
            if abs(float(tv.font_size) - new_size) > 0.05:
                tv.font_size = new_size

    # PanZoomCamera doesn't fire an event when its rect changes — it just
    # calls self.view_changed() (a regular method, not an EventEmitter).
    # Monkey-patch it on this instance to piggyback the rescale call.
    _orig_view_changed = view.camera.view_changed

    def _view_changed_with_rescale():
        _orig_view_changed()
        _rescale_text()

    view.camera.view_changed = _view_changed_with_rescale
    _rescale_text()

    if png_path is not None:
        from vispy.io import write_png

        canvas.render()  # force a frame
        img = canvas.render(alpha=True)
        write_png(str(png_path), img)
        print(f"Wrote {png_path}")
        canvas.close()
        return

    # Interactive help
    print(
        "Pipeline map — drag to pan, scroll to zoom, press 'r' to reset view,\n"
        "click a box to highlight its outgoing arrows (click again or off-box to clear)."
    )

    @canvas.events.key_press.connect
    def _on_key(event):
        if event.key and event.key.name.lower() == "r":
            view.camera.set_range(x=(min(all_x), max(all_x)), y=(min(all_y), max(all_y)), margin=0.05)

    # Click-to-highlight: when the user clicks a node box, recolor the
    # outgoing edges. Distinguishes click from pan by tracking the press
    # position and only firing when release is within a few pixels.
    node_rects_scene: dict = {
        ln.id: (
            (ln.x - ln.w / 2) * SCALE,
            (ln.y - ln.h / 2) * SCALE,
            (ln.x + ln.w / 2) * SCALE,
            (ln.y + ln.h / 2) * SCALE,
        )
        for ln in layout.nodes.values()
    }
    highlight_state: dict = {"node": None, "press_pos": None}

    def _reset_highlight():
        for line, arrow in all_edge_visuals:
            line.set_data(color=EDGE_BASE_LINE_COLOR, width=EDGE_BASE_LINE_WIDTH)
            if arrow is not None:
                arrow.color = EDGE_BASE_ARROW_COLOR
                arrow.border_color = EDGE_BASE_ARROW_COLOR
        canvas.update()

    def _apply_highlight(node_id: str):
        _reset_highlight()
        for line, arrow in edge_visuals_by_src.get(node_id, []):
            line.set_data(color=EDGE_HIGHLIGHT_COLOR, width=EDGE_HIGHLIGHT_WIDTH)
            if arrow is not None:
                arrow.color = EDGE_HIGHLIGHT_COLOR
                arrow.border_color = EDGE_HIGHLIGHT_COLOR
        highlight_state["node"] = node_id
        canvas.update()

    @canvas.events.mouse_press.connect
    def _on_press(event):
        if event.button == 1:
            highlight_state["press_pos"] = tuple(event.pos)

    @canvas.events.mouse_release.connect
    def _on_release(event):
        if event.button != 1 or highlight_state["press_pos"] is None:
            return
        px, py = highlight_state["press_pos"]
        rx, ry = event.pos
        highlight_state["press_pos"] = None
        # Distinguish click from drag/pan.
        if abs(rx - px) > 4 or abs(ry - py) > 4:
            return
        scene_pos = view.scene.transform.imap(event.pos)
        sx, sy = float(scene_pos[0]), float(scene_pos[1])
        hit = None
        for nid, (x0, y0, x1, y1) in node_rects_scene.items():
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                hit = nid
                break
        if hit is None or hit == highlight_state["node"]:
            if highlight_state["node"] is not None:
                _reset_highlight()
                highlight_state["node"] = None
        else:
            _apply_highlight(hit)

    from vispy import app as vispy_app

    vispy_app.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_CACHED_LAYOUT_PATH = Path(__file__).resolve().parent / "pipeline_layout.plain"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=None, help="Render to PNG and exit.")
    parser.add_argument("--dot", type=Path, default=None, help="Write DOT source to file.")
    parser.add_argument(
        "--regenerate-layout",
        action="store_true",
        help=(
            "Invoke graphviz `dot -Tplain` to recompute the cached layout and "
            "save to pipeline_layout.plain. Run this after NODES/EDGES changes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the layout and print a summary, but skip vispy rendering.",
    )
    args = parser.parse_args(argv)

    # Layout is read from a static cached file by default — no graphviz at
    # runtime. Use --regenerate-layout (requires `dot`) when NODES/EDGES change.
    if args.regenerate_layout or not _CACHED_LAYOUT_PATH.exists():
        dot_src = build_dot()
        if args.dot:
            args.dot.write_text(dot_src)
            print(f"Wrote {args.dot}")
        try:
            plain = run_dot_plain(dot_src)
        except FileNotFoundError:
            print("Error: 'dot' not found. Install graphviz (brew install graphviz).", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as e:
            print(f"dot failed: {e.stderr}", file=sys.stderr)
            return 1
        _CACHED_LAYOUT_PATH.write_text(plain)
        print(f"Wrote {_CACHED_LAYOUT_PATH}")
    else:
        plain = _CACHED_LAYOUT_PATH.read_text()
        if args.dot:
            args.dot.write_text(build_dot())
            print(f"Wrote {args.dot}")

    layout = parse_plain(plain)
    # Spread bundles of parallel near-horizontal segments BEFORE rendering, so
    # 3-way pile-ups like the P2 → {WING_GJ, P3, OUT_ISO} fan all get their
    # own visible track.
    _spread_parallel_horizontal_segments(layout)
    # Rescue any edge labels that graphviz dumped at its (0, 0.11) fallback
    # position — they get reseated to their edge's midpoint before the next
    # passes nudge them into a clear spot.
    _reseat_fallback_edge_labels(layout)
    # Lift edge labels off horizontal arrow lines so the italic text sits
    # cleanly above the arrow instead of being bisected by it.
    _lift_labels_off_horizontal_segments(layout)
    # Move any labels that landed inside a node rectangle OR on top of
    # another label (e.g. ``anchored`` on MM, ``wing_outline`` under I3,
    # ``wing_axis`` overlapping ``split polys + veins``) into the nearest
    # empty gap. Iterated since moving one label can free space for another.
    _displace_labels_off_obstacles(layout)
    # Then push any UNRELATED horizontal arrows up so they don't cross a
    # label they don't belong to. Runs after the label displacement so we
    # know each label's final position.
    _lift_horizontals_above_crossed_labels(layout)
    # Final cleanup: a lift can push two segments to almost identical y
    # (both got lifted above the same cluster label). One more tight-bundle
    # spread separates them; the follow-up lift catches anyone the spread
    # bumped back into a label band.
    _spread_parallel_horizontal_segments(layout, y_proximity_max=0.10, min_separation=0.40)
    _lift_horizontals_above_crossed_labels(layout)
    # Finally, snap each spline's endpoints onto its src/dst rect so arrow
    # tips actually touch the boxes. Preserves the direction of the adjacent
    # Bezier segment, so no diagonals are introduced.
    _snap_endpoints_to_node_rects(layout)
    # When the snapped endpoint is on the rect's side but the perpendicular
    # axis is still outside (e.g. horizontal line meets P5's left edge but
    # below P5's bottom), append a 90° bridge segment so the line turns
    # cleanly into the box.
    _add_bridge_turns_into_node_rects(layout)
    nodes_meta = {n.id: n for n in NODES}

    missing = set(layout.nodes) - set(nodes_meta)
    if missing:
        print(f"Warning: layout has unknown nodes: {sorted(missing)}", file=sys.stderr)

    if args.dry_run:
        print(f"Graph bbox: {layout.width:.2f} x {layout.height:.2f} inches")
        print(f"Nodes: {len(layout.nodes)}")
        print(f"Edges: {len(layout.edges)}")
        by_group: dict[str, int] = {}
        for nid in layout.nodes:
            g = nodes_meta[nid].group
            by_group[g] = by_group.get(g, 0) + 1
        for g, c in by_group.items():
            print(f"  {g}: {c} nodes")
        tall = sorted(layout.nodes.values(), key=lambda n: n.h, reverse=True)[:3]
        print("Tallest nodes (sanity check for label escape fix):")
        for n in tall:
            print(f"  {n.id}: {n.w:.2f} x {n.h:.2f} in")
        return 0

    render(layout, nodes_meta, png_path=args.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
