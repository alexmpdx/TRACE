"""Interactive pipeline map for the TRACE + identifyFeatures workflow.

Graphviz computes the layout; vispy renders it with pan/zoom. Run with the
venv at ``.venv-pipeline-map/bin/python pipeline_map.py``.

Flags:
    --png PATH   Render a static PNG to PATH and exit (no window).
    --dot PATH   Write the generated DOT source to PATH (for debugging).

Before editing this file — especially before adding nodes, edges, or
adjusting the layout — read `PIPELINE_MAP_GUIDE.md` (same directory). It
records the layout constraints, techniques (rank pairs, ortho routing,
side-reference `constraint=false`, …), and what's been tried and
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
    "input": {"label": "External inputs", "fill": "#e8eef7", "stroke": "#4a6fa5", "cluster": True},
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
    # --- External inputs ---
    # Ordered so each input sits above its primary consumer column:
    # IMG → P1, WING_MODEL → P2, LMK_MODEL → P3, SEG_MODEL → P5, CONFIG → ifeat (right).
    Node("IMG", "Wing image", "input", ["*.tif / *.bmp / *.png / *.jpg / *.psd"], "data"),
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
            "Apply same affine to every produced GeoJSON",
            "Optional mirror-correct for opposite-chirality wings",
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
        "Step 5: Trace + assign + split",
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
        "Step 6: Name intervein regions",
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
    Node("OUT_LMK_OV", "landmarks overlay", "output", ["{stem}_landmarks_overlay.png"], "data"),
    Node("OUT_SEG_OV", "segmentation overlay", "output", ["{stem}_segmentation_overlay.png"], "data"),
    Node("OUT_AP_OV", "AP compartment overlay", "output", ["{stem}_ap_overlay.png"], "data"),
    Node("OUT_CV_OV", "CV ratio overlay", "output", ["{stem}_cv_ratio_overlay.png"], "data"),
    Node("OUT_CSV", "batch CSV", "output", ["measurements.csv (wide, wing-level)"], "data"),
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
    # Landmarks overlay derives from LMK_GJ. constraint=false so dot routes
    # this as a short side reference instead of stretching the artifact row
    # down toward outputs.
    ("LMK_GJ", "OUT_LMK_OV", None, {"constraint": "false"}),
    # Segmentation overlay derives from the raw SEG_GJ — same treatment.
    ("SEG_GJ", "OUT_SEG_OV", None, {"constraint": "false"}),
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
        ["OUT_AP_OV", "OUT_CV_OV", "OUT_CSV"],
    ],
    "input": [["IMG", "CONFIG"], ["LMK_MODEL", "SEG_MODEL", "WING_MODEL"]],
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
        '  node  [fontname="Helvetica", fontsize=13, shape=box, ' 'style="rounded,filled", margin="0.30,0.18"];',
        '  edge  [fontname="Helvetica", fontsize=10];',
    ]

    by_group: dict[str, list[Node]] = {g: [] for g in GROUPS}
    for n in NODES:
        by_group.setdefault(n.group, []).append(n)

    def _emit_node(n: Node, meta: dict, indent: str) -> str:
        label_lines = [n.title] + ([""] + [f"• {s}" for s in n.substeps] if n.substeps else [])
        label = "\\l".join(label_lines) + "\\l"
        shape = "ellipse" if n.kind == "data" else "box"
        return (
            f'{indent}{n.id} [label="{_escape_label(label)}", '
            f'shape={shape}, fillcolor="{meta["fill"]}", color="{meta["stroke"]}"];'
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
            CLUSTER_LABEL_FONT = 14.0
            CLUSTER_PAD = 8.0
            cluster_label_ascent = CLUSTER_LABEL_FONT * 0.80
            baseline_y = y1 - CLUSTER_PAD - cluster_label_ascent
            visuals.Text(
                label_text,
                pos=((x0 + x1) / 2, baseline_y),
                color=_hex_to_rgba(GROUPS[group_id]["stroke"]),
                font_size=CLUSTER_LABEL_FONT,
                bold=True,
                anchor_x="center",
                anchor_y="baseline",
                parent=view.scene,
            )

    # Edges --------------------------------------------------------------------
    for e in layout.edges:
        pts = _catmull_rom_to_lines(e.spline) * SCALE
        visuals.Line(
            pos=pts,
            color=(0.30, 0.33, 0.40, 0.85),
            width=1.6,
            method="gl",
            parent=view.scene,
        )
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
                visuals.Polygon(
                    pos=tri,
                    color=(0.30, 0.33, 0.40, 1.0),
                    border_color=(0.30, 0.33, 0.40, 1.0),
                    parent=view.scene,
                )
        if e.label and e.label_pos:
            visuals.Text(
                e.label,
                pos=(e.label_pos[0] * SCALE, e.label_pos[1] * SCALE),
                color=(0.25, 0.25, 0.35, 1.0),
                font_size=8,
                italic=True,
                anchor_x="center",
                anchor_y="center",
                parent=view.scene,
            )

    # Nodes --------------------------------------------------------------------
    # vispy Text with anchor_y="baseline" places the glyph baseline exactly at
    # pos.y — that's the only anchor whose meaning is unambiguous, so we use
    # it for everything and compute baselines ourselves.
    #
    # Font metrics (approximate for the default OpenSans face):
    #   ascent  ≈ 0.80 * font_size   (top of capital letters above baseline)
    #   descent ≈ 0.20 * font_size   (descenders below baseline)
    #   line_height = 1.1 * font_size (vispy default for multi-line Text)
    TITLE_FONT = 12.0
    SUB_FONT = 9.0
    ASCENT_TITLE = TITLE_FONT * 0.80
    LINE_H_TITLE = TITLE_FONT * 1.10
    LINE_H_SUB = SUB_FONT * 1.10
    ASCENT_SUB = SUB_FONT * 0.80
    PAD_TOP = 6.0  # distance from box top edge to title baseline-minus-ascent
    PAD_LEFT = 10.0  # left inset for substeps
    GAP_TITLE_SUB = 6.0

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
        visuals.Text(
            meta.title,
            pos=(cx, title_baseline),
            color=_hex_to_rgba(group["stroke"]),
            font_size=TITLE_FONT,
            bold=True,
            anchor_x="center",
            anchor_y="baseline",
            parent=view.scene,
        )
        if meta.substeps:
            # First substep baseline sits below the title by (title descent
            # + gap + substep ascent).
            sub_first_baseline = (
                title_baseline - (TITLE_FONT - ASCENT_TITLE) - GAP_TITLE_SUB - ASCENT_SUB  # title descent
            )
            sub_text = "\n".join(f"• {s}" for s in meta.substeps)
            visuals.Text(
                sub_text,
                pos=(cx - w / 2 + PAD_LEFT, sub_first_baseline),
                color=(0.18, 0.20, 0.25, 1.0),
                font_size=SUB_FONT,
                anchor_x="left",
                anchor_y="baseline",
                parent=view.scene,
            )

    # Frame the view to fit everything ----------------------------------------
    all_x: list[float] = []
    all_y: list[float] = []
    for ln in layout.nodes.values():
        all_x += [(ln.x - ln.w / 2 - 0.5) * SCALE, (ln.x + ln.w / 2 + 0.5) * SCALE]
        all_y += [(ln.y - ln.h / 2 - 0.5) * SCALE, (ln.y + ln.h / 2 + 0.9) * SCALE]
    view.camera.set_range(x=(min(all_x), max(all_x)), y=(min(all_y), max(all_y)), margin=0.05)

    if png_path is not None:
        from vispy.io import write_png

        canvas.render()  # force a frame
        img = canvas.render(alpha=True)
        write_png(str(png_path), img)
        print(f"Wrote {png_path}")
        canvas.close()
        return

    # Interactive help
    print("Pipeline map — drag to pan, scroll to zoom, press 'r' to reset view.")

    @canvas.events.key_press.connect
    def _on_key(event):
        if event.key and event.key.name.lower() == "r":
            view.camera.set_range(x=(min(all_x), max(all_x)), y=(min(all_y), max(all_y)), margin=0.05)

    from vispy import app as vispy_app

    vispy_app.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=None, help="Render to PNG and exit.")
    parser.add_argument("--dot", type=Path, default=None, help="Write DOT source to file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the layout and print a summary, but skip vispy rendering.",
    )
    args = parser.parse_args(argv)

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

    layout = parse_plain(plain)
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
