"""Side-by-side comparison of Voronoi vs skeletonization centerlines on test wings."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from WingVeinAnalyzer.models.geojson_parser import parse_geojson
from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon, extract_veins_from_mask

# Distinct colors for Voronoi segments (BGR for cv2, but we convert to RGB for matplotlib)
VORONOI_COLORS = [
    (255, 0, 0),  # red
    (0, 200, 0),  # green
    (0, 100, 255),  # orange
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 128, 0),  # blue-ish
    (128, 0, 255),  # purple
    (0, 255, 128),  # spring green
    (255, 255, 0),  # cyan
    (128, 255, 0),  # lime
]


def discover_test_wings(base_dir: Path | None = None, pattern: str = "testwing*") -> list[tuple[str, Path, Path]]:
    """Find all test wings with matching GeoJSON and TIFF files."""
    base = base_dir or (PROJECT_ROOT / "test_data")
    wings = []
    for wing_dir in sorted(base.glob(pattern)):
        if not wing_dir.is_dir():
            continue
        wing_name = wing_dir.name
        # Match primary annotation geojson (not _expected, _landmarks, etc.)
        geojson_files = [
            f
            for f in wing_dir.glob("*.geojson")
            if not any(s in f.stem for s in ("_expected", "_landmarks")) and f.stem == wing_name
        ]
        if not geojson_files:
            continue
        geojson_path = geojson_files[0]
        # Prefer non-chopped TIF
        tif_files = list(wing_dir.glob("*.tif"))
        tif_files = [t for t in tif_files if "_chopped" not in t.stem] or tif_files
        if not tif_files:
            print(f"  Skipping {wing_name}: no TIFF found")
            continue
        wings.append((wing_name, geojson_path, tif_files[0]))
    return wings


def rasterize_vein_mask(vein_polygons: list, image_shape: tuple[int, int], closing_kernel_size: int = 11) -> np.ndarray:
    """Rasterize vein polygons into binary mask with morphological closing."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(mask, poly, 1)
    if closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def measure_branches(skeleton: np.ndarray) -> list[int]:
    """Measure all terminal branch lengths in a skeleton (endpoint to nearest junction)."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)

    endpoints = (skeleton > 0) & (neighbor_count == 1)
    ep_ys, ep_xs = np.where(endpoints)

    branch_lengths = []
    for ey, ex in zip(ep_ys, ep_xs):
        cy, cx = ey, ex
        visited = set()
        length = 0

        while True:
            visited.add((cy, cx))
            length += 1

            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < skeleton.shape[0]
                        and 0 <= nx < skeleton.shape[1]
                        and skeleton[ny, nx] > 0
                        and (ny, nx) not in visited
                    ):
                        neighbors.append((ny, nx))

            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                cy, cx = neighbors[0]
            else:
                break

        branch_lengths.append(length)

    return branch_lengths


def prune_skeleton(skeleton: np.ndarray, min_branch_length: int = 100) -> np.ndarray:
    """Remove terminal branches shorter than min_branch_length (single-pass)."""
    pruned = skeleton.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(pruned.astype(np.uint8), -1, kernel)

    endpoints = (pruned > 0) & (neighbor_count == 1)
    ep_ys, ep_xs = np.where(endpoints)

    to_remove = []
    for ey, ex in zip(ep_ys, ep_xs):
        branch_pixels = []
        cy, cx = ey, ex
        visited = set()

        while True:
            branch_pixels.append((cy, cx))
            visited.add((cy, cx))

            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < pruned.shape[0]
                        and 0 <= nx < pruned.shape[1]
                        and pruned[ny, nx] > 0
                        and (ny, nx) not in visited
                    ):
                        neighbors.append((ny, nx))

            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                cy, cx = neighbors[0]
            else:
                break

        if len(branch_pixels) < min_branch_length:
            to_remove.extend(branch_pixels)

    for py, px in to_remove:
        pruned[py, px] = 0

    return pruned


def apply_vein_mask_tint(
    image: np.ndarray, vein_mask: np.ndarray, color: tuple = (255, 200, 200), alpha: float = 0.25
) -> np.ndarray:
    """Apply a semi-transparent tint where vein mask is active."""
    overlay = image.copy()
    mask_bool = vein_mask > 0
    tint = np.array(color, dtype=np.float32)
    overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool].astype(np.float32) + alpha * tint).astype(np.uint8)
    return overlay


def draw_voronoi_centerlines(image: np.ndarray, voronoi_result, vein_mask: np.ndarray | None = None) -> np.ndarray:
    """Draw Voronoi centerlines on image copy, colored by segment."""
    overlay = image.copy()
    if vein_mask is not None:
        overlay = apply_vein_mask_tint(overlay, vein_mask)
    for i, (key, line) in enumerate(voronoi_result.centerlines.items()):
        color = VORONOI_COLORS[i % len(VORONOI_COLORS)]
        coords = np.array(line.coords, dtype=np.int32)
        if len(coords) >= 2:
            cv2.polylines(overlay, [coords], isClosed=False, color=color, thickness=4)
    return overlay


def draw_skeleton_pixels(
    image: np.ndarray,
    skeleton: np.ndarray,
    vein_mask: np.ndarray | None = None,
    color: tuple = (0, 220, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw skeleton pixels on image copy, dilated for visibility."""
    overlay = image.copy()
    if vein_mask is not None:
        overlay = apply_vein_mask_tint(overlay, vein_mask)
    # Dilate skeleton for visibility
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        drawn = cv2.dilate(skeleton.astype(np.uint8), kernel) > 0
    else:
        drawn = skeleton > 0
    overlay[drawn] = color
    return overlay


def process_wing(wing_name: str, geojson_path: Path, tif_path: Path) -> list[int]:
    """Generate comparison image for a single test wing. Returns branch lengths."""
    print(f"\nProcessing {wing_name}...")

    # Load image
    image = cv2.imread(str(tif_path))
    if image is None:
        print(f"  ERROR: Could not load {tif_path}")
        return
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  Image: {w}x{h}")

    # Parse GeoJSON
    annotations = parse_geojson(geojson_path)
    print(f"  Intervein polygons: {len(annotations.intervein_polygons)}")
    print(f"  Vein polygons: {len(annotations.vein_polygons)}")

    if not annotations.vein_polygons:
        print(f"  Skipping {wing_name}: no vein polygons found")
        return []

    # --- Rasterize vein mask (shared by both methods) ---
    vein_mask = rasterize_vein_mask(annotations.vein_polygons, (h, w))
    mask_pixels = int(np.sum(vein_mask > 0))
    raw_skeleton = skeletonize(vein_mask > 0)
    raw_skel_pixels = int(np.sum(raw_skeleton))
    print(f"  Vein mask: {mask_pixels:,} pixels ({100*mask_pixels/(h*w):.1f}% of image)")
    print(f"  Raw skeleton pixels: {raw_skel_pixels:,}")

    # --- Voronoi centerlines ---
    voronoi_result = extract_veins_from_mask(
        annotations.vein_polygons,
        (h, w),
        intervein_polygons=annotations.intervein_polygons,
    )
    voronoi_overlay = draw_voronoi_centerlines(image_rgb.copy(), voronoi_result, vein_mask)
    voronoi_count = len(voronoi_result.centerlines)
    voronoi_total_length = sum(line.length for line in voronoi_result.centerlines.values())

    # --- Measure branch lengths ---
    branch_lengths = measure_branches(raw_skeleton.astype(np.uint8))
    sorted_lengths = sorted(branch_lengths)
    print(
        f"  Branches: {len(branch_lengths)} total, "
        f"min={min(branch_lengths)}px, max={max(branch_lengths)}px, "
        f"median={sorted_lengths[len(sorted_lengths)//2]}px"
    )

    # --- Skeletonization centerlines ---
    skeleton_pruned = prune_skeleton(raw_skeleton.astype(np.uint8), min_branch_length=200)
    pruned_skel_pixels = int(np.sum(skeleton_pruned > 0))
    skeleton_overlay = draw_skeleton_pixels(
        image_rgb.copy(),
        skeleton_pruned,
        vein_mask,
        color=(0, 255, 255),
        thickness=4,
    )

    # --- Stats ---
    print(f"  Voronoi:  {voronoi_count} segments, {voronoi_total_length:.0f} px total length")
    print(f"  Skeleton: {pruned_skel_pixels:,} px ({raw_skel_pixels - pruned_skel_pixels} spurs pruned)")

    # --- Render side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(32, 14))
    fig.suptitle(f"{wing_name} — Centerline Comparison", fontsize=16, fontweight="bold")

    axes[0].imshow(voronoi_overlay)
    axes[0].set_title(f"Voronoi ({voronoi_count} segments, {voronoi_total_length:.0f} px)")
    axes[0].axis("off")

    axes[1].imshow(skeleton_overlay)
    axes[1].set_title(
        f"Skeletonization ({pruned_skel_pixels:,} px, {raw_skel_pixels - pruned_skel_pixels} spurs pruned)"
    )
    axes[1].axis("off")

    plt.tight_layout()

    out_path = geojson_path.parent / "centerline_comparison.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return branch_lengths


def main():
    # Accept optional directory argument
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1]).resolve()
        wings = discover_test_wings(base_dir, pattern="*")
    else:
        wings = discover_test_wings()
    if not wings:
        print("No test wings found!")
        sys.exit(1)

    print(f"Found {len(wings)} test wings")
    all_branches: dict[str, list[int]] = {}
    for wing_name, geojson_path, tif_path in wings:
        try:
            lengths = process_wing(wing_name, geojson_path, tif_path)
            if lengths:
                all_branches[wing_name] = lengths
        except Exception as e:
            print(f"  ERROR processing {wing_name}: {e}")

    # --- Branch length frequency histogram ---
    if all_branches:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle("Skeleton Branch Length Distribution (all test wings)", fontsize=14, fontweight="bold")

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_branches)))

        # Top: per-wing overlaid histograms
        ax = axes[0]
        max_len = max(max(v) for v in all_branches.values())
        bins = np.arange(0, min(max_len + 20, 2000), 10)
        for (name, lengths), color in zip(all_branches.items(), colors):
            ax.hist(lengths, bins=bins, alpha=0.5, label=f"{name} (n={len(lengths)})", color=color, edgecolor="none")
        ax.set_xlabel("Branch length (px)")
        ax.set_ylabel("Count")
        ax.set_title("Per-wing branch length frequency")
        ax.legend()
        ax.axvline(x=150, color="red", linestyle="--", linewidth=1.5, label="prune threshold (200px)")
        ax.legend()

        # Bottom: combined cumulative distribution
        ax2 = axes[1]
        combined = []
        for lengths in all_branches.values():
            combined.extend(lengths)
        combined_sorted = np.sort(combined)
        cumulative = np.arange(1, len(combined_sorted) + 1) / len(combined_sorted) * 100
        ax2.plot(combined_sorted, cumulative, color="steelblue", linewidth=2)
        ax2.axvline(x=200, color="red", linestyle="--", linewidth=1.5)
        ax2.axhline(
            y=100 * np.searchsorted(combined_sorted, 200) / len(combined_sorted),
            color="red",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )
        pct_below_200 = 100 * np.searchsorted(combined_sorted, 200) / len(combined_sorted)
        ax2.annotate(
            f"{pct_below_200:.1f}% below 200px",
            xy=(200, pct_below_200),
            xytext=(350, pct_below_200 - 10),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
            fontsize=11,
        )
        ax2.set_xlabel("Branch length (px)")
        ax2.set_ylabel("Cumulative %")
        ax2.set_title("Combined cumulative distribution (all wings)")
        ax2.set_xlim(0, min(max_len + 20, 2000))
        ax2.set_ylim(0, 105)

        plt.tight_layout()
        hist_base = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PROJECT_ROOT / "test_data"
        out_path = hist_base / "branch_length_histogram.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nBranch length histogram saved: {out_path}")

    print("\nDone! Check test_data/testwing*/centerline_comparison.png")


if __name__ == "__main__":
    main()
