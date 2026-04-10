"""IoU evaluation: compare pipeline output against GT_naming ground truth."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union

BASE = Path(__file__).parent
DEFAULT_GT_DIR = BASE / "GT_naming"
DEFAULT_OUT_DIR = BASE / "viz_output"
DEFAULT_THRESHOLD = 0.5

# Name mapping: GT name -> canonical name (output convention)
NAME_MAP = {
    "costal": "costa",
}

CANONICAL_VEINS = {
    "costa",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "ACV",
    "PCV",
    "Rs",
}
CANONICAL_REGIONS = {
    "marginal",
    "submarginal",
    "1st basal",
    "1st posterior",
    "discal",
    "2nd posterior",
    "3rd posterior",
}


def normalize_name(name: str) -> str:
    """Map GT names to canonical output names."""
    return NAME_MAP.get(name, name)


def categorize(name: str) -> str:
    """Return 'vein', 'region', or 'ectopic'."""
    if name in CANONICAL_VEINS:
        return "vein"
    if name in CANONICAL_REGIONS:
        return "region"
    if name == "ectopic" or name.startswith("EV"):
        return "ectopic"
    return "other"


def load_features(path: Path, aggregate_ev: bool = False) -> dict[str, object]:
    """Load GeoJSON features into a dict of name -> geometry.

    Normalizes names, unions duplicate-named features, and optionally
    aggregates EV* polygons under 'ectopic'.
    """
    with open(path) as f:
        data = json.load(f)

    polys: dict[str, list] = {}
    for feat in data.get("features", []):
        name = feat["properties"]["classification"]["name"]
        name = normalize_name(name)

        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.area == 0:
            continue

        # Handle merged names like "discal + 2nd posterior"
        parts = [p.strip() for p in name.split(" + ")] if " + " in name else [name]
        for part in parts:
            key = part
            if aggregate_ev and key.startswith("EV"):
                key = "ectopic"
            polys.setdefault(key, []).append(geom)

    # Union same-named polygons
    result = {}
    for name, geom_list in polys.items():
        merged = unary_union(geom_list)
        if not merged.is_valid:
            merged = merged.buffer(0)
        if not merged.is_empty and merged.area > 0:
            result[name] = merged
    return result


def compute_iou(poly_a, poly_b) -> float:
    """Compute IoU between two geometries."""
    if poly_a is None or poly_b is None:
        return 0.0
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    try:
        intersection = poly_a.intersection(poly_b)
        union = poly_a.union(poly_b)
        if union.area == 0:
            return 0.0
        return intersection.area / union.area
    except Exception:
        return 0.0


def extract_stem(gt_filename: str) -> str | None:
    """Extract specimen stem from a GT filename."""
    for suffix in [".tif .geojson", ".tif.geojson", ".bmp.geojson"]:
        if gt_filename.endswith(suffix):
            return gt_filename[: -len(suffix)]
    return None


def find_matched_pairs(gt_dir: Path, out_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find (stem, gt_path, out_path) triples for evaluable specimens."""
    pairs = []
    for gt_path in sorted(gt_dir.glob("*.geojson")):
        stem = extract_stem(gt_path.name)
        if stem is None:
            continue
        # Check for empty GT
        with open(gt_path) as f:
            data = json.load(f)
        if not data.get("features"):
            continue
        out_path = out_dir / f"{stem}_output.geojson"
        if not out_path.exists():
            continue
        pairs.append((stem, gt_path, out_path))
    return pairs


def evaluate_specimen(stem: str, gt_path: Path, out_path: Path) -> list[dict]:
    """Compute per-feature IoU for one specimen."""
    gt_feats = load_features(gt_path, aggregate_ev=False)
    out_feats = load_features(out_path, aggregate_ev=True)

    # Also aggregate GT ectopic
    if "ectopic" in gt_feats:
        pass  # already normalized
    # Union any remaining GT ectopic entries (handled by load_features)

    all_names = set(gt_feats.keys()) | set(out_feats.keys())
    rows = []
    for name in sorted(all_names):
        cat = categorize(name)
        if cat == "other":
            continue
        gt_poly = gt_feats.get(name)
        out_poly = out_feats.get(name)
        iou = compute_iou(gt_poly, out_poly)
        rows.append(
            {
                "stem": stem,
                "feature": name,
                "category": cat,
                "iou": iou,
                "gt_area": gt_poly.area if gt_poly else 0.0,
                "out_area": out_poly.area if out_poly else 0.0,
                "in_gt": gt_poly is not None,
                "in_out": out_poly is not None,
            }
        )
    return rows


def print_report(df: pd.DataFrame, threshold: float) -> None:
    """Print evaluation report."""
    # --- Per-specimen summary ---
    print(f"\n{'=' * 70}")
    print(f"Per-specimen summary (IoU threshold = {threshold:.2f})")
    print(f"{'=' * 70}")
    print(f"{'Specimen':<45s} {'Veins':>8s} {'Regions':>8s} {'Match':>7s}")
    print("-" * 70)

    for stem, group in df.groupby("stem", sort=False):
        veins = group[group["category"] == "vein"]
        regions = group[group["category"] == "region"]
        both_present = group[group["in_gt"] & group["in_out"]]
        matched = both_present[both_present["iou"] >= threshold]
        total = len(both_present)

        vein_iou = veins["iou"].mean() if len(veins) else 0.0
        region_iou = regions["iou"].mean() if len(regions) else 0.0
        print(f"{stem:<45s} {vein_iou:>7.2f}  {region_iou:>7.2f}  " f"{len(matched):>3d}/{total:<3d}")

    # --- Per-feature aggregate ---
    print(f"\n{'=' * 70}")
    print("Per-feature IoU (mean across specimens)")
    print(f"{'=' * 70}")
    print(f"{'Feature':<20s} {'Cat':>7s} {'Mean':>6s} {'Min':>6s} " f"{'Max':>6s} {'Detect%':>8s} {'N':>4s}")
    print("-" * 70)

    for cat_label, cat_name in [
        ("vein", "vein"),
        ("region", "region"),
        ("ectopic", "ectopic"),
    ]:
        cat_df = df[df["category"] == cat_name]
        if cat_df.empty:
            continue
        for feature, fgroup in cat_df.groupby("feature"):
            both = fgroup[fgroup["in_gt"] & fgroup["in_out"]]
            detect_rate = len(both[both["iou"] >= threshold]) / len(both) * 100 if len(both) else 0.0
            print(
                f"{feature:<20s} {cat_label:>7s} {fgroup['iou'].mean():>6.3f} "
                f"{fgroup['iou'].min():>6.3f} {fgroup['iou'].max():>6.3f} "
                f"{detect_rate:>7.1f}% {len(fgroup):>4d}"
            )

    # --- Overall ---
    print(f"\n{'=' * 70}")
    print("Overall")
    print(f"{'=' * 70}")

    veins = df[df["category"] == "vein"]
    regions = df[df["category"] == "region"]
    ectopic = df[df["category"] == "ectopic"]
    both_present = df[df["in_gt"] & df["in_out"]]
    matched = both_present[both_present["iou"] >= threshold]

    print(f"  Mean vein IoU:     {veins['iou'].mean():.3f}")
    print(f"  Mean region IoU:   {regions['iou'].mean():.3f}")
    if not ectopic.empty:
        print(f"  Mean ectopic IoU:  {ectopic['iou'].mean():.3f}")
    combined = df[df["category"].isin(["vein", "region"])]
    print(f"  Mean combined IoU: {combined['iou'].mean():.3f}")
    if len(both_present):
        print(
            f"  Detection rate:    {len(matched)}/{len(both_present)} "
            f"({len(matched) / len(both_present) * 100:.1f}%) "
            f"features with IoU >= {threshold:.2f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="IoU evaluation against GT_naming ground truth")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Ground truth GeoJSON directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Pipeline output GeoJSON directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="IoU threshold for detection (default: 0.5)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write detailed results to CSV")
    args = parser.parse_args()

    pairs = find_matched_pairs(args.gt_dir, args.out_dir)
    print(f"Found {len(pairs)} evaluable specimens")
    if not pairs:
        print("No matched GT/output pairs found.")
        sys.exit(1)

    all_rows = []
    for stem, gt_path, out_path in pairs:
        rows = evaluate_specimen(stem, gt_path, out_path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print_report(df, args.threshold)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Detailed results written to {args.csv}")


if __name__ == "__main__":
    main()
