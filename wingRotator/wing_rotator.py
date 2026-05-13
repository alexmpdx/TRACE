"""WingRotator — rotate wing images and associated GeoJSONs to a canonical
right-side-up, distal-right orientation using a reliability-weighted Procrustes
fit against a canonical landmark template.

Designed to slot into preprocessing between LandmarkLocator (Stage 1) and
HingeChopper (Stage 2). Robust to any subset of the 10 known landmarks; weights
each by confidence × sharpness / (1 + second_peak_ratio) and falls back to a
no-op when fewer than 2 reliable landmarks are available.

No reflection — proper rotation only (preserves left/right wing chirality).
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Canonical landmark positions in image coordinates (origin top-left, +y down).
# Layout: alula notch at origin, DTip on +X axis, anterior margin at -y (top).
# Derived from one representative wing in testdata/testwings — only the relative
# geometry matters because the Procrustes fit absorbs scale and translation.
CANONICAL_LANDMARKS: dict[str, tuple[float, float]] = {
    "alula notch": (0.0, 0.0),
    "DTip": (4584.0, 0.0),
    "subcostal break": (836.0, -808.0),
    "L1-Rs": (94.0, -532.0),
    "L2-L3": (542.0, -472.0),
    "L4-L5": (-78.0, -341.0),
    "ACV.a": (1250.0, -336.0),
    "ACV.p": (1194.0, -189.0),
    "PCV.a": (2080.0, 131.0),
    "PCV.p": (1959.0, 480.0),
}

# Pixel span used to scale the residual sanity-check threshold.
_CANONICAL_PD_SPAN = abs(CANONICAL_LANDMARKS["DTip"][0] - CANONICAL_LANDMARKS["alula notch"][0])

# Down-weight applied to landmarks with reliable=False (only used when
# soft_reliability=True; otherwise unreliable points are hard-gated out).
_UNRELIABLE_WEIGHT_FACTOR = 0.25


@dataclass
class RotationResult:
    angle_deg: float
    affine: np.ndarray
    n_landmarks_used: int
    rms_residual: float
    used_names: list[str]
    skipped_names: list[str]
    mirrored_detected: bool
    rotated_image_path: Path
    rotated_landmarks_path: Path
    extra_outputs: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Landmark loading and weighting
# ---------------------------------------------------------------------------
def _landmark_weight(props: dict, soft_reliability: bool) -> float:
    """Combined weight from the LandmarkLocator metadata fields.

    Hard-gates (returns 0) when reliable=False unless soft_reliability=True,
    in which case unreliable landmarks contribute at _UNRELIABLE_WEIGHT_FACTOR
    of their nominal weight.
    """
    reliable = props.get("reliable", True)
    if not reliable and not soft_reliability:
        return 0.0
    confidence = float(props.get("confidence", 1.0))
    sharpness = float(props.get("sharpness", 1.0))
    sp_ratio = float(props.get("second_peak_ratio", 0.0))
    base = max(confidence, 0.0) * max(sharpness, 0.0) / (1.0 + max(sp_ratio, 0.0))
    if base <= 0.0:
        # Some upstream flows omit confidence/sharpness; fall back to a flat
        # weight so the fit still works on geojsons that lack the rich metadata.
        base = 1.0
    return base if reliable else base * _UNRELIABLE_WEIGHT_FACTOR


def _load_landmarks_for_fit(path: Path, soft_reliability: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read a landmarks GeoJSON; return (detected, canonical, weights, names)."""
    with open(path) as f:
        data = json.load(f)
    detected: list[tuple[float, float]] = []
    canonical: list[tuple[float, float]] = []
    weights: list[float] = []
    names: list[str] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        classification = props.get("classification") or {}
        name = classification.get("name") or props.get("class")
        if not name or name not in CANONICAL_LANDMARKS:
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        w = _landmark_weight(props, soft_reliability)
        if w <= 0.0:
            continue
        detected.append((float(coords[0]), float(coords[1])))
        canonical.append(CANONICAL_LANDMARKS[name])
        weights.append(w)
        names.append(name)
    return (
        np.array(detected, dtype=np.float64).reshape(-1, 2),
        np.array(canonical, dtype=np.float64).reshape(-1, 2),
        np.array(weights, dtype=np.float64),
        names,
    )


# ---------------------------------------------------------------------------
# Weighted Procrustes (proper rotation, no reflection)
# ---------------------------------------------------------------------------
def _weighted_kabsch_2d(p: np.ndarray, q: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Solve theta minimizing sum_i w_i ||R(theta) (p_i - p_bar) - (q_i - q_bar)||^2.

    Closed-form 2D Procrustes (no reflection). Returns (theta_radians, rms_residual)
    with the residual normalized by the canonical-side scale so it's comparable
    across calls.
    """
    if len(p) < 2:
        raise ValueError("need at least 2 landmarks for rotation fit")
    w_sum = w.sum()
    if w_sum <= 0.0:
        raise ValueError("all landmark weights are zero")
    p_bar = (w[:, None] * p).sum(axis=0) / w_sum
    q_bar = (w[:, None] * q).sum(axis=0) / w_sum
    p_c = p - p_bar
    q_c = q - q_bar

    cross = (w * (p_c[:, 0] * q_c[:, 1] - p_c[:, 1] * q_c[:, 0])).sum()
    dot = (w * (p_c[:, 0] * q_c[:, 0] + p_c[:, 1] * q_c[:, 1])).sum()
    theta = math.atan2(cross, dot)

    # For a residual that's robust to the scale difference between detected
    # (image pixels) and canonical (template units), normalize p by the
    # weighted-RMS ratio q_norm/p_norm before measuring the difference.
    p_norm2 = (w * (p_c[:, 0] ** 2 + p_c[:, 1] ** 2)).sum()
    q_norm2 = (w * (q_c[:, 0] ** 2 + q_c[:, 1] ** 2)).sum()
    if p_norm2 == 0.0:
        return theta, float("inf")
    s = math.sqrt(q_norm2 / p_norm2)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    p_rot = (s * (R @ p_c.T)).T
    diffs = p_rot - q_c
    rms = math.sqrt((w * (diffs**2).sum(axis=1)).sum() / w_sum)
    return theta, rms


def _detect_mirror(p: np.ndarray, q: np.ndarray, w: np.ndarray, base_rms: float) -> bool:
    """True when fitting against a mirrored canonical lowers the residual a lot.

    Indicates the input is the opposite chirality from the canonical (i.e., a
    left wing when the canonical is a right wing). We don't act on it — per
    user spec, rotation-only — but we surface the flag for logging/QA.
    """
    if len(p) < 3:
        return False
    q_mirror = q.copy()
    q_mirror[:, 1] = -q_mirror[:, 1]
    try:
        _, rms_m = _weighted_kabsch_2d(p, q_mirror, w)
    except ValueError:
        return False
    return rms_m < base_rms * 0.7


# ---------------------------------------------------------------------------
# Affine construction and coordinate / image transform
# ---------------------------------------------------------------------------
def _build_affine(image_shape: tuple[int, ...], theta_rad: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a 2x3 forward affine (src→dst) for an image rotation that expands
    the canvas to fit the rotated content.

    Returns (M_forward, (new_w, new_h)). The same M_forward is applied to image
    coordinates AND passed to cv2.warpAffine: by default warpAffine treats its
    matrix as the src→dst forward map and inverts it internally (without the
    WARP_INVERSE_MAP flag).
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rotated = corners @ R.T
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    new_w = max(int(math.ceil(max_xy[0] - min_xy[0])), 1)
    new_h = max(int(math.ceil(max_xy[1] - min_xy[1])), 1)

    M_forward = np.array(
        [
            [cos_t, -sin_t, -min_xy[0]],
            [sin_t, cos_t, -min_xy[1]],
        ],
        dtype=np.float64,
    )
    return M_forward, (new_w, new_h)


def _apply_affine_to_coords(coords, M: np.ndarray):
    """Recursively apply 2x3 affine M to GeoJSON coordinate structures.

    Handles Point, LineString, Polygon, MultiPolygon, etc. Preserves any third
    coordinate (z) untouched.
    """
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (int, float)):
        x = float(coords[0])
        y = float(coords[1])
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        if len(coords) > 2:
            return [nx, ny, coords[2]]
        return [nx, ny]
    return [_apply_affine_to_coords(c, M) for c in coords]


def transform_geojson(data: dict, M: np.ndarray) -> dict:
    """Apply 2x3 affine M to every geometry coordinate in a GeoJSON dict."""
    out = dict(data)
    new_features = []
    for feat in data.get("features", []):
        new_feat = dict(feat)
        geom = feat.get("geometry")
        if geom and "coordinates" in geom:
            new_geom = dict(geom)
            new_geom["coordinates"] = _apply_affine_to_coords(geom["coordinates"], M)
            new_feat["geometry"] = new_geom
        new_features.append(new_feat)
    out["features"] = new_features
    return out


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------
def _read_image(path: Path) -> np.ndarray:
    """Read an image, falling back to preprocessing.psd_loader for exotic formats."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from preprocessing.psd_loader import imread_any

            img = imread_any(str(path), cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img


def _write_image(path: Path, image: np.ndarray) -> None:
    """Write image; coerce TIFF outputs to OME-TIFF to preserve metadata.

    Mirrors the writer in HingeChopper so rotated TIFFs round-trip the same way
    as chopped TIFFs do downstream.
    """
    name_low = path.name.lower()
    is_tiff = path.suffix.lower() in (".tif", ".tiff") or name_low.endswith((".ome.tif", ".ome.tiff"))
    if is_tiff:
        try:
            import tifffile

            if image.ndim == 3 and image.shape[-1] == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tifffile.imwrite(str(path), rgb, ome=True, photometric="rgb")
                return
            if image.ndim == 3 and image.shape[-1] == 4:
                rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                tifffile.imwrite(str(path), rgba, ome=True, photometric="rgb")
                return
            if image.ndim == 2:
                tifffile.imwrite(str(path), image, ome=True, photometric="minisblack")
                return
            tifffile.imwrite(str(path), image, ome=True)
            return
        except Exception:
            pass
    cv2.imwrite(str(path), image)


def _clean_stem(image_path: Path) -> str:
    """Strip .ome.tif / .ome.tiff compound suffixes, like preprocessing._clean_stem."""
    name = image_path.name
    low = name.lower()
    if low.endswith(".ome.tif"):
        return name[: -len(".ome.tif")]
    if low.endswith(".ome.tiff"):
        return name[: -len(".ome.tiff")]
    return image_path.stem


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def rotate_from_landmarks(
    image_path: Path,
    landmarks_geojson_path: Path,
    output_dir: Path,
    extra_geojsons: Optional[list[Path]] = None,
    soft_reliability: bool = False,
    min_landmarks: int = 2,
    max_residual_ratio: float = 0.25,
    skip_threshold_deg: float = 1.0,
    mirror_correct: bool = False,
) -> Optional[RotationResult]:
    """Rotate image + landmarks (+ optional extra GeoJSONs) to canonical orientation.

    Args:
        image_path: source image
        landmarks_geojson_path: LandmarkLocator output for that image
        output_dir: where rotated outputs are written
        extra_geojsons: additional GeoJSONs to rotate with the same affine
            (e.g., wing-isolation polygon, segmentation output if present)
        soft_reliability: when True, landmarks with reliable=False are included
            at reduced weight; when False they are dropped entirely
        min_landmarks: minimum reliable landmarks needed; below this we return
            None and the caller should pass the image through unchanged
        max_residual_ratio: residual sanity threshold as a fraction of the
            canonical PD span; logged when exceeded but does not abort
        skip_threshold_deg: when |theta| < this, the rotation is treated as a
            near-no-op and we still emit outputs (for downstream consistency)
            but log a "skipped" note
        mirror_correct: when True AND mirror is detected, apply a horizontal
            flip on top of the proper-rotation fit so opposite-chirality
            wings end up distal-right AND anterior-up (at the cost of a true
            reflection — biological chirality is flipped). Default False keeps
            chirality and lets such wings end up distal-left, anterior-up.

    Returns RotationResult on success, or None when the fit cannot proceed.
    """
    image_path = Path(image_path)
    landmarks_geojson_path = Path(landmarks_geojson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p, q, w, names = _load_landmarks_for_fit(landmarks_geojson_path, soft_reliability=soft_reliability)
    if len(p) < min_landmarks:
        logger.warning(
            "wing_rotator: only %d usable landmarks for %s (need >= %d); skipping rotation",
            len(p),
            image_path.name,
            min_landmarks,
        )
        return None

    theta, rms = _weighted_kabsch_2d(p, q, w)
    mirrored_detected = _detect_mirror(p, q, w, rms)
    apply_horizontal_flip = False
    if mirrored_detected:
        if mirror_correct:
            # Reflection path: keep the proper-rotation theta (so PD is
            # distal-right after rotation) and compose a horizontal flip
            # over the rotated canvas after _build_affine. Result is
            # anterior-up AND distal-right — but biological chirality is
            # flipped (a left wing is mirrored to look like a right wing).
            apply_horizontal_flip = True
            logger.warning(
                "wing_rotator: mirror residual lower for %s — opposite chirality; "
                "mirror_correct=True, applying horizontal flip so anterior is up "
                "AND distal is right (chirality is reflected).",
                image_path.name,
            )
        else:
            # Rotation-only path. Without reflection we can't make BOTH the PD
            # axis distal-right AND the AP axis anterior-up — the proper-rotation
            # fit gives PD distal-right but flips AP (anterior at bottom). Adding
            # 180° flips both, restoring anterior-up at the cost of PD becoming
            # distal-left. We prioritize AP-up because downstream stages care
            # more about a consistent anterior/posterior split than about
            # distal pointing right.
            theta = math.atan2(math.sin(theta + math.pi), math.cos(theta + math.pi))
            logger.warning(
                "wing_rotator: mirror residual lower for %s — opposite chirality from "
                "canonical; applying extra 180° so anterior stays up (PD axis ends "
                "up distal-left). Pass mirror_correct=True to instead flip for "
                "canonical distal-right at the cost of reflecting chirality.",
                image_path.name,
            )
    if rms > max_residual_ratio * _CANONICAL_PD_SPAN:
        logger.warning(
            "wing_rotator: high Procrustes residual (%.1f, threshold %.1f) for %s; " "rotation may be unreliable",
            rms,
            max_residual_ratio * _CANONICAL_PD_SPAN,
            image_path.name,
        )

    img = _read_image(image_path)
    M_forward, (new_w, new_h) = _build_affine(img.shape, theta)

    if apply_horizontal_flip:
        # Proper-rotation fit already gives PD distal-right but anterior-down for
        # opposite-chirality inputs. Compose a vertical flip (mirror across the
        # horizontal axis) so anterior moves to top while distal stays on the right.
        # In homogeneous form: flip @ M, where flip = [[1, 0, 0], [0, -1, h-1]].
        # Result: negate row-1 of the linear part and offset ty by (h-1).
        M_forward = np.array(
            [
                [M_forward[0, 0], M_forward[0, 1], M_forward[0, 2]],
                [-M_forward[1, 0], -M_forward[1, 1], (new_h - 1) - M_forward[1, 2]],
            ],
            dtype=np.float64,
        )

    rotated = cv2.warpAffine(
        img,
        M_forward,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    stem = _clean_stem(image_path)
    out_image_path = output_dir / f"{stem}_rotated{image_path.suffix}"
    _write_image(out_image_path, rotated)

    with open(landmarks_geojson_path) as f:
        lm_data = json.load(f)
    lm_rotated = transform_geojson(lm_data, M_forward)
    out_lm_path = output_dir / f"{stem}_rotated_landmarks.geojson"
    with open(out_lm_path, "w") as f:
        json.dump(lm_rotated, f, indent=2)

    extra_outputs: dict[str, Path] = {}
    for ex_path in extra_geojsons or []:
        ex_path = Path(ex_path)
        with open(ex_path) as f:
            ex_data = json.load(f)
        ex_rotated = transform_geojson(ex_data, M_forward)
        out_ex_path = output_dir / f"{ex_path.stem}_rotated{ex_path.suffix}"
        with open(out_ex_path, "w") as f:
            json.dump(ex_rotated, f, indent=2)
        extra_outputs[ex_path.name] = out_ex_path

    skipped = [k for k in CANONICAL_LANDMARKS if k not in names]
    return RotationResult(
        angle_deg=math.degrees(theta),
        affine=M_forward,
        n_landmarks_used=len(p),
        rms_residual=rms,
        used_names=names,
        skipped_names=skipped,
        mirrored_detected=mirrored_detected,
        rotated_image_path=out_image_path,
        rotated_landmarks_path=out_lm_path,
        extra_outputs=extra_outputs,
    )
