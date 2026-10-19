"""Legacy landmark names are aliased on read; the current name is used everywhere else."""

from landmark_locator.data.dataset import _normalize_name
from landmark_locator.inference.predict import canonicalize_gate_config
from landmark_locator.names import LEGACY_LANDMARK_NAMES, canonical_landmark_name
from landmark_locator.scripts.visualize import landmark_display_label


def test_legacy_dtip_maps_to_l3_distal_in_both_forms():
    assert canonical_landmark_name("DTip") == "L3.d"
    assert canonical_landmark_name("dtip") == "l3_d"
    assert canonical_landmark_name(" DTip ") == "L3.d"


def test_current_names_pass_through():
    for name in ("L3.d", "l3_d", "L2.d", "L4.d", "L5.d", "L1-Rs", "ACV.p", "alula notch", "alula_notch"):
        assert canonical_landmark_name(name) == name


def test_normalize_name_aliases_before_snake_casing():
    assert _normalize_name("DTip") == "l3_d"
    assert _normalize_name("L3.d") == "l3_d"
    assert _normalize_name("L1-Rs") == "l1_rs"
    assert _normalize_name("alula notch") == "alula_notch"


def test_alias_table_never_maps_a_name_to_itself():
    for legacy, current in LEGACY_LANDMARK_NAMES.items():
        assert legacy != current


def test_gate_config_renames_per_landmark_keys_and_core_list_at_every_depth():
    gate = {
        "peak": {"global": 0.1, "per_landmark": {"dtip": 0.5, "l1_rs": 0.2}},
        "core_landmarks": ["alula_notch", "dtip"],
        "second_peak_suppression_radius_px": 30,
        "tiers": {"strict": {"sharpness": {"per_landmark": {"dtip": 1.3}}}},
    }
    out = canonicalize_gate_config(gate)
    assert out["peak"]["per_landmark"] == {"l3_d": 0.5, "l1_rs": 0.2}
    assert out["core_landmarks"] == ["alula_notch", "l3_d"]
    assert out["tiers"]["strict"]["sharpness"]["per_landmark"] == {"l3_d": 1.3}
    assert out["second_peak_suppression_radius_px"] == 30
    # The input is left untouched.
    assert "dtip" in gate["peak"]["per_landmark"]


def test_display_label_spaces_snake_case_and_passes_geojson_names_through():
    assert landmark_display_label("alula_notch") == "alula notch"
    for name in ("L3.d", "L2.d", "L1-Rs", "ACV.p"):
        assert landmark_display_label(name) == name
