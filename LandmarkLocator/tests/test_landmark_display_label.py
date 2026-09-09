"""Overlay label text is a display concern; the data key stays "DTip"."""

from landmark_locator.scripts.visualize import LANDMARK_DISPLAY_NAMES, landmark_display_label


def test_dtip_is_painted_as_l3_distal():
    assert landmark_display_label("DTip") == "L3.d"


def test_other_distal_labels_pass_through():
    for name in ("L2.d", "L4.d", "L5.d", "L1-Rs", "ACV.p", "PCV.a"):
        assert landmark_display_label(name) == name


def test_snake_case_internal_names_are_spaced():
    assert landmark_display_label("alula_notch") == "alula notch"


def test_display_map_never_renames_a_data_key():
    # The map must only ever be consulted at paint time — guard against
    # someone adding a key that would collide with a real landmark name.
    for raw, shown in LANDMARK_DISPLAY_NAMES.items():
        assert raw != shown
