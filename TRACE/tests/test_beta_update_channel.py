"""Beta-channel release selection for the Help-tab updater."""

import sys
from pathlib import Path

# Sibling modules export bare package names that don't match their directory,
# so importing TRACE.* needs the same sys.path set run_cli.py builds.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    _ROOT,
    _ROOT / "HingeChopper",
    _ROOT / "modelTOjson",
    _ROOT / "identifyFeatures",
    _ROOT / "wingRotator",
    _ROOT / "measurementMaker",
    _ROOT / "scaleEstimator",
    _ROOT / "LandmarkLocator",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from TRACE.inline_panels import InlineHelpPanel  # noqa: E402


def _release(tag, *, prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "html_url": f"https://github.com/alexmpdx/TRACE/releases/tag/{tag}",
        "assets": [{"name": "TRACE-Setup.exe", "browser_download_url": f"https://x/{tag}.exe", "size": 123}],
    }


def test_pick_newest_release_prefers_highest_version_not_list_order():
    releases = [
        _release("windows-v0.2.34"),
        _release("windows-v0.2.35", prerelease=True),
        _release("windows-v0.2.30"),
    ]
    assert InlineHelpPanel._pick_newest_release(releases)["tag_name"] == "windows-v0.2.35"


def test_pick_newest_release_skips_drafts_and_non_semver_tags():
    releases = [
        _release("v0.1.0-assets"),
        _release("windows-v0.2.36", draft=True),
        _release("windows-v0.2.35", prerelease=True),
    ]
    assert InlineHelpPanel._pick_newest_release(releases)["tag_name"] == "windows-v0.2.35"


def test_pick_newest_release_empty():
    assert InlineHelpPanel._pick_newest_release([]) == {}
    assert InlineHelpPanel._pick_newest_release([_release("v0.1.0-assets")]) == {}


def test_parse_release_payload_reports_prerelease_flag():
    parsed = InlineHelpPanel._parse_release_payload(_release("windows-v0.2.35", prerelease=True))
    assert parsed["latest_version"] == "0.2.35"
    assert parsed["prerelease"] is True
    assert parsed["asset_url"] == "https://x/windows-v0.2.35.exe"
    assert InlineHelpPanel._parse_release_payload({})["latest_version"] == ""
