"""TRACE — Wing analysis pipeline (preprocessing stage only; analysis TODO).

Usage::

    from TRACE import trace_folder

    results = trace_folder(
        input_dir="path/to/wings/",
        output_dir="path/to/output/",
        landmark_checkpoint="model.pt",
        segmentation_model_dir="seg_model/",
        scale=0.483,
    )
"""

# Bumped when shipping a new Windows installer. The Help-tab update
# check compares this against the latest GitHub Release tag (strip the
# "windows-v" prefix when comparing) to tell the user whether they're
# behind.
#
# IMPORTANT: keep all three in lockstep using the full MAJOR.MINOR.PATCH
# form (no truncating trailing zeroes):
#   1. __version__ here in TRACE/__init__.py
#   2. MyAppVersion in TRACE/build/installer.iss
#   3. git tag pushed to trigger the build, prefixed `windows-v`
#      (e.g. `windows-v0.2.0`, not `windows-v0.2`)
# Tags like `windows-v0.2` strip to `0.2` and won't match `__version__`
# of `0.2.0`, which is why the update check shows the user as "out of
# date" forever.
__version__ = "0.1.1"

from TRACE.pipeline import TraceResult, trace_folder

__all__ = [
    "trace_folder",
    "TraceResult",
    "__version__",
]
