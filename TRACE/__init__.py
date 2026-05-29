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
# Single source of truth: the CI workflow reads this string and passes
# it to Inno Setup via /DMyAppVersion=…, AND verifies it matches the
# pushed git tag's version segment. So the day-to-day ritual is:
#   1. Bump __version__ here (full MAJOR.MINOR.PATCH form — no
#      truncating trailing zeroes, e.g. "0.2.0" not "0.2").
#   2. Push a matching tag: `git tag windows-v0.2.0 && git push origin
#      windows-v0.2.0`.
# A mismatch between the two fails the CI build fast, before producing
# an installer whose Help-tab updater shows the user as "out of date"
# forever.
__version__ = "0.1.57"

from TRACE.pipeline import TraceResult, trace_folder

__all__ = [
    "trace_folder",
    "TraceResult",
    "__version__",
]
