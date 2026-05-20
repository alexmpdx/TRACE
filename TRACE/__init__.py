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

# Bumped when shipping a new Windows installer; the Help-tab update check
# compares this against the latest GitHub Release tag (strip the
# "windows-v" prefix when comparing) to tell the user whether they're
# behind. Bump in lockstep with the build tag you push.
__version__ = "0.1.0"

from TRACE.pipeline import TraceResult, trace_folder

__all__ = [
    "trace_folder",
    "TraceResult",
    "__version__",
]
