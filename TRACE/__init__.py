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

from TRACE.pipeline import TraceResult, trace_folder

__all__ = [
    "trace_folder",
    "TraceResult",
]
