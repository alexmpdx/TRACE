"""TRACE — Combined preprocessing + WingVeinAnalyzer pipeline.

Usage::

    from TRACE import trace_folder

    results, csv_path = trace_folder(
        input_dir="path/to/wings/",
        output_dir="path/to/output/",
        landmark_checkpoint="model.pt",
        segmentation_model_dir="seg_model/",
        scale=0.483,
    )
"""

from TRACE.landmark_measures import LandmarkMeasurements, compute_landmark_measurements, draw_landmark_overlay
from TRACE.pipeline import TraceResult, trace_folder

__all__ = [
    "trace_folder",
    "TraceResult",
    "LandmarkMeasurements",
    "compute_landmark_measurements",
    "draw_landmark_overlay",
]
