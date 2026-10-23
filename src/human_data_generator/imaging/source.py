"""EXR source buffers. ray_range_m is raw ERP range, never a restored image."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from human_data_generator.io import read_combined_depth_index


@dataclass
class SourceERP:
    """RGB [H,W,3]; raw metre range and float IndexOB [H,W].

    RGB is linear float32 for Omni, saved uint8 display RGB for DS. The source
    range is retained untouched; no optical-Z conversion occurs here.
    """

    rgb: np.ndarray
    ray_range_m: np.ndarray
    instance: np.ndarray
    channels: list


def read_source_erp(source):
    """Read linear EXR RGB, raw ray range in metres and float instance IDs."""
    rgb, depth, index, channels = read_combined_depth_index(source)
    return SourceERP(rgb, depth, index, channels)


def read_ds_source(directory, name):
    """DS uses saved uint8 preview RGB but raw EXR range and instances."""
    _, depth, instance, channels = read_combined_depth_index(directory / f"{name}.exr")
    rgb = np.array(Image.open(directory / f"{name}_preview.png").convert("RGB"))
    return SourceERP(rgb, depth, instance, channels)
