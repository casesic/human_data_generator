"""OpenEXR channel loading used by fisheye post-processing."""

from pathlib import Path

import numpy as np


def read_combined_depth_index(path: str | Path):
    import OpenEXR

    file = OpenEXR.File(str(path))
    channels = {}
    for part in file.parts:
        for name, channel in part.channels.items():
            channels[name] = np.asarray(channel.pixels, np.float32)

    def pick(token):
        for name, value in channels.items():
            if name == token or name.endswith("." + token) or name.split(".")[0] == token:
                return value
        return None

    combined = pick("Combined")
    if combined is not None and combined.ndim == 3:
        rgb = combined[..., :3]
    else:
        values = []
        for name in ("R", "G", "B"):
            value = pick("Combined." + name)
            if value is None:
                value = pick(name)
            values.append(value)
        rgb = np.stack(values, axis=-1)
    depth = pick("Depth")
    index = pick("IndexOB")
    if depth is None or index is None:
        raise RuntimeError(f"missing EXR passes: {list(channels)}")
    return rgb, depth, index, list(channels)
