import json
import math

import torch

INTERPOLATIONS = ["smooth", "linear", "step"]
DEFAULT_CURVE = '{"points": [[0.0, 0.1], [0.25, 0.9], [0.5, 0.1], [0.75, 0.9], [1.0, 0.1]]}'


def parse_points(curve_data):
    points = json.loads(curve_data)["points"]
    if not points:
        raise ValueError("Syntax Action Curve: curve_data has no points")
    return sorted((min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)) for x, y in points)


def _monotone_tangents(xs, ys):
    # Fritsch-Carlson: cubic through every key without overshooting past 0..1
    n = len(xs)
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] > xs[i] else 0.0 for i in range(n - 1)]
    m = [d[0]] + [0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2 for i in range(1, n - 1)] + [d[-1]]
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
            continue
        a, b = m[i] / d[i], m[i + 1] / d[i]
        s = a * a + b * b
        if s > 9:
            t = 3 / math.sqrt(s)
            m[i], m[i + 1] = t * a * d[i], t * b * d[i]
    return m


def evaluate(points, interpolation, length):
    """Sample the keyframes at `length` evenly spaced frames, returns a list of floats in 0..1."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    m = _monotone_tangents(xs, ys) if interpolation == "smooth" and len(points) > 1 else None
    out = []
    seg = 0
    for f in range(length):
        x = f / (length - 1) if length > 1 else 0.0
        if x <= xs[0]:
            out.append(ys[0])
            continue
        if x >= xs[-1]:
            out.append(ys[-1])
            continue
        while xs[seg + 1] < x:
            seg += 1
        x0, x1, y0, y1 = xs[seg], xs[seg + 1], ys[seg], ys[seg + 1]
        h = x1 - x0
        t = (x - x0) / h if h > 0 else 1.0
        if interpolation == "step":
            y = y0 if t < 1.0 else y1
        elif interpolation == "linear":
            y = y0 + (y1 - y0) * t
        else:
            t2, t3 = t * t, t * t * t
            y = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m[seg] + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m[seg + 1]
        out.append(min(max(y, 0.0), 1.0))
    # a key always lands on its nearest frame, so a spike narrower than a frame is not skipped
    if length > 1:
        for x, y in points:
            f = round(x * (length - 1))
            out[f] = max(out[f], y)
    return out


def resample(values, length):
    """Linearly stretch a curve over `length` samples on normalized time."""
    n = len(values)
    if n == length:
        return list(values)
    out = []
    for i in range(length):
        pos = i * (n - 1) / (length - 1) if length > 1 else 0.0
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        out.append(values[lo] + (values[hi] - values[lo]) * (pos - lo))
    return out


def to_latent_frames(values, length):
    """Fit a per-frame curve to `length` latent frames. Downsampling keeps the peak of the frames
    each latent frame covers, so a one-frame spike still lands at full strength."""
    n = len(values)
    if length >= n or length < 2:
        return resample(values, length)
    step = (n - 1) / (length - 1)
    return [max(values[max(0, math.ceil((i - 0.5) * step)):min(n, math.ceil((i + 0.5) * step))]) for i in range(length)]


def _time_dim(shape):
    # video [B, C, T, H, W]; audio is [B, C, T, F] (LTX) or [B, C, 2, T] (MiniMax H3): time is the longer axis
    if len(shape) == 4:
        return 2 if shape[2] >= shape[3] else 3
    return 2


def latent_time_weights(latent_shapes, curve, device):
    """Per-element weights with the curve stretched over each stream's time axis,
    packed the way comfy.utils.pack_latents packs multi-stream latents. Batch dim is 1."""
    parts = []
    for shape in latent_shapes:
        t_dim = _time_dim(shape)
        w = torch.tensor(to_latent_frames(curve, shape[t_dim]), device=device, dtype=torch.float32)
        view = [1] * len(shape)
        view[t_dim] = -1
        parts.append(w.view(view).expand(1, *shape[1:]))
    if len(parts) == 1:
        return parts[0]
    return torch.cat([p.reshape(1, 1, -1) for p in parts], dim=-1)
