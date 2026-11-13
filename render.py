import torch

import comfy.model_management

from .curves import resample

STYLES = ["bounce", "pulse", "intensity", "meter"]

CURVE_BLUE = (0.30, 0.62, 1.0)
PLAYHEAD_YELLOW = (1.0, 0.84, 0.2)
STRIP_BG = (0.08, 0.09, 0.11)


def _soft(signed_dist):
    # ~1.5px anti-aliased edge, inside where signed_dist > 0
    return (signed_dist / 1.5 + 0.5).clamp(0.0, 1.0)


def control_video(curve, width, height, style):
    """White-on-black control frames, one per curve sample: IMAGE [N, H, W, 3]."""
    device = comfy.model_management.intermediate_device()
    v = torch.tensor(curve, device=device, dtype=torch.float32).view(-1, 1, 1)
    ys = torch.arange(height, device=device, dtype=torch.float32).view(1, -1, 1) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, -1) + 0.5
    short = min(width, height)

    if style == "intensity":
        frames = v.expand(-1, height, width)
    elif style == "meter":
        frames = _soft(ys - height * (1.0 - v)).expand(-1, -1, width)
    else:
        if style == "bounce":
            radius = 0.09 * short
            cy = height - radius - 0.04 * short - v * (height - 2 * radius - 0.08 * short)
        else:
            radius = (0.06 + 0.36 * v) * short
            cy = height / 2
        dist = ((xs - width / 2) ** 2 + (ys - cy) ** 2).sqrt()
        frames = _soft(radius - dist)
    return frames.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def overlay(images, curve, strip_fraction):
    """Burn a curve strip with a per-frame playhead into the bottom of each image."""
    out = images[..., :3].clone()
    n, height, width = out.shape[0], out.shape[1], out.shape[2]
    strip_h = max(8, round(height * strip_fraction))
    device, dtype = out.device, out.dtype
    pad = 0.12 * strip_h

    col = torch.tensor(resample(curve, width), device=device, dtype=dtype)
    y_col = pad + (1.0 - col) * (strip_h - 2 * pad)
    y_next = torch.cat([y_col[1:], y_col[-1:]])
    rows = torch.arange(strip_h, device=device, dtype=dtype).view(-1, 1) + 0.5
    lo = torch.minimum(y_col, y_next) - 1.2
    hi = torch.maximum(y_col, y_next) + 1.2
    line = (_soft(rows - lo) * _soft(hi - rows)).unsqueeze(-1)
    fill = (0.22 * _soft(rows - y_col)).unsqueeze(-1)

    blue = torch.tensor(CURVE_BLUE, device=device, dtype=dtype)
    strip = torch.tensor(STRIP_BG, device=device, dtype=dtype).expand(strip_h, width, 3)
    strip = strip + (blue - strip) * fill
    strip = strip + (blue - strip) * line

    region = out[:, height - strip_h:]
    region.mul_(0.2).add_(strip, alpha=0.8)

    playhead_x = torch.linspace(0, width - 1, n, device=device, dtype=dtype).view(-1, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, -1)
    head = _soft(1.5 - (xs - playhead_x).abs()).view(n, 1, width, 1)
    yellow = torch.tensor(PLAYHEAD_YELLOW, device=device, dtype=dtype)
    region.add_((yellow - region) * head)
    return out
