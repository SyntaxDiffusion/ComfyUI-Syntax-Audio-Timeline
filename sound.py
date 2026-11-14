import math

import torch

SAMPLE_RATE = 32000
HITS = ["thump", "crash", "boom"]
SOUNDS = HITS + ["wind", "murmur"]
AUDIO_MODES = ["loudness", "onsets"]


def rises(curve):
    """(start, end) frames of every stretch where the curve is at or above 0.5"""
    out, start = [], None
    for i, v in enumerate(list(curve) + [0.0]):
        if v >= 0.5 and start is None:
            start = i
        elif v < 0.5 and start is not None:
            out.append((start, i))
            start = None
    return out


def _noise(n, seed):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


def _bandpass(x, lo, hi):
    f = torch.fft.rfftfreq(x.numel(), 1 / SAMPLE_RATE)
    return torch.fft.irfft(torch.fft.rfft(x) * ((f >= lo) & (f < hi)), x.numel())


def _seconds(duration):
    return torch.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE


def hit(sound, held):
    """One hit, `held` seconds being how long the curve stays up after the rise."""
    if sound == "thump":
        t = _seconds(0.15)
        return _noise(t.numel(), 1) * torch.exp(-t / 0.015) * 0.6 + torch.sin(2 * math.pi * 80 * t) * torch.exp(-t / 0.05) * 0.8
    if sound == "crash":
        t = _seconds(0.6)
        cymbal = _bandpass(_noise(t.numel(), 4), 3000, 14000)
        return cymbal / cymbal.std() * 0.5 * torch.exp(-t / 0.18) + torch.sin(2 * math.pi * 70 * t) * torch.exp(-t / 0.06)
    # boom: a crack and a low rumble that rings for as long as the curve stays up
    dur = min(max(held, 0.5), 3.0)
    t = _seconds(dur)
    noise = _noise(t.numel(), 2)
    crack = _bandpass(noise, 1500, 9000) * torch.exp(-t / 0.06)
    rumble = _bandpass(noise, 25, 300) * torch.exp(-t / (dur / 2.5))
    body = torch.sin(2 * math.pi * torch.cumsum(35 + 60 * torch.exp(-t / 0.1), 0) / SAMPLE_RATE) * torch.exp(-t / (dur / 3))
    return torch.tanh(crack / crack.std() * 0.35 + rumble / rumble.std() * 0.9 + body * 1.2)


def wind(env):
    """broadband noise as loud as the curve, brighter as it rises"""
    noise = _noise(env.numel(), 3)
    low, high = _bandpass(noise, 80, 900), _bandpass(noise, 900, 6000)
    return env * (low / low.std() * 0.6 + high / high.std() * 0.4 * env) * 0.5


def murmur(env):
    """muffled speech-like murmur wherever the curve is up: a gliding voiced tone chopped into syllables"""
    g = torch.Generator().manual_seed(3)
    t = torch.arange(env.numel()) / SAMPLE_RATE
    drift = torch.rand(14, generator=g) * 2 - 1
    pitch = 150 * 2 ** (0.2 * _stretch(drift, env.numel()))
    phase = torch.cumsum(2 * math.pi * pitch / SAMPLE_RATE, 0)
    voice = _bandpass(sum(torch.sin(k * phase) / k for k in range(1, 14)), 220, 1300)
    wobble = _stretch(torch.rand(9, generator=g) * 2 - 1, env.numel())
    syllables = (0.5 - 0.5 * torch.cos(2 * math.pi * 4.5 * t + 2 * wobble)) ** 2
    return voice / voice.abs().max() * syllables * (env * 3).clamp(0, 1)


def _stretch(values, n):
    """linear interpolation of `values` over n samples"""
    pos = torch.linspace(0, values.numel() - 1, n)
    i = pos.long().clamp(max=values.numel() - 2)
    w = pos - i
    return values[i] * (1 - w) + values[i + 1] * w


def guide_audio(curve, fps, sound, clip=None):
    """Mono guide sound for a per-frame curve at SAMPLE_RATE: a hit (or `clip`) on every rise
    past 0.5 for the hit sounds, a continuous sound as loud as the curve for wind and murmur."""
    n = round(len(curve) / fps * SAMPLE_RATE)
    if clip is None and sound not in HITS:
        env = _stretch(torch.tensor(curve, dtype=torch.float32), n) if len(curve) > 1 else torch.full((n,), float(curve[0]))
        out = wind(env) if sound == "wind" else murmur(env)
    else:
        out = torch.zeros(n)
        for start, end in rises(curve):
            x = clip if clip is not None else hit(sound, (end - start) / fps)
            i = round(start / fps * SAMPLE_RATE)
            x = x[:n - i]
            out[i:i + x.numel()] += x
    peak = out.abs().max()
    return out / peak * 0.95 if peak > 0 else out


def audio_curve(waveform, sample_rate, length, fps, mode, range_db):
    """Per-frame curve from a soundtrack's loudness: the loudest frame maps to 1 and `range_db`
    below it to 0. onsets keeps only the frames where the loudness jumps up."""
    mono = waveform[0].mean(0)
    hop = sample_rate / fps
    db = []
    for k in range(length):
        x = mono[round(k * hop):round((k + 1) * hop)]
        db.append(20 * math.log10(x.square().mean().sqrt().item() + 1e-8) if x.numel() else -160.0)
    top = max(db)
    level = [min(max((v - top + range_db) / range_db, 0.0), 1.0) for v in db]
    if mode == "loudness":
        return level
    jumps = [max(b - a, 0.0) for a, b in zip([0.0] + level, level)]
    peak = max(jumps)
    return [j / peak for j in jumps] if peak > 0 else jumps
