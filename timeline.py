import math

import torch

H3_FPS = 24
SHOT_SECONDS = 5.5
ENV_RATE = 100  # onset envelope frames per second
TEMPO_RANGE = (70.0, 140.0)  # one octave: faster songs read at half time, which still cuts on downbeats


def h3_frames(n):
    """Smallest MiniMax H3 frame count (17k + 5) that holds `n` frames."""
    n = max(int(n), 5)
    return n + (5 - n) % 17


def auto_bars(bpm):
    """Whole bars closest to SHOT_SECONDS, at least one."""
    return max(1, round(SHOT_SECONDS / (240.0 / bpm)))


def shots_from_cuts(cuts, fps=H3_FPS):
    """Shots between consecutive cut points (seconds). Frame counts come from the rounded cut points,
    so the joined shots stay on the track."""
    shots = []
    for a, b in zip(cuts, cuts[1:]):
        frames = round(b * fps) - round(a * fps)
        shots.append({"start": a, "frames": frames, "length": h3_frames(frames)})
    return shots


def bar_cuts(downbeat, bpm, bars, n_shots):
    """Cut points (seconds) for `n_shots` back-to-back shots sharing `bars` whole bars from `downbeat`,
    the longer shots first."""
    cuts, at = [downbeat], 0
    for i in range(n_shots):
        at += bars // n_shots + (i < bars % n_shots)
        cuts.append(downbeat + at * 240.0 / bpm)
    return cuts


def _band(x, sample_rate, lo, hi):
    f = torch.fft.rfftfreq(x.numel(), 1 / sample_rate)
    return torch.fft.irfft(torch.fft.rfft(x) * ((f >= lo) & (f < hi)), x.numel())


def find_kicks(mono, sample_rate):
    """Kick drum hits (seconds) in a drums stem: rises in the 35-120 Hz band above its own held level, so
    the ring of a tom or a bass note doesn't count twice. In a full mix, bass notes that start hard count too."""
    mono = mono.float()
    hop = round(sample_rate * 0.005)
    n_fft = 1 << round(math.log2(sample_rate * 0.093))
    spec = torch.stft(mono, n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    f = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    low = spec[(f >= 35) & (f < 120)]
    held = torch.nn.functional.pad(low.unsqueeze(0), (50, 50), mode="replicate").squeeze(0).unfold(1, 101, 1).quantile(0.3, dim=-1)
    energy = torch.relu(low - held).sum(0)
    energy = torch.log1p(energy / (energy.max() + 1e-9) * 3)
    onset = torch.relu(torch.diff(energy, prepend=energy[:1]))
    onset = torch.nn.functional.avg_pool1d(onset.view(1, 1, -1), 3, 1, 1).flatten()
    if not (onset > 0).any():
        return []
    # peaks: the highest within 90 ms, above the median of the second around it and a floor
    peak = torch.nn.functional.max_pool1d(onset.view(1, 1, -1), 37, 1, 18).flatten()
    median = torch.nn.functional.pad(onset.view(1, -1), (100, 100), mode="replicate").flatten().unfold(0, 201, 1).median(-1).values
    floor = 0.25 * torch.quantile(onset[onset > 0], 0.98)
    found = torch.nonzero((onset == peak) & (onset > median + 0.5 * floor) & (onset > floor)).flatten() * hop

    # the 93 ms STFT frame blurs the time: each hit is where the low end rises fastest near its frame
    w = round(0.002 * sample_rate) * 2 + 1
    level = torch.nn.functional.avg_pool1d(_band(mono, sample_rate, 30, 200).square().view(1, 1, -1), w, 1, w // 2).flatten().sqrt()
    rise = torch.diff(level, prepend=level[:1])
    kicks, reach = [], round(0.08 * sample_rate)
    for i in found.tolist():
        a = max(i - reach, 0)
        t = (a + int(rise[a:i + reach].argmax())) / sample_rate
        if not kicks or t - kicks[-1] > 0.03:
            kicks.append(t)
    return kicks


def on_the_kicks(grid, kicks):
    """`grid` moved half a beat when most kicks land on its off-beats: in a 4/4 song the kick is on the beat."""
    beat = 60.0 / grid["bpm"]
    phase = [((k - grid["downbeat"]) / beat) % 1 for k in kicks]
    if sum(abs(p - 0.5) < 0.15 for p in phase) > sum(min(p, 1 - p) < 0.15 for p in phase):
        d = grid["downbeat"] + beat / 2
        return {**grid, "downbeat": d - beat if d - beat >= 0 else d}
    return grid


def peaks(mono, n=2000):
    """The loudest sample in each of `n` stretches, 0..1, for drawing the waveform."""
    step = max(1, mono.numel() // n)
    p = mono[: mono.numel() // step * step].float().abs().view(-1, step).amax(1)
    return [round(v, 3) for v in (p / p.max().clamp(min=1e-9)).tolist()]


def _loudness(mono, sample_rate, rate=ENV_RATE):
    """RMS level in dB below the loudest frame, at `rate` frames per second."""
    hop = round(sample_rate / rate)
    n = mono.numel() // hop
    rms = mono[: n * hop].float().view(n, hop).square().mean(1).sqrt()
    db = 20 * torch.log10(rms + 1e-8)
    return db - db.max()


def _sound_end(db, start, rate=ENV_RATE, floor=-50.0):
    """Seconds where the sound ends: after the last frame above `floor` dB."""
    loud = torch.nonzero(db[int(start * rate):] > floor).flatten()
    return start + (int(loud[-1]) + 1) / rate if loud.numel() else db.numel() / rate


def even_cuts(mono, sample_rate, start, n_shots, end=None):
    """`n_shots` equal shots from `start` to `end`, by default the end of the sound."""
    if end is None:
        end = _sound_end(_loudness(mono, sample_rate), start)
    return [start + i * (end - start) / n_shots for i in range(n_shots + 1)]


def pause_cuts(mono, sample_rate, start, n_shots, end=None, move_end=False):
    """Like even_cuts, but each cut moves into the quietest moment near it, so no shot cuts a line
    or a hit in half. With `move_end` the end moves into a pause too."""
    db = _loudness(mono, sample_rate)
    smooth = torch.nn.functional.avg_pool1d(db.view(1, 1, -1), 15, 1, 7, count_include_pad=False).flatten()
    cuts = even_cuts(mono, sample_rate, start, n_shots, end)
    reach = min(1.5, (cuts[-1] - start) / n_shots / 3)
    for i in range(1, n_shots + 1 if move_end else n_shots):
        lo, hi = int((cuts[i] - reach) * ENV_RATE), int((cuts[i] + reach) * ENV_RATE) + 1
        window = smooth[lo:hi]
        # the middle of the quiet stretch around the quietest moment, so the cut sits inside the pause
        quiet = window <= window.min() + 6.0
        j = int(window.argmin())
        a, b = j, j
        while a > 0 and quiet[a - 1]:
            a -= 1
        while b < quiet.numel() - 1 and quiet[b + 1]:
            b += 1
        cuts[i] = (lo + (a + b) / 2 + 0.5) / ENV_RATE
    return cuts


def _hit(sound, sample_rate):
    """One synthetic hit: thump (punches, blows, kicks), crash (cymbals) or boom (thunder, explosions)."""
    def noise(n, seed):
        return torch.randn(n, generator=torch.Generator().manual_seed(seed))

    t = torch.arange(int({"thump": 0.15, "crash": 0.6, "boom": 1.0}[sound] * sample_rate)) / sample_rate
    if sound == "thump":
        return noise(t.numel(), 1) * torch.exp(-t / 0.015) * 0.6 + torch.sin(2 * math.pi * 80 * t) * torch.exp(-t / 0.05) * 0.8
    if sound == "crash":
        cymbal = _band(noise(t.numel(), 4), sample_rate, 3000, 14000)
        return cymbal / cymbal.std() * 0.5 * torch.exp(-t / 0.18) + torch.sin(2 * math.pi * 70 * t) * torch.exp(-t / 0.06)
    n = noise(t.numel(), 2)
    crack = _band(n, sample_rate, 1500, 9000) * torch.exp(-t / 0.06)
    rumble = _band(n, sample_rate, 25, 300) * torch.exp(-t / 0.4)
    body = torch.sin(2 * math.pi * torch.cumsum(35 + 60 * torch.exp(-t / 0.1), 0) / sample_rate) * torch.exp(-t / 0.33)
    return torch.tanh(crack / crack.std() * 0.35 + rumble / rumble.std() * 0.9 + body * 1.2)


HIT_SOUNDS = ["thump", "crash", "boom"]
CURVE_SOUNDS = ["flow", "motor", "pulse", "mumble"]


def curve_level(points, n, sample_rate):
    """A drawn curve at each of `n` samples: straight lines between its (seconds, 0..1) points, silent outside them."""
    if len(points) < 2:
        return torch.zeros(n)
    t = torch.arange(n) / sample_rate
    xs = torch.tensor([float(p[0]) for p in points])
    ys = torch.tensor([float(p[1]) for p in points])
    i = torch.searchsorted(xs, t).clamp(1, len(points) - 1)
    w = ((t - xs[i - 1]) / (xs[i] - xs[i - 1]).clamp(min=1e-6)).clamp(0, 1)
    return torch.where((t >= xs[0]) & (t <= xs[-1]), ys[i - 1] + (ys[i] - ys[i - 1]) * w, 0.0)


def _curve_sound(sound, level, sample_rate, seed):
    """A sound that follows `level` (0..1 per sample): flow (wind, water, light), motor (an engine, revving
    as it rises), pulse (magic, sci-fi, faster as it rises) or mumble (someone talking, faster as it rises)."""
    n = level.numel()
    g = torch.Generator().manual_seed(seed)
    on = (level > 0).float()

    def wander(per_second):
        v = torch.rand(int(n / sample_rate * per_second) + 2, generator=g)
        return torch.nn.functional.interpolate(v.view(1, 1, -1), size=n, mode="linear", align_corners=True).flatten()

    def cycles(freq):
        return torch.cumsum(freq, 0) / sample_rate

    if sound == "flow":
        noise = torch.randn(n, generator=g)
        low, high = _band(noise, sample_rate, 60, 500), _band(noise, sample_rate, 500, 5000)
        return (low / low.std() * (1 - 0.5 * level) + high / high.std() * level) * level * (0.75 + 0.25 * wander(3))
    if sound == "motor":
        c = cycles(28 + 100 * level)
        x = 2 * (c % 1) - 1 + 0.5 * torch.sin(math.pi * c) + 0.15 * torch.randn(n, generator=g)
        return x * on * (0.35 + 0.65 * level)
    if sound == "pulse":
        c = cycles(180 * (1 + level))
        beat = 0.5 + 0.5 * torch.sin(2 * math.pi * cycles(2 + 12 * level))
        return (torch.sin(2 * math.pi * c) + 0.4 * torch.sin(3 * math.pi * c)) * beat * level
    # mumble: a voice of harmonics under two formants, a new vowel each syllable
    f0 = 115 + 45 * wander(2)
    c = cycles(f0)
    syllables = cycles(2.5 + 4.5 * level)
    vowels = torch.tensor([(730.0, 1090.0), (530.0, 1840.0), (270.0, 2290.0), (570.0, 840.0), (300.0, 870.0), (660.0, 1720.0)])
    vowel = vowels[torch.randint(0, len(vowels), (int(syllables[-1]) + 1,), generator=g)][syllables.long()]
    x = torch.zeros(n)
    for k in range(1, 25):
        weight = torch.exp(-((k * f0 - vowel[:, 0]) / 90) ** 2) + 0.7 * torch.exp(-((k * f0 - vowel[:, 1]) / 150) ** 2) + 0.03
        x += weight * torch.sin(2 * math.pi * k * c)
    return x * torch.sin(math.pi * (syllables % 1)) ** 2 * on * (0.5 + 0.5 * level)


def drawn_audio(hits, curves, duration, sample_rate):
    """Mono track of `duration` seconds: a hit at each (seconds, sound) of `hits`, plus each curve's sound
    ({"sound", "points": [(seconds, 0..1)]}) following its curve."""
    out = torch.zeros(round(duration * sample_rate))
    for t, sound in hits:
        one = _hit(sound, sample_rate)
        i = round(t * sample_rate)
        if 0 <= i < out.numel():
            n = min(one.numel(), out.numel() - i)
            out[i:i + n] += one[:n]
    for k, curve in enumerate(curves):
        level = curve_level(curve["points"], out.numel(), sample_rate)
        if level.max() > 0:
            x = _curve_sound(curve["sound"], level, sample_rate, k)
            out += x / x.abs().max() * 0.6
    peak = out.abs().max()
    return out / peak * 0.9 if peak > 0 else out


def frame_levels(mono, sample_rate, frames, fps=H3_FPS):
    """Loudness of each video frame, 0..1 over the 40 dB below the loudest."""
    hop = sample_rate / fps
    db = []
    for k in range(frames):
        x = mono[round(k * hop):round((k + 1) * hop)]
        db.append(20 * math.log10(x.square().mean().sqrt().item() + 1e-8) if x.numel() else -160.0)
    top = max(db)
    return [min(max((v - top + 40) / 40, 0.0), 1.0) for v in db]


def _soft(signed_dist):
    # ~1.5px anti-aliased edge, inside where signed_dist > 0
    return (signed_dist / 1.5 + 0.5).clamp(0.0, 1.0)


def strip_overlay(images, level, cuts=(), hits=(), offset=0, strip_fraction=0.12):
    """Burn a strip into the bottom of the frames: the guide's loudness, white cut lines, red hit ticks
    and a yellow playhead. `level` has one value per frame of the whole video; `cuts` and `hits` are
    frame indices in it; `images` start at frame `offset`."""
    out = images[..., :3].clone()
    n, height, width = out.shape[0], out.shape[1], out.shape[2]
    strip_h = max(8, round(height * strip_fraction))
    dev, dt = out.device, out.dtype
    pad = 0.12 * strip_h
    col = torch.nn.functional.interpolate(torch.tensor(level, dtype=dt).view(1, 1, -1), size=width, mode="linear", align_corners=True).flatten().to(dev)
    y_col = pad + (1.0 - col) * (strip_h - 2 * pad)
    y_next = torch.cat([y_col[1:], y_col[-1:]])
    rows = torch.arange(strip_h, device=dev, dtype=dt).view(-1, 1) + 0.5
    line = (_soft(rows - (torch.minimum(y_col, y_next) - 1.2)) * _soft(torch.maximum(y_col, y_next) + 1.2 - rows)).unsqueeze(-1)
    fill = (0.22 * _soft(rows - y_col)).unsqueeze(-1)
    xs = torch.arange(width, device=dev, dtype=dt)

    total = len(level)

    def marks(at, half_width):
        if not len(at):
            return torch.zeros(width, device=dev, dtype=dt)
        px = torch.tensor(at, device=dev, dtype=dt).view(-1, 1) * (width - 1) / max(total - 1, 1)
        return _soft(half_width - (xs - px).abs()).amax(0)

    blue = torch.tensor((0.30, 0.62, 1.0), device=dev, dtype=dt)
    strip = torch.tensor((0.08, 0.09, 0.11), device=dev, dtype=dt).expand(strip_h, width, 3)
    strip = strip + (blue - strip) * fill
    strip = strip + (blue - strip) * line
    strip = strip + (1.0 - strip) * marks(cuts, 1.0).view(1, -1, 1)
    red = torch.tensor((1.0, 0.25, 0.3), device=dev, dtype=dt)
    tick = (marks(hits, 1.5).view(1, -1) * (rows < strip_h * 0.3)).unsqueeze(-1)
    strip = strip + (red - strip) * tick

    region = out[:, height - strip_h:]
    region.mul_(0.2).add_(strip, alpha=0.8)
    playhead = (torch.arange(offset, offset + n, device=dev, dtype=dt) * (width - 1) / max(total - 1, 1)).view(-1, 1)
    head = _soft(1.5 - (xs.view(1, -1) - playhead).abs()).view(n, 1, width, 1)
    region.add_((torch.tensor((1.0, 0.84, 0.2), device=dev, dtype=dt) - region) * head)
    return out


def slice_audio(waveform, sample_rate, start, seconds):
    """`seconds` of `waveform` [B, C, L] from `start`, padded with silence outside the track."""
    a = round(start * sample_rate)
    n = round(seconds * sample_rate)
    out = waveform.new_zeros(waveform.shape[:-1] + (n,))
    lo, hi = max(a, 0), min(a + n, waveform.shape[-1])
    if hi > lo:
        out[..., lo - a:hi - a] = waveform[..., lo:hi]
    return out


def downbeat_at(grid, t):
    """First downbeat at or after `t` seconds."""
    bar = 240.0 / grid["bpm"]
    d0 = grid["downbeat"]
    return d0 + max(0, math.ceil((t - d0) / bar - 1e-9)) * bar


def _envelopes(mono, sample_rate):
    """Onset strength (positive spectral flux, lightly compressed so hats don't outvote kicks and snares)
    at ENV_RATE: the beat envelope (kick + low-mid band), the kick band and the snare band."""
    hop = sample_rate / ENV_RATE
    n_fft = 1 << math.ceil(math.log2(sample_rate * 0.046))
    spec = torch.stft(mono, n_fft, hop_length=round(hop), window=torch.hann_window(n_fft), return_complex=True).abs()
    flux = torch.relu(torch.diff(torch.log1p(10 * spec), dim=1))
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)

    def band(lo, hi):
        b = flux[(freqs >= lo) & (freqs < hi)].sum(0)
        return b / (b.max() + 1e-9)

    kick = band(30, 150)
    return kick + band(150, 1500), kick, band(1500, 8000), round(hop) / sample_rate


def _at(env, pos):
    """Linear interpolation of `env` at fractional frame positions."""
    pos = pos.clamp(0, env.numel() - 1.001)
    i = pos.long()
    w = pos - i
    return env[i] * (1 - w) + env[i + 1] * w


def analyse(mono, sample_rate):
    """Tempo and first downbeat of a (drums) track: {"bpm", "downbeat"} in seconds."""
    env, kick, snare, dt = _envelopes(mono.float(), sample_rate)
    if env.numel() < 4 * ENV_RATE or env.max() <= 0:
        return {"bpm": 120.0, "downbeat": 0.0}
    env = env / env.max()

    # tempo: autocorrelation scored on beat, half bar and bar, inside one octave
    x = env - env.mean()
    spec = torch.fft.rfft(x, 2 * x.numel())
    ac = torch.fft.irfft(spec * spec.conj())[: x.numel()]
    ac = ac / ac[0]
    bpms = torch.arange(TEMPO_RANGE[0], TEMPO_RANGE[1], 0.05)
    period = 60.0 / bpms / dt
    score = _at(ac, period) + _at(ac, 2 * period) + _at(ac, 4 * period)
    period = float(period[score.argmax()])

    # period and phase: the beat grid within 3% of that tempo that lands on the most onset energy
    # over the whole track (a small tempo error would drift a full beat over a song)
    best = (-1.0, period, 0.0)
    for p in (period * (1 + torch.linspace(-0.03, 0.03, 601))).tolist():
        phases = torch.arange(0, p, 0.5)
        k = torch.arange(int((env.numel() - 1) / p))
        comb = _at(env, phases[:, None] + k[None, :] * p).sum(1)
        i = int(comb.argmax())
        if float(comb[i]) > best[0]:
            best = (float(comb[i]), p, float(phases[i]))
    _, period, phase = best

    # fit the grid to the onset peaks over growing spans, re-predicting each time so it never drifts off
    radius = max(1, int(period / 8))
    for span in (30, 60, None):
        last = env.numel() - 1 if span is None else min(env.numel() - 1, phase + span * ENV_RATE)
        idx, pos, weight = [], [], []
        for b in range(int((last - phase) / period) + 1):
            c = round(phase + b * period)
            lo, hi = max(c - radius, 0), min(c + radius + 1, env.numel())
            j = lo + int(env[lo:hi].argmax())
            idx.append(b)
            pos.append(j)
            weight.append(float(env[j]))
        if len(idx) < 8:
            break
        b, p, w = (torch.tensor(v, dtype=torch.float64) for v in (idx, pos, weight))
        sol = torch.linalg.lstsq(torch.stack([b, torch.ones_like(b)], 1) * w.sqrt()[:, None], p * w.sqrt()).solution
        period, phase = float(sol[0]), float(sol[1])

    # downbeat: on a half-beat grid (so an off-beat phase is caught too), the position whose bar has
    # the kick on 1 and the snare on 2 and 4
    half = period / 2
    phase %= half
    grid = phase + torch.arange(int((env.numel() - 1 - phase) / half) + 1) * half
    n = (grid.numel() // 8) * 8
    if n < 16:
        return {"bpm": 60.0 / (period * dt), "downbeat": phase * dt}
    K = (_at(kick, grid[:n]) / (kick.max() + 1e-9)).view(-1, 8).mean(0)
    S = (_at(snare, grid[:n]) / (snare.max() + 1e-9)).view(-1, 8).mean(0)
    scores = [K[o] - K[(o + 4) % 8] + S[(o + 2) % 8] + S[(o + 6) % 8] - S[o] - S[(o + 4) % 8] for o in range(8)]
    first = max(range(8), key=lambda o: float(scores[o]))
    return {"bpm": 60.0 / (period * dt), "downbeat": float(grid[first]) * dt}
