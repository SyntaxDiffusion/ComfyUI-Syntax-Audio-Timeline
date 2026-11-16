import math
import os
import sys
from importlib import import_module

import pytest
import torch

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(COMFY_ROOT))

pack = __import__("custom_nodes.ComfyUI-SyntaxActionCurves", fromlist=["timeline"])
timeline = import_module(pack.__name__ + ".timeline")

SR = 22050


def drum_loop(bpm, bars, offset, sixteenths=False, four_on_floor=False, seed=0):
    """Mono drum loop: kick on 1 and the 'and' of 3 (or on every beat), snare on 2 and 4, optional 16th hats.
    `offset` seconds of silence before the first downbeat."""
    g = torch.Generator().manual_seed(seed)
    beat = 60.0 / bpm
    n = int((offset + bars * 4 * beat + 1.0) * SR)
    x = torch.zeros(n)

    def put(sig, at):
        i = int(round(at * SR))
        sig = sig[: n - i]
        x[i:i + sig.numel()] += sig

    t = torch.arange(int(0.25 * SR)) / SR
    kick = torch.sin(2 * math.pi * torch.cumsum(50 + 100 * torch.exp(-t / 0.03), 0) / SR) * torch.exp(-t / 0.1)
    snare = torch.randn(t.numel(), generator=g) * torch.exp(-t / 0.05) * 0.6
    hat = torch.randn(int(0.04 * SR), generator=g) * torch.exp(-torch.arange(int(0.04 * SR)) / SR / 0.01) * 0.15
    for b in range(bars * 4):
        at = offset + b * beat
        if four_on_floor or b % 4 == 0:
            put(kick, at)
        if not four_on_floor and b % 4 == 2:
            put(kick, at + beat / 2)
        if b % 2 == 1:
            put(snare, at)
        if sixteenths:
            for s in range(4):
                put(hat, at + s * beat / 4)
    return x


def test_h3_frames_snap_up_to_the_17k_plus_5_grid():
    assert [timeline.h3_frames(n) for n in (1, 5, 6, 22, 23, 136, 144)] == [5, 5, 22, 22, 39, 141, 158]


def test_auto_bars_aims_for_about_five_and_a_half_seconds():
    assert timeline.auto_bars(120) == 3   # 2.0 s bars
    assert timeline.auto_bars(85) == 2    # 2.82 s bars
    assert timeline.auto_bars(131) == 3   # 1.83 s bars
    assert timeline.auto_bars(40) == 1    # never below one bar


def test_shots_start_on_bar_lines_without_drift():
    shots = timeline.shots_from_cuts(timeline.beat_cuts(4, 85.0, 1.785, 2))
    bar = 4 * 60 / 85
    for i, s in enumerate(shots):
        assert s["start"] == pytest.approx(1.785 + i * 2 * bar)
        assert s["length"] == timeline.h3_frames(s["frames"])
    # frame counts follow the rounded cut points, so the joined video stays on the track
    assert sum(s["frames"] for s in shots) == round((1.785 + 8 * bar) * 24) - round(1.785 * 24)


def test_guide_slices_cover_the_generated_length_and_pad_past_the_end():
    sr = 1000
    audio = torch.arange(5 * sr, dtype=torch.float32).view(1, 1, -1)  # 5 s
    s = timeline.slice_audio(audio, sr, start=4.0, seconds=2.0)
    assert s.shape[-1] == 2 * sr
    assert s[0, 0, 0].item() == 4 * sr
    assert torch.all(s[..., sr:] == 0)


@pytest.mark.parametrize("bpm,sixteenths", [(100, False), (85, True), (128, False)])
def test_tempo_from_a_drum_loop(bpm, sixteenths):
    x = drum_loop(bpm, bars=16, offset=0.3, sixteenths=sixteenths)
    grid = timeline.analyse(x, SR)
    assert grid["bpm"] == pytest.approx(bpm, rel=0.01)


def test_tempo_is_exact_enough_not_to_drift_over_a_song():
    x = drum_loop(132.7, bars=60, offset=0.2)   # ~1:50
    grid = timeline.analyse(x, SR)
    assert grid["bpm"] == pytest.approx(132.7, abs=0.05)


def test_downbeat_from_kick_and_backbeat():
    x = drum_loop(100, bars=16, offset=0.9)
    grid = timeline.analyse(x, SR)
    bar = 4 * 60 / 100
    # nearest downbeat to 0.9 s modulo the bar
    d = (grid["downbeat"] - 0.9) % bar
    assert min(d, bar - d) < 0.03


def test_downbeat_with_four_on_the_floor_still_lands_on_a_bar_line_or_half_bar():
    x = drum_loop(124, bars=16, offset=0.5, four_on_floor=True)
    grid = timeline.analyse(x, SR)
    beat = 60 / 124
    d = (grid["downbeat"] - 0.5) % (2 * beat)   # kick every beat, snare on 2 and 4: only the half bar is knowable
    assert min(d, 2 * beat - d) < 0.03


def test_first_downbeat_at_or_after_a_rough_start():
    grid = {"bpm": 120.0, "downbeat": 0.5}
    assert timeline.downbeat_at(grid, 0.0) == pytest.approx(0.5)
    assert timeline.downbeat_at(grid, 3.1) == pytest.approx(4.5)
    assert timeline.downbeat_at(grid, 4.5) == pytest.approx(4.5)


nodes = import_module(pack.__name__ + ".nodes")


def test_timeline_node_cuts_one_shot_per_image_on_downbeats():
    x = drum_loop(120, bars=16, offset=0.5)
    track = {"waveform": x.view(1, 1, -1).repeat(1, 2, 1), "sample_rate": SR}
    images = {f"image{i}": torch.full((1, 8, 8, 3), i / 10) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=4.0, bpm=0.0, bars_per_shot=0).result
    image, guide_audio, length, frames, soundtrack, info, strip = out
    assert [float(i[0, 0, 0, 0]) for i in image] == pytest.approx([0.0, 0.1, 0.2])
    assert frames == [144, 144, 144]                      # 3 bars of 2 s at 24 fps
    assert length == [158, 158, 158]
    assert all(g["waveform"].shape[-1] == round(158 / 24 * SR) for g in guide_audio)
    assert soundtrack["waveform"].shape[-1] == round(18.0 * SR)
    assert "120.00 BPM" in info


def test_join_shots_trims_each_shot_and_keeps_order():
    shots = [torch.full((10, 4, 4, 3), float(i)) for i in range(3)]
    out = nodes.SyntaxJoinShots.execute(shots, [4, 6, 5]).result[0]
    assert out.shape[0] == 15
    assert out[:4].eq(0).all() and out[4:10].eq(1).all() and out[10:].eq(2).all()


def speech(segments, total, sr=SR, seed=1):
    """Noise bursts at the (start, end) seconds in `segments`, silence elsewhere."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(int(total * sr))
    for a, b in segments:
        n = int((b - a) * sr)
        x[int(a * sr):int(a * sr) + n] = torch.randn(n, generator=g) * 0.3
    return x


def test_pause_cuts_land_in_the_gaps_between_lines():
    # three lines with gaps at 4.0-4.8 s and 9.1-9.9 s; an even split of 0.5-14 s would cut at 5.0 and 9.5
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    bounds = timeline.pause_cuts(x, SR, start=0.5, n_shots=3)
    assert bounds[0] == pytest.approx(0.5)
    assert bounds[1] == pytest.approx(4.4, abs=0.15)    # the middle of the gap, not the end of the line
    assert bounds[2] == pytest.approx(9.5, abs=0.15)
    assert bounds[3] == pytest.approx(14.0, abs=0.05)   # ends where the sound ends


def test_even_cuts_split_the_sound_equally():
    x = speech([(1.0, 13.0)], 15.0)
    bounds = timeline.even_cuts(x, SR, start=1.0, n_shots=4)
    assert bounds == pytest.approx([1.0, 4.0, 7.0, 10.0, 13.0], abs=0.05)


def test_shots_from_cut_points_keep_the_frame_total():
    shots = timeline.shots_from_cuts([0.5, 4.37, 9.52, 14.0])
    assert [s["start"] for s in shots] == pytest.approx([0.5, 4.37, 9.52])
    assert sum(s["frames"] for s in shots) == round(14.0 * 24) - round(0.5 * 24)
    assert all(s["length"] == timeline.h3_frames(s["frames"]) for s in shots)


def test_timeline_node_cuts_dialogue_in_the_pauses():
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    image, guide_audio, length, frames, soundtrack, info, strip = nodes.SyntaxTimeline.execute(
        track=track, images=images, start_at=0.5, bpm=0.0, bars_per_shot=0, cut_on="pauses").result
    cuts = [0.5]
    for f in frames:
        cuts.append(cuts[-1] + f / 24)
    assert 4.0 < cuts[1] < 4.8 and 9.1 < cuts[2] < 9.9
    assert "pauses" in info


def test_edited_cuts_override_the_automatic_ones():
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="pauses",
                                       edits='{"cuts": [1.0, 3.0, 6.0, 12.0]}')
    frames = out.result[3]
    assert frames == [48, 72, 144]
    assert out.ui["timeline"][0]["cuts"] == pytest.approx([1.0, 3.0, 6.0, 12.0])


def test_edits_for_a_different_number_of_images_are_ignored():
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.5, bpm=0.0, bars_per_shot=0, cut_on="pauses",
                                       edits='{"cuts": [1.0, 3.0, 12.0]}')
    assert len(out.result[3]) == 3
    assert "ignored" in out.result[5]


def test_timeline_sends_the_beat_grid_to_the_editor():
    x = drum_loop(120, bars=16, offset=0.5)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    ui = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0).ui["timeline"][0]
    assert ui["bpm"] == pytest.approx(120, rel=0.01)
    assert ui["duration"] == pytest.approx(x.numel() / SR)
    assert len(ui["cuts"]) == 3 and ui["mode"] == "beats"


def test_hit_sounds_land_only_at_the_marked_times():
    sr = 32000
    x = timeline.hits_audio([1.0, 2.5], 4.0, sr, "thump")
    assert x.numel() == 4 * sr
    energy = x.square().view(-1, sr // 10).sum(1)          # 0.1 s bins
    loud = set(torch.nonzero(energy > energy.max() * 0.05).flatten().tolist())
    assert loud and loud <= {10, 11, 25, 26}


def test_drawn_hits_without_a_track_drive_the_shots():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="pauses",
                                       duration=12.0, hit_sound="thump", edits='{"hits": [1.5, 5.0, 9.0]}')
    image, guide_audio, length, frames, soundtrack, info, strip = out.result
    cuts = [0.0]
    for f in frames:
        cuts.append(cuts[-1] + f / 24)
    assert cuts[-1] == pytest.approx(12.0, abs=0.05)            # the whole timeline
    assert not any(abs(c - h) < 0.3 for c in cuts[1:-1] for h in (1.5, 5.0, 9.0))   # no cut on a hit
    assert soundtrack["waveform"].abs().max() > 0                # the hits are the soundtrack
    assert out.ui["timeline"][0]["hits"] == [1.5, 5.0, 9.0]


def test_drawn_hits_are_mixed_into_a_track_guide():
    x = speech([(0.5, 3.0)], 6.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    plain = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="even",
                                         edits='{"cuts": [0.0, 6.0]}').result[1][0]
    mixed = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="even",
                                         edits='{"cuts": [0.0, 6.0], "hits": [4.0]}').result[1][0]
    i = int(4.0 * SR)
    assert plain["waveform"][..., i:i + 2000].abs().max() == 0
    assert mixed["waveform"][..., i:i + 2000].abs().max() > 0


def test_no_track_and_no_hits_says_what_to_do():
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    with pytest.raises(ValueError, match="track"):
        nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="even", duration=8.0)


def test_preview_strip_data_matches_the_cut():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="even",
                                       duration=8.0, preview_strip=True, edits='{"hits": [2.0, 6.0]}').result
    frames, strip = out[3], out[6]
    assert len(strip["level"]) == sum(frames) == 192
    assert strip["cuts"] == [frames[0]]
    assert strip["hits"] == [48, 144]
    assert out[6] is not None and nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, bpm=0.0, bars_per_shot=0, cut_on="even",
                                                              duration=8.0, edits='{"hits": [2.0]}').result[6] is None


def test_join_shots_draws_the_strip_at_the_bottom_with_a_moving_playhead():
    shots = [torch.full((10, 64, 96, 3), 0.5) for _ in range(2)]
    strip = {"level": [i / 19 for i in range(20)], "cuts": [10], "hits": [5]}
    plain = nodes.SyntaxJoinShots.execute(shots, [10, 10]).result[0]
    drawn = nodes.SyntaxJoinShots.execute(shots, [10, 10], [strip]).result[0]
    assert torch.equal(drawn[:, :50], plain[:, :50])          # picture untouched above the strip
    assert not torch.equal(drawn[:, -6:], plain[:, -6:])
    px = drawn[:, -4:]
    yellow = (px[..., 0] > 0.75) & (px[..., 1] > 0.65) & (px[..., 2] < 0.45)   # the yellow playhead
    first, last = yellow[0].any(0).nonzero().float().mean(), yellow[-1].any(0).nonzero().float().mean()
    assert first < 20 and last > 75
