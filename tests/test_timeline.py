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
    shots = timeline.shots_from_cuts(timeline.bar_cuts(1.785, 85.0, 8, 4))
    bar = 4 * 60 / 85
    for i, s in enumerate(shots):
        assert s["start"] == pytest.approx(1.785 + i * 2 * bar)
        assert s["length"] == timeline.h3_frames(s["frames"])
    # frame counts follow the rounded cut points, so the joined video stays on the track
    assert sum(s["frames"] for s in shots) == round((1.785 + 8 * bar) * 24) - round(1.785 * 24)


def test_bars_are_shared_out_longest_shots_first():
    bar = 2.0   # 120 BPM
    cuts = timeline.bar_cuts(1.0, 120.0, 10, 4)
    assert [round((b - a) / bar) for a, b in zip(cuts, cuts[1:])] == [3, 3, 2, 2]
    assert cuts[-1] == pytest.approx(21.0)


def kick_times(bpm, bars, offset):
    """When drum_loop(four_on_floor=False) hits the kick: beat 1 and the 'and' of 3."""
    beat = 60.0 / bpm
    return [offset + b * 4 * beat + d * beat for b in range(bars) for d in (0, 2.5)]


def test_find_kicks_takes_the_kicks_not_the_snare_or_hats():
    x = drum_loop(110, bars=8, offset=0.4, sixteenths=True)
    kicks = kick_times(110, 8, 0.4)
    found = timeline.find_kicks(x, SR)
    assert len(found) == len(kicks)
    assert all(abs(f - k) < 0.01 for f, k in zip(found, kicks))       # on the hit, not a frame early


def test_the_grid_moves_onto_the_kicks():
    grid = {"bpm": 120.0, "downbeat": 0.75}                            # half a beat late
    kicks = [0.5 + 0.5 * i for i in range(16)]
    assert timeline.on_the_kicks(grid, kicks)["downbeat"] == pytest.approx(0.5)
    assert timeline.on_the_kicks({"bpm": 120.0, "downbeat": 0.5}, kicks)["downbeat"] == pytest.approx(0.5)


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
    out = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=4.0, duration=0.0, bpm=0.0).result
    image, guide_audio, length, frames, soundtrack, info, strip = out
    assert [float(i[0, 0, 0, 0]) for i in image] == pytest.approx([0.0, 0.1, 0.2])
    assert frames == [144, 144, 144]                      # 3 bars of 2 s at 24 fps
    assert length == [158, 158, 158]
    assert all(g["waveform"].shape[-1] == round(158 / 24 * SR) for g in guide_audio)
    assert soundtrack["waveform"].shape[-1] == round(18.0 * SR)
    assert "120.00 BPM" in info


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
        track=track, images=images, start_at=0.5, duration=0.0, bpm=0.0, mode="dialogue").result
    cuts = [0.5]
    for f in frames:
        cuts.append(cuts[-1] + f / 24)
    assert 4.0 < cuts[1] < 4.8 and 9.1 < cuts[2] < 9.9
    assert "pauses" in info


def one_shot(mode, track, duration, start_at=0.5):
    out = nodes.SyntaxTimeline.execute(track=track, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=start_at, duration=duration,
                                       bpm=0.0, mode=mode).result
    return out[3], out[5]


def test_duration_sets_the_length_of_one_shot_in_each_mode():
    voice = {"waveform": speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0).view(1, 1, -1), "sample_rate": SR}
    frames, _ = one_shot("dialogue", voice, 4.0)
    assert 4.0 < 0.5 + frames[0] / 24 < 4.8                 # the end moves into the pause after the first line
    drums = {"waveform": drum_loop(120, bars=8, offset=0.5).view(1, 1, -1), "sample_rate": SR}
    frames, info = one_shot("music", drums, 6.2, start_at=0.0)
    assert frames == [144] and "3 bars" in info             # whole 2 s bars closest to 6.2 s


def test_one_image_can_run_thirty_seconds():
    drums = {"waveform": drum_loop(120, bars=20, offset=0.5).view(1, 1, -1), "sample_rate": SR}
    frames, info = one_shot("music", drums, 30.0, start_at=0.0)
    assert frames == [720] and "Context Windows" in info     # 15 bars, past H3's 15 s


def test_duration_zero_runs_to_the_end_of_the_sound():
    voice = {"waveform": speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0).view(1, 1, -1), "sample_rate": SR}
    frames, _ = one_shot("dialogue", voice, 0.0)
    assert 0.5 + frames[0] / 24 == pytest.approx(14.0, abs=0.05)


def test_music_is_guided_by_just_the_kicks():
    x = drum_loop(120, bars=8, offset=0.5, sixteenths=True)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    out = nodes.SyntaxTimeline.execute(track=track, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=0.0,
                                       duration=8.0, bpm=120.0, mode="music", edits='{"hits": [1.25]}')
    guide = out.result[1][0]["waveform"][0, 0]
    start = out.ui["timeline"][0]["cuts"][0]
    energy = guide[: 150 * (SR // 20)].square().view(150, -1).sum(1)   # 50 ms bins over 7.5 s
    loud = {int(i) for i in torch.nonzero(energy > energy.max() * 0.05).flatten()}
    on_kicks = {int((k - start) * 20) + d for k in kick_times(120, 8, 0.5) for d in (0, 1, 2, 3)}
    assert loud and loud <= on_kicks                                  # sound only at the kicks: no snare, hats or drawn hit
    assert "kicks" in out.result[5] and "connect the drums stem" in out.result[5]


def test_music_finds_the_kicks_in_the_drums_input_when_given():
    drums = {"waveform": drum_loop(120, bars=8, offset=0.5).view(1, 1, -1), "sample_rate": SR}
    hats = torch.sin(2 * math.pi * 9000 * torch.arange(drums["waveform"].shape[-1]) / SR).view(1, 1, -1)
    track = {"waveform": hats, "sample_rate": SR}
    out = nodes.SyntaxTimeline.execute(track=track, drums=drums, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=0.0,
                                       duration=4.0, bpm=0.0, mode="music")
    ui = out.ui["timeline"][0]
    assert ui["bpm"] == pytest.approx(120, rel=0.01)                 # the grid comes from the drums, not the track
    assert ui["kicks"] == pytest.approx(kick_times(120, 8, 0.5), abs=0.01)


def test_music_is_guided_by_the_drums_stem_by_default():
    drums = {"waveform": drum_loop(120, bars=8, offset=0.5, sixteenths=True).view(1, 1, -1), "sample_rate": SR}
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    out = nodes.SyntaxTimeline.execute(track=drums, drums=drums, images=images, start_at=0.0, duration=4.0, bpm=120.0, mode="music")
    start = out.ui["timeline"][0]["cuts"][0]
    guide = out.result[1][0]["waveform"][0, 0]
    stem = timeline.slice_audio(drums["waveform"], SR, start, guide.numel() / SR)[0, 0]
    assert torch.allclose(guide, stem) and "guided by the drums" in out.result[5]   # every hit of the kit, hats and snare too
    kicks = nodes.SyntaxTimeline.execute(track=drums, drums=drums, images=images, start_at=0.0, duration=4.0, bpm=120.0, mode="music",
                                         music_guide="kicks")
    assert "guided by the kicks" in kicks.result[5] and not torch.allclose(kicks.result[1][0]["waveform"][0, 0], guide)


def test_a_shot_with_no_kick_is_guided_by_the_low_hits_of_the_track():
    loop = drum_loop(120, bars=4, offset=0.5, four_on_floor=True)
    n = int(0.5 * SR) + 8 * 2 * SR                                    # 8 bars at 120 BPM after 0.5 s
    drums = torch.zeros(n)
    drums[: loop.numel()] = loop[:n]                                  # kicks for 4 bars, then a breakdown
    track = drums.clone()
    t = torch.arange(int(0.6 * SR)) / SR
    boom = torch.sin(2 * math.pi * 45 * t) * torch.exp(-t / 0.25)    # 808s in the breakdown, only in the track
    for at in (9.0, 10.5, 12.0, 13.5, 15.0):
        i = int(at * SR)
        track[i:i + boom.numel()] += boom
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    for guide in ("drums", "kicks"):
        out = nodes.SyntaxTimeline.execute(track={"waveform": track.view(1, 1, -1), "sample_rate": SR},
                                           drums={"waveform": drums.view(1, 1, -1), "sample_rate": SR}, images=images,
                                           start_at=0.0, duration=16.0, bpm=120.0, mode="music", music_guide=guide)
        assert "low hits in shot 2" in out.result[5]
        second = out.result[1][1]["waveform"][0, 0]
        assert second.abs().max() > 0.1                                # not silent: H3 gets the 808s


def test_edited_cuts_override_the_automatic_ones():
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0, mode="dialogue",
                                       edits='{"cuts": [1.0, 3.0, 6.0, 12.0]}')
    frames = out.result[3]
    assert frames == [48, 72, 144]
    assert out.ui["timeline"][0]["cuts"] == pytest.approx([1.0, 3.0, 6.0, 12.0])


def test_edited_shots_need_not_match_the_images_and_reuse_them_in_order():
    x = speech([(0.5, 4.0), (4.8, 9.1), (9.9, 14.0)], 15.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.full((1, 8, 8, 3), i / 10) for i in range(2)}
    one = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0, mode="action",
                                       edits='{"cuts": [0.5, 14.0]}').result
    assert one[3] == [324] and len(one[0]) == 1
    five = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0, mode="action",
                                        edits='{"cuts": [0, 2, 4, 6, 8, 10]}').result
    assert [float(i[0, 0, 0, 0]) for i in five[0]] == pytest.approx([0.0, 0.1, 0.0, 0.1, 0.0])


def test_timeline_sends_the_beat_grid_to_the_editor():
    x = drum_loop(120, bars=16, offset=0.5)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    ui = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0).ui["timeline"][0]
    assert ui["bpm"] == pytest.approx(120, rel=0.01)
    assert ui["duration"] == pytest.approx(x.numel() / SR)
    assert len(ui["cuts"]) == 3 and ui["mode"] == "music" and ui["images"] == 2


def test_hit_sounds_land_only_at_the_marked_times():
    sr = 32000
    x = timeline.drawn_audio([(1.0, "thump"), (2.5, "thump")], [], 4.0, sr)
    assert x.numel() == 4 * sr
    energy = x.square().view(-1, sr // 10).sum(1)          # 0.1 s bins
    loud = set(torch.nonzero(energy > energy.max() * 0.05).flatten().tolist())
    assert loud and loud <= {10, 11, 25, 26}


def test_drawn_hits_without_a_track_drive_the_shots():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(3)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, duration=12.0, bpm=0.0, mode="action",
                                       hit_sound="thump", edits='{"hits": [1.5, 5.0, 9.0]}')
    image, guide_audio, length, frames, soundtrack, info, strip = out.result
    cuts = [0.0]
    for f in frames:
        cuts.append(cuts[-1] + f / 24)
    assert cuts[-1] == pytest.approx(12.0, abs=0.05)            # the whole timeline
    assert not any(abs(c - h) < 0.3 for c in cuts[1:-1] for h in (1.5, 5.0, 9.0))   # no cut on a hit
    assert soundtrack["waveform"].abs().max() > 0                # the hits are the soundtrack
    assert strip[0]["hits"] == [36, 120, 216]


def test_drawn_hits_are_mixed_into_a_track_guide():
    x = speech([(0.5, 3.0)], 6.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    plain = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0, mode="action",
                                         edits='{"cuts": [0.0, 6.0]}').result[1][0]
    mixed = nodes.SyntaxTimeline.execute(track=track, images=images, start_at=0.0, duration=0.0, bpm=0.0, mode="action",
                                         edits='{"cuts": [0.0, 6.0], "hits": [4.0]}').result[1][0]
    i = int(4.0 * SR)
    assert plain["waveform"][..., i:i + 2000].abs().max() == 0
    assert mixed["waveform"][..., i:i + 2000].abs().max() > 0


def test_timeline_gives_each_shot_its_slice_of_the_strip():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, duration=8.0, bpm=0.0, mode="action",
                                       edits='{"hits": [2.0, 6.0], "cuts": [0.0, 4.0, 8.0]}').result
    frames, strips = out[3], out[6]
    assert len(strips) == 2
    assert [s["offset"] for s in strips] == [0, frames[0]]
    for s in strips:
        assert len(s["level"]) == sum(frames) == 192             # the whole video, so the playhead runs across cuts
        assert s["cuts"] == [frames[0]] and s["hits"] == [48, 144]


def _yellow_x(frame):
    px = frame[-4:]
    yellow = (px[..., 0] > 0.75) & (px[..., 1] > 0.65) & (px[..., 2] < 0.45)
    return yellow.any(0).nonzero().float().mean()


def test_preview_strip_draws_at_the_bottom_with_a_playhead_across_the_whole_video():
    level = [i / 19 for i in range(20)]
    first = torch.full((10, 64, 96, 3), 0.5)
    second = torch.full((10, 64, 96, 3), 0.5)
    a = nodes.SyntaxPreviewStrip.execute(first, {"level": level, "cuts": [10], "hits": [5], "offset": 0}).result[0]
    b = nodes.SyntaxPreviewStrip.execute(second, {"level": level, "cuts": [10], "hits": [5], "offset": 10}).result[0]
    assert torch.equal(a[:, :50], first[:, :50])              # picture untouched above the strip
    assert not torch.equal(a[:, -6:], first[:, -6:])
    assert _yellow_x(a[0]) < 10 and 40 < _yellow_x(a[-1]) < 50   # first shot: left half
    assert 45 < _yellow_x(b[0]) < 55 and _yellow_x(b[-1]) > 85   # second shot carries on to the right edge


@pytest.mark.parametrize("sound", timeline.CURVE_SOUNDS)
def test_a_curve_sounds_only_where_it_is_drawn_and_louder_where_it_is_higher(sound):
    sr = 16000
    x = timeline.drawn_audio([], [{"sound": sound, "points": [[1.0, 0.2], [2.0, 0.2], [2.01, 1.0], [3.0, 1.0]]}], 4.0, sr)
    level = x.square().view(-1, sr // 4).mean(1).sqrt()                 # 0.25 s bins
    assert level[:4].max() == 0 and level[13:].max() == 0             # silent before and after the curve
    assert level[9:12].mean() > 1.5 * level[4:8].mean()               # the high part is louder


def test_each_hit_keeps_its_own_sound():
    images = {"image0": torch.zeros(1, 8, 8, 3)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, duration=4.0, bpm=0.0, mode="action",
                                       hit_sound="thump", edits='{"hits": [1.0, [2.0, "boom"]]}').result
    x = out[4]["waveform"][0, 0]
    sr = out[4]["sample_rate"]
    assert x[int(1.3 * sr):int(1.9 * sr)].abs().max() < 0.01          # the thump is short
    assert x[int(2.5 * sr):int(2.9 * sr)].abs().max() > 0.01           # the boom rings on


def test_curves_drive_a_blank_action_timeline():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=0.0, duration=8.0, bpm=0.0, mode="action",
                                       edits='{"curves": [{"sound": "motor", "points": [[1, 0], [2, 1], [3, 0]]}, {"sound": "flow", "points": [[5, 0.5], [7, 0.5]]}]}')
    frames = out.result[3]
    assert sum(frames) == 8 * 24 and out.ui["timeline"][0]["mode"] == "action"


def test_dialogue_ignores_drawn_hits():
    x = speech([(0.5, 3.0)], 6.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    guide = nodes.SyntaxTimeline.execute(track=track, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=0.0, duration=0.0, bpm=0.0,
                                         mode="dialogue", edits='{"cuts": [0.0, 6.0], "hits": [4.0]}').result[1][0]
    assert guide["waveform"][..., int(4.0 * SR):int(4.0 * SR) + 2000].abs().max() == 0


def test_the_editor_gets_the_waveform_of_any_track():
    x = speech([(0.5, 3.0)], 6.0)
    track = {"waveform": x.view(1, 1, -1), "sample_rate": SR}
    ui = nodes.SyntaxTimeline.execute(track=track, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=0.0, duration=0.0, bpm=0.0,
                                      mode="dialogue").ui["timeline"][0]
    p = ui["peaks"]
    assert 1900 <= len(p) <= 2100 and max(p) == 1.0
    assert max(p[: len(p) // 15]) < 0.01 < min(p[len(p) // 10: len(p) // 3])   # silent, then the line


def test_a_blank_timeline_runs_duration_seconds_from_start_at():
    images = {f"image{i}": torch.zeros(1, 8, 8, 3) for i in range(2)}
    out = nodes.SyntaxTimeline.execute(track=None, images=images, start_at=13.64, duration=14.0, bpm=0.0, mode="action",
                                       edits='{"hits": [15.0, 20.0, 25.0]}')
    assert sum(out.result[3]) == pytest.approx(14.0 * 24, abs=2)
    assert out.ui["timeline"][0]["cuts"][0] == pytest.approx(13.64)


def test_an_empty_blank_timeline_still_runs_with_a_silent_guide():
    out = nodes.SyntaxTimeline.execute(track=None, images={"image0": torch.zeros(1, 8, 8, 3)}, start_at=0.0, duration=6.0, bpm=0.0,
                                       mode="action").result
    assert out[3] == [144] and out[1][0]["waveform"].abs().max() == 0
