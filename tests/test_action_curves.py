import os
import sys

import pytest
import torch

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(COMFY_ROOT))

import comfy.nested_tensor  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch  # noqa: E402

pack = __import__("custom_nodes.ComfyUI-SyntaxActionCurves", fromlist=["nodes"])
from importlib import import_module  # noqa: E402

curves = import_module(pack.__name__ + ".curves")
apply = import_module(pack.__name__ + ".apply")
render = import_module(pack.__name__ + ".render")
sound = import_module(pack.__name__ + ".sound")
nodes = import_module(pack.__name__ + ".nodes")


@pytest.mark.parametrize("interp", curves.INTERPOLATIONS)
def test_evaluate_hits_keys_and_stays_in_range(interp):
    pts = curves.parse_points(curves.DEFAULT_CURVE)
    vals = curves.evaluate(pts, interp, 9)  # frames land exactly on the 0/.25/.5/.75/1 keys
    assert len(vals) == 9
    assert all(0.0 <= v <= 1.0 for v in vals)
    for f, (_, y) in zip([0, 2, 4, 6, 8], pts):
        assert vals[f] == pytest.approx(y)
    dense = curves.evaluate(pts, interp, 500)
    assert max(dense) <= 0.9 + 1e-9 and min(dense) >= 0.1 - 1e-9  # monotone cubic never overshoots


def test_resample_endpoints():
    assert curves.resample([0.0, 1.0], 5) == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert curves.resample([0.2, 0.4, 0.9], 2) == pytest.approx([0.2, 0.9])


def test_h3_av_weights_follow_pack_layout():
    video = torch.zeros(1, 24, 7, 4, 6)
    audio = torch.zeros(1, 32, 2, 20)
    shapes = [video.shape, audio.shape]
    curve = [i / 9 for i in range(10)]
    w = curves.latent_time_weights(shapes, curve, "cpu")
    v, a = comfy.utils.unpack_latents(w, shapes)
    assert v.shape == (1, 24, 7, 4, 6) and a.shape == (1, 32, 2, 20)
    assert torch.allclose(v[0, 3, :, 1, 2], torch.tensor(curves.to_latent_frames(curve, 7)))
    assert torch.allclose(a[0, 5, 1, :], torch.tensor(curves.resample(curve, 20)))  # H3 audio time is the last axis


def test_single_frame_spike_survives():
    pts = curves.parse_points('{"points": [[0, 0], [0.368, 0], [0.37, 1], [0.372, 0], [1, 0]]}')
    for interp in curves.INTERPOLATIONS:
        vals = curves.evaluate(pts, interp, 124)
        assert max(vals) == 1.0 and sum(v > 0.5 for v in vals) == 1
    for offset in range(124):  # wherever the spike frame falls, exactly one H3 latent frame gets it at full strength
        frame = [0.0] * 124
        frame[offset] = 1.0
        lat = curves.to_latent_frames(frame, 37)
        assert sorted(lat)[-2:] == [0.0, 1.0]


def test_ltx_audio_time_axis():
    w = curves.latent_time_weights([(1, 8, 30, 16)], [0.0, 1.0], "cpu")
    assert torch.allclose(w[0, 0, :, 3], torch.linspace(0, 1, 30))


def _run_masks(conds, x):
    conds = [{**c[1], "model_conds": {}, "uuid": None} for c in conds]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(conds, x.shape[2:], "cpu")
    total = 0
    for c in conds:
        area, mult = comfy.samplers.get_area_and_mult(c, x, torch.tensor([500.0]))[:2]
        assert mult.shape == x.shape
        total = total + mult
    return total, conds


def test_blend_masks_sum_to_one_on_packed_and_plain_latents():
    curve = [0.0, 1.0, 0.0, 1.0]
    pos = [[torch.zeros(1, 4, 8), {"pooled_output": None, "guide": "base"}]]
    act = [[torch.ones(1, 6, 8), {"pooled_output": None}]]

    nested = comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 6, 4, 4), torch.zeros(1, 32, 2, 12)))
    packed, shapes = comfy.utils.pack_latents(nested.unbind())
    out = apply.blend_conditioning(pos, act, shapes, curve, 1.0)
    assert out[1][1]["guide"] == "base"  # action inherits base extras
    total, _ = _run_masks(out, packed)
    assert torch.allclose(total, torch.ones_like(total))

    plain = torch.zeros(1, 128, 9, 6, 10)
    out = nodes.SyntaxActionCurveConditioning.execute(pos, act, {"samples": plain}, curve, 0.7, True).result[0]
    total, conds = _run_masks(out, plain)
    assert torch.allclose(total, torch.ones_like(total))
    action_w = conds[1]["mask"][0, :, 0, 0]
    assert action_w.max() == pytest.approx(0.7) and action_w.min() == pytest.approx(0.0)


class _FakeModel:
    latent_shapes = None


def test_curve_cfg_matches_plain_cfg_at_full_curve():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 200)
    shapes = [(1, 4, 5, 2, 2), (1, 4, 2, 15)]
    model = _FakeModel()
    model.latent_shapes = shapes
    cond, uncond = torch.randn_like(x), torch.randn_like(x)
    args = {"cond": cond, "uncond": uncond, "cond_scale": 5.0, "model": model}
    assert torch.allclose(apply.CurveCFG([1.0] * 8, 1.0)(args), uncond + (cond - uncond) * 5.0, atol=1e-6)
    assert torch.allclose(apply.CurveCFG([0.0] * 8, 1.0)(args), cond, atol=1e-6)
    assert torch.allclose(apply.CurveCFG([0.0] * 8, 0.0)(args), uncond + (cond - uncond) * 5.0, atol=1e-6)


class _FakeControlNet:
    injection_layers = (0, 2)

    def init_stream(self, h, control_latent, layout, t_emb):
        return h.clone()

    def step(self, index, c, t_emb, mod_segments, rope_freqs, transformer_options):
        return c, c * 0 + (index + 1)


class _FakePatch:
    model = _FakeControlNet()


def _run_patch(patch, layout, hidden=3):
    h = torch.zeros(layout.seq_len, hidden)
    patch.active = True
    patch.control_latent = torch.zeros(1)
    outs = []
    for block in (0, 2):
        args = {"img": h, "layout": layout, "t_emb": None, "mod_segments": None, "rope_freqs": None, "transformer_options": {}}
        patch.before_block(block, args)
        outs.append(patch.after_block(block, args, {"img": torch.zeros_like(h)})["img"])
    return outs


def test_fun_control_curve_weights_video_rows():
    layout = PackedLayout(text_len=5, latent_t=4, latent_h=4, latent_w=4, audio_t=3)
    curve = [0.0, 1.0, 0.0, 0.5]
    ours = _run_patch(apply.CurveFunControlPatch(_FakePatch(), None, None, 2.0, 1.0, 0.0, curve), layout)
    native = _run_patch(MiniMaxH3FunControlPatch(_FakePatch(), None, None, None, None, 2.0, 1.0, 0.0), layout)
    va, vb, _ = next(s for s in layout.segments if s[2] == "video")
    rows_per_frame = (vb - va) // 4
    for o, n in zip(ours, native):
        assert torch.equal(o[:va], n[:va])  # text/audio rows untouched
        for t, w in enumerate(curve):
            frame = slice(va + t * rows_per_frame, va + (t + 1) * rows_per_frame)
            assert torch.allclose(o[frame], n[frame] * w)


@pytest.mark.parametrize("style", render.STYLES)
def test_control_video(style):
    frames = render.control_video([0.0, 0.5, 1.0], 64, 32, style)
    assert frames.shape == (3, 32, 64, 3)
    assert frames.min() >= 0 and frames.max() <= 1
    assert not torch.equal(frames[0], frames[2])


def test_overlay_and_mixer():
    images = torch.rand(5, 40, 60, 3)
    out = render.overlay(images, [0.0, 1.0, 0.0], 0.25)
    assert out.shape == images.shape
    assert torch.equal(out[:, :30], images[:, :30])
    mixed = nodes.SyntaxCurveMixer.execute("subtract", {"curve_1": [1.0, 1.0, 1.0], "curve_0": [0.2, 0.9], "curve_2": None}).result[0]
    assert mixed == pytest.approx([0.0, 0.0, 0.0])  # curve_0 - curve_1, clamped
    mixed = nodes.SyntaxCurveMixer.execute("average", {"curve_0": [0.0, 1.0], "curve_1": [1.0, 1.0]}).result[0]
    assert mixed == pytest.approx([0.5, 1.0])


def test_blend_strength_above_one_still_sums_to_one():
    pos = [[torch.zeros(1, 4, 8), {}]]
    act = [[torch.ones(1, 4, 8), {}]]
    plain = torch.zeros(1, 16, 5, 4, 4)
    out = nodes.SyntaxActionCurveConditioning.execute(pos, act, {"samples": plain}, [0.0, 1.0, 0.0], 2.5, False).result[0]
    total, conds = _run_masks(out, plain)
    assert torch.allclose(total, torch.ones_like(total))
    assert conds[0]["mask"].min() == pytest.approx(-1.5)


def test_chained_blends_add_lanes():
    pos = [[torch.zeros(1, 4, 8), {"guide": "base"}]]
    drift = [[torch.ones(1, 4, 8), {}]]
    boom = [[torch.full((1, 4, 8), 2.0), {}]]
    plain = torch.zeros(1, 16, 4, 4, 4)
    latent = {"samples": plain}
    lane1 = nodes.SyntaxActionCurveConditioning.execute(pos, drift, latent, [1.0, 1.0, 0.0, 0.0], 1.0, False).result[0]
    lane2 = nodes.SyntaxActionCurveConditioning.execute(lane1, boom, latent, [0.0, 0.0, 1.0, 1.0], 1.0, False).result[0]
    assert len(lane2) == 3 and lane2[2][1]["guide"] == "base"
    total, conds = _run_masks(lane2, plain)
    assert torch.allclose(total, torch.ones_like(total))
    assert torch.allclose(conds[1]["mask"][0, :, 0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))  # drift survives where boom is 0
    assert torch.allclose(conds[2]["mask"][0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_rises():
    assert sound.rises([0.0, 0.2, 0.9, 1.0, 0.1, 0.6]) == [(2, 4), (5, 6)]
    assert sound.rises([1.0, 1.0]) == [(0, 2)]
    assert sound.rises([0.0, 0.4]) == []


def _rms(x, start, end, fps=24):
    sr = sound.SAMPLE_RATE
    return x[round(start / fps * sr):round(end / fps * sr)].square().mean().sqrt().item()


@pytest.mark.parametrize("name", sound.SOUNDS)
def test_guide_audio_follows_curve(name):
    curve = [0.0] * 24 + [1.0] * 12 + [0.0] * 12
    x = sound.guide_audio(curve, 24, name)
    assert x.numel() == 2 * sound.SAMPLE_RATE
    assert x.abs().max().item() == pytest.approx(0.95, abs=1e-4)
    assert _rms(x, 0, 23) < 1e-6  # silent before the rise
    assert _rms(x, 24, 36) > 0.05


def test_guide_audio_places_clip_on_every_rise():
    clip = torch.ones(100)
    x = sound.guide_audio([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 4, "thump", clip)
    starts = (x[1:] > 0) & (x[:-1] == 0)
    assert starts.nonzero().flatten().tolist() == [round(sound.SAMPLE_RATE / 4) - 1, round(3 * sound.SAMPLE_RATE / 4) - 1]


def test_curve_to_guide_audio_node_is_stereo():
    clip = {"waveform": torch.ones(1, 1, 4410), "sample_rate": 44100}
    out = nodes.SyntaxCurveToGuideAudio.execute([0.0, 1.0, 0.0], "thump", 24.0, clip).result[0]
    assert out["sample_rate"] == sound.SAMPLE_RATE
    assert out["waveform"].shape == (1, 2, round(3 / 24 * sound.SAMPLE_RATE))


def test_audio_to_curve():
    sr, fps = 8000, 10
    wave = torch.zeros(1, 2, sr * 2)
    wave[..., sr // 2:sr] = 0.5  # loud from frame 5 to 9
    wave[..., sr:] = 0.005  # 40 dB quieter after
    level = nodes.SyntaxAudioToCurve.execute({"waveform": wave, "sample_rate": sr}, 20, fps, "loudness", 40.0).result[0]
    assert level[:5] == pytest.approx([0.0] * 5)
    assert level[5:10] == pytest.approx([1.0] * 5)
    assert level[10:] == pytest.approx([0.0] * 10, abs=1e-6)
    onsets = sound.audio_curve(wave, sr, 20, fps, "onsets", 80.0)
    assert onsets.index(1.0) == 5 and onsets[6] == 0.0
    assert sound.audio_curve(wave, sr, 25, fps, "loudness", 40.0)[20:] == [0.0] * 5  # past the end of the audio


class _AudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.ones(1, 32, 2, waveform.shape[1] // 1280)


def test_h3_sound_guide_anchors_audio_and_releases_it():
    latent = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 9, 4, 4), torch.zeros(1, 32, 2, 50)))}
    pos = [[torch.zeros(1, 4, 8), {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": "first"}]}]]
    audio = {"waveform": torch.zeros(1, 2, 32000 * 3), "sample_rate": 32000}
    held = nodes.SyntaxH3SoundGuide.execute(pos, _AudioVAE(), latent, audio, 1.0).result[0]
    assert len(held) == 1
    keys = held[0][1]["minimax_keyframes"]
    assert keys[0]["latent"] == "first" and keys[1]["audio_latent"].shape[-1] == 50  # cropped to the video's audio
    assert len(pos[0][1]["minimax_keyframes"]) == 1
    released = nodes.SyntaxH3SoundGuide.execute(pos, _AudioVAE(), latent, audio, 0.3).result[0]
    assert [(c[1]["start_percent"], c[1]["end_percent"]) for c in released] == [(0.0, 0.3), (0.3, 1.0)]
    assert len(released[0][1]["minimax_keyframes"]) == 2 and len(released[1][1]["minimax_keyframes"]) == 1
