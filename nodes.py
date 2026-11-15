import comfy.audio
from comfy_api.latest import io

from .apply import CurveCFG, CurveFunControlPatch, add_sound_guide, blend_conditioning
from .curves import DEFAULT_CURVE, INTERPOLATIONS, evaluate, parse_points, resample
from .render import STYLES, control_video, overlay
from .sound import AUDIO_MODES, SAMPLE_RATE, SOUNDS, audio_curve, guide_audio

CATEGORY = "SyntaxNodes/Action Curves"

Curve = io.Custom("ACTION_CURVE")


class SyntaxActionCurveEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxActionCurveEditor",
            display_name="Syntax Action Curve Editor",
            category=CATEGORY,
            description="Draw an action timing/intensity curve. Click to add a key, drag to move, right-click to delete.",
            inputs=[
                io.Int.Input("length", default=124, min=2, max=10000,
                             tooltip="Frames in the curve. Match the video length (H3: 17n+5 at 24 fps, LTX: 8n+1); apply nodes stretch the curve over the latent anyway."),
                io.Combo.Input("interpolation", options=INTERPOLATIONS, default="smooth",
                               tooltip="smooth: monotone cubic, never overshoots. linear: straight segments. step: holds each key."),
                io.String.Input("curve_data", default=DEFAULT_CURVE,
                                tooltip="Keyframes as JSON {\"points\": [[time 0-1, value 0-1], ...]}. Written by the editor."),
            ],
            outputs=[Curve.Output(display_name="curve")],
        )

    @classmethod
    def execute(cls, length, interpolation, curve_data) -> io.NodeOutput:
        return io.NodeOutput(evaluate(parse_points(curve_data), interpolation, length))


class SyntaxCurveMixer(io.ComfyNode):
    OPERATIONS = ["max", "add", "multiply", "average", "min", "subtract"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxCurveMixer",
            display_name="Syntax Multi Curve Mixer",
            category=CATEGORY,
            description="Combine curves left to right into one control signal, clamped to 0..1. Shorter curves are stretched to the longest.",
            inputs=[
                io.Combo.Input("operation", options=cls.OPERATIONS, default="max"),
                io.Autogrow.Input("curves", template=io.Autogrow.TemplatePrefix(Curve.Input("curve"), prefix="curve_", min=2)),
            ],
            outputs=[Curve.Output(display_name="curve")],
        )

    @classmethod
    def execute(cls, operation, curves) -> io.NodeOutput:
        names = sorted(curves, key=lambda name: int(name.rsplit("_", 1)[-1]))
        curves = [curves[name] for name in names if curves[name] is not None]
        if not curves:
            raise ValueError("Syntax Multi Curve Mixer: connect at least one curve")
        length = max(len(c) for c in curves)
        curves = [resample(c, length) for c in curves]
        if operation == "average":
            return io.NodeOutput([sum(v) / len(curves) for v in zip(*curves)])
        out = curves[0]
        for c in curves[1:]:
            if operation == "max":
                out = [max(a, b) for a, b in zip(out, c)]
            elif operation == "min":
                out = [min(a, b) for a, b in zip(out, c)]
            elif operation == "add":
                out = [a + b for a, b in zip(out, c)]
            elif operation == "subtract":
                out = [a - b for a, b in zip(out, c)]
            else:
                out = [a * b for a, b in zip(out, c)]
        return io.NodeOutput([min(max(v, 0.0), 1.0) for v in out])


class SyntaxCurveToControlVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxCurveToControlVideo",
            display_name="Syntax Curve to Control Video",
            category=CATEGORY,
            description="Render the curve as white-on-black control frames, one per curve frame, for a ControlNet video input.",
            inputs=[
                Curve.Input("curve"),
                io.Combo.Input("style", options=STYLES, default="bounce",
                               tooltip="bounce: ball height follows the curve. pulse: disk size follows the curve. intensity: whole-frame brightness. meter: filled bar."),
                io.Int.Input("width", default=672, min=16, max=8192, step=16,
                             tooltip="Control frames are rescaled to the generation size by the ControlNet, so half resolution is enough."),
                io.Int.Input("height", default=384, min=16, max=8192, step=16),
            ],
            outputs=[io.Image.Output(display_name="control_video")],
        )

    @classmethod
    def execute(cls, curve, style, width, height) -> io.NodeOutput:
        return io.NodeOutput(control_video(curve, width, height, style))


class SyntaxCurvePreviewOverlay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxCurvePreviewOverlay",
            display_name="Syntax Curve Preview Overlay",
            category=CATEGORY,
            description="Draw the curve and a per-frame playhead along the bottom of a video, to check the action lands on the curve.",
            inputs=[
                io.Image.Input("images"),
                Curve.Input("curve"),
                io.Float.Input("strip_height", default=0.18, min=0.05, max=1.0, step=0.01, tooltip="Strip height as a fraction of the frame height."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, curve, strip_height) -> io.NodeOutput:
        return io.NodeOutput(overlay(images, curve, strip_height))


class SyntaxActionCurveConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxActionCurveConditioning",
            display_name="Syntax Action Curve Apply (Conditioning Blend)",
            category=CATEGORY,
            description="Blend base and action prompts frame by frame: where the curve is high the video follows the action prompt. Works on any video model, including MiniMax H3 and LTX. Costs one extra positive pass per step.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Base prompt, after image/reference/guide conditioning."),
                io.Conditioning.Input("action_positive", tooltip="Action prompt. Inherits the guides and references of the base positive."),
                io.Latent.Input("latent", tooltip="The latent going into the sampler, used for the frame layout."),
                Curve.Input("curve"),
                io.Float.Input("strength", default=1.0, min=0.0, max=4.0, step=0.01, tooltip="How far the curve peaks move toward the action prompt. Above 1 pushes past it, like CFG between the two prompts; keep 1 for spikes and short hits, above 1 corrupts those frames on MiniMax H3 at CFG 5."),
                io.Boolean.Input("hard_switch", default=False,
                                 tooltip="Switch prompts where the curve crosses 0.5 instead of blending (prompt segments)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, action_positive, latent, curve, strength, hard_switch) -> io.NodeOutput:
        samples = latent["samples"]
        shapes = [t.shape for t in samples.unbind()] if samples.is_nested else [samples.shape]
        if hard_switch:
            curve = [1.0 if v >= 0.5 else 0.0 for v in curve]
        return io.NodeOutput(blend_conditioning(positive, action_positive, shapes, curve, strength))


class SyntaxActionCurveGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxActionCurveGuidance",
            display_name="Syntax Action Curve Apply (Guidance)",
            category=CATEGORY,
            description="Scale CFG per frame by the curve: full CFG at curve peaks, CFG 1 at zero. No effect when the sampler runs at CFG 1 (distilled models).",
            inputs=[
                io.Model.Input("model"),
                Curve.Input("curve"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, curve, strength) -> io.NodeOutput:
        m = model.clone()
        m.set_model_sampler_cfg_function(CurveCFG(curve, strength))
        return io.NodeOutput(m)


class SyntaxActionCurveH3FunControl(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxActionCurveH3FunControl",
            display_name="Syntax Action Curve Apply (H3 Fun ControlNet)",
            category=CATEGORY,
            description="Apply a MiniMax H3 Fun ControlNet whose strength follows the curve on each frame. Use in place of Apply MiniMax H3 Fun ControlNet.",
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Image.Input("control_video"),
                Curve.Input("curve"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength at curve value 1."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, control_video, curve, strength, start_percent, end_percent) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(model)
        m = model.clone()
        model_sampling = model.get_model_object("model_sampling")
        patch = CurveFunControlPatch(
            model_patch, vae, control_video[..., :3].movedim(-1, 1), strength,
            float(model_sampling.percent_to_sigma(start_percent)),
            float(model_sampling.percent_to_sigma(end_percent)),
            curve,
        )
        patch.register(m)
        return io.NodeOutput(m)


class SyntaxCurveToGuideAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxCurveToGuideAudio",
            display_name="Syntax Curve to Guide Audio",
            category=CATEGORY,
            description="Turn the curve into a guide sound for an audio-video model: a hit on every rise past 0.5, or a wind/murmur bed as loud as the curve. Feed it to Syntax H3 Sound Guide: the video syncs its action to the sound.",
            inputs=[
                Curve.Input("curve"),
                io.Combo.Input("sound", options=SOUNDS, default="thump",
                               tooltip="thump: punches, hammer blows, beats. crash: cymbal hits. boom: thunder and explosions, rings while the curve stays up. wind: gusts as loud as the curve. murmur: speech-like sound wherever the curve is up, for dialogue timing."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01, tooltip="Frame rate of the video the curve spans (H3: 24)."),
                io.Audio.Input("sound_clip", optional=True, tooltip="Your own one-shot, placed on every rise in place of the built-in sound."),
            ],
            outputs=[io.Audio.Output(display_name="guide_audio")],
        )

    @classmethod
    def execute(cls, curve, sound, fps, sound_clip=None) -> io.NodeOutput:
        clip = None
        if sound_clip is not None:
            clip = comfy.audio.resample(sound_clip["waveform"][:1], sound_clip["sample_rate"], SAMPLE_RATE)[0].mean(0)
        x = guide_audio(curve, fps, sound, clip)
        return io.NodeOutput({"waveform": x.expand(1, 2, -1).clone(), "sample_rate": SAMPLE_RATE})


class SyntaxH3SoundGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxH3SoundGuide",
            display_name="Syntax H3 Sound Guide",
            category=CATEGORY,
            description="Anchor a guide sound at frame 0 of a MiniMax H3 video. H3 syncs the picture to its soundtrack, so the action lands on the sound's hits. Keep the prompt plain and in line with the sound: a prompt that fights it wins.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive after MiniMax H3 Image to Video (first frame) or other guides."),
                io.Vae.Input("audio_vae"),
                io.Latent.Input("latent", tooltip="The MiniMax H3 AV latent going into the sampler."),
                io.Audio.Input("guide_audio"),
                io.Float.Input("guide_steps", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of sampling steps, from the start, that see the guide. 1: the guide sound becomes the soundtrack (hits, beats). About 0.3: the guide sets the timing, then the model writes its own audio (dialogue)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, audio_vae, latent, guide_audio, guide_steps) -> io.NodeOutput:
        return io.NodeOutput(add_sound_guide(positive, audio_vae, latent, guide_audio, guide_steps))


class SyntaxAudioToCurve(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxAudioToCurve",
            display_name="Syntax Audio to Curve",
            category=CATEGORY,
            description="Make a curve from a soundtrack's loudness, one value per video frame, to time action to music or a voice track.",
            inputs=[
                io.Audio.Input("audio"),
                io.Int.Input("length", default=141, min=2, max=10000, tooltip="Frames in the curve, usually the video length."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01),
                io.Combo.Input("mode", options=AUDIO_MODES, default="loudness",
                               tooltip="loudness: follows the volume. onsets: a spike wherever the volume jumps up (beats, hits)."),
                io.Float.Input("range_db", default=40.0, min=1.0, max=120.0, step=0.5,
                               tooltip="Loudness range mapped to 0..1, in dB below the loudest frame. Lower it to ignore quiet parts."),
            ],
            outputs=[Curve.Output(display_name="curve")],
        )

    @classmethod
    def execute(cls, audio, length, fps, mode, range_db) -> io.NodeOutput:
        return io.NodeOutput(audio_curve(audio["waveform"], audio["sample_rate"], length, fps, mode, range_db))


NODES = [
    SyntaxActionCurveEditor,
    SyntaxCurveMixer,
    SyntaxCurveToControlVideo,
    SyntaxCurvePreviewOverlay,
    SyntaxActionCurveConditioning,
    SyntaxActionCurveGuidance,
    SyntaxActionCurveH3FunControl,
    SyntaxCurveToGuideAudio,
    SyntaxH3SoundGuide,
    SyntaxAudioToCurve,
]
