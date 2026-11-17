import json

import torch

from comfy_api.latest import io

from .timeline import (H3_FPS, HIT_SOUNDS, analyse, auto_bars, beat_cuts, downbeat_at, even_cuts, frame_levels, hits_audio, pause_cuts,
                       shots_from_cuts, slice_audio, strip_overlay)

CUT_MODES = ["beats", "pauses", "even"]

CATEGORY = "SyntaxNodes/Timeline"

Strip = io.Custom("SYNTAX_STRIP")


def _mono(audio):
    return audio["waveform"][0].float().mean(0)


class SyntaxTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxTimeline",
            display_name="Syntax Timeline",
            category=CATEGORY,
            description="Cut your track into one shot per image: on the beat, in the pauses, or around hits you mark. Most outputs are lists, so the nodes after it run once per shot; join the shot videos with Concatenate Video.",
            inputs=[
                io.Audio.Input("track", optional=True,
                               tooltip="Your song, dialogue or sound effects. It sets the timeline and is the soundtrack of the finished video. Leave it empty to mark hits on a blank timeline."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image", tooltip="One reference image per shot, in order."), prefix="image", min=1, max=50)),
                io.Audio.Input("guide", optional=True,
                               tooltip="What each shot moves to: the drums stem for music (Audio Separation -> Drums), the voice for dialogue. Also used to find the beat. Default: the track."),
                io.Combo.Input("cut_on", options=CUT_MODES, default="beats",
                               tooltip="beats: music, shots of whole bars from a downbeat. pauses: dialogue and action, the rest of the track split across the images with each cut moved into a pause. even: the rest of the track split equally."),
                io.Float.Input("start_at", default=0.0, min=0.0, max=36000.0, step=0.01,
                               tooltip="Where the first shot starts, in seconds. With beats it snaps to the next downbeat."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=300.0, step=0.01, tooltip="beats only. 0 finds the tempo; type it in if you know it."),
                io.Int.Input("bars_per_shot", default=0, min=0, max=64, tooltip="beats only. 0 picks the whole bars closest to 5.5 seconds."),
                io.Float.Input("duration", default=10.0, min=1.0, max=600.0, step=0.1, tooltip="Length of the timeline in seconds when there is no track."),
                io.Combo.Input("hit_sound", options=HIT_SOUNDS, default="thump",
                               tooltip="The sound each marked hit makes in the guide: thump (punches, blows), crash (cymbals), boom (thunder, explosions)."),
                io.String.Input("edits", default="", tooltip="Cuts and hits from the timeline editor, in seconds. Written by the editor; clear it to go back to automatic cuts."),
            ],
            has_intermediate_output=True,
            outputs=[
                io.Image.Output(display_name="image", is_output_list=True),
                io.Audio.Output(display_name="guide_audio", is_output_list=True),
                io.Int.Output(display_name="length", is_output_list=True, tooltip="Frames to generate, on MiniMax H3's 17k+5 grid."),
                io.Int.Output(display_name="frames", is_output_list=True, tooltip="Frames each shot keeps in the cut."),
                io.Audio.Output(display_name="soundtrack", tooltip="The part of the track the shots cover."),
                io.String.Output(display_name="info"),
                Strip.Output(display_name="strip", is_output_list=True, tooltip="For Syntax Preview Strip: each shot's part of the sync strip."),
            ],
        )

    @classmethod
    def execute(cls, images, start_at, bpm, bars_per_shot, cut_on="beats", duration=10.0, hit_sound="thump",
                edits="", track=None, guide=None) -> io.NodeOutput:
        images = [images[k] for k in sorted(images, key=lambda k: int(k[len("image"):])) if images[k] is not None]
        edits = json.loads(edits) if edits.strip() else {}
        hits = sorted(float(h) for h in edits.get("hits", []))
        end = None
        if track is None:
            if not hits:
                raise ValueError("Syntax Timeline: connect a track, or double-click the timeline to mark hits")
            sr = 44100
            track = {"waveform": hits_audio(hits, duration, sr, hit_sound).view(1, 1, -1), "sample_rate": sr}
            guide, end = track, duration
            if cut_on == "beats":
                cut_on = "pauses"
        else:
            guide = guide or track
            if hits:
                sr = guide["sample_rate"]
                drawn = hits_audio(hits, guide["waveform"].shape[-1] / sr, sr, hit_sound)
                peak = float(guide["waveform"].abs().max()) or 1.0
                guide = {"waveform": guide["waveform"] + drawn.to(guide["waveform"]) * peak, "sample_rate": sr}
        grid = None
        if cut_on == "beats":
            grid = analyse(_mono(guide), guide["sample_rate"])
            if bpm > 0:
                grid["bpm"] = bpm
            bars = bars_per_shot or auto_bars(grid["bpm"])
            first = downbeat_at(grid, start_at)
            cuts = beat_cuts(len(images), grid["bpm"], first, bars)
            how = "{:.2f} BPM, {} bars per shot".format(grid["bpm"], bars)
        else:
            cuts = (pause_cuts if cut_on == "pauses" else even_cuts)(_mono(guide), guide["sample_rate"], start_at, len(images), end)
            how = "cut on {}".format(cut_on)

        edited = edits.get("cuts")
        if edited and len(edited) == len(images) + 1:
            cuts = sorted(float(c) for c in edited)
            how = "cuts from the editor"
        elif edited:
            how += " (editor cuts ignored: they are for {} images)".format(len(edited) - 1)

        shots = shots_from_cuts(cuts)
        guides = [{"waveform": slice_audio(guide["waveform"], guide["sample_rate"], s["start"], s["length"] / H3_FPS),
                   "sample_rate": guide["sample_rate"]} for s in shots]
        total = sum(s["frames"] for s in shots) / H3_FPS
        soundtrack = {"waveform": slice_audio(track["waveform"], track["sample_rate"], cuts[0], total), "sample_rate": track["sample_rate"]}
        info = "{}, {} shots from {:.2f} s ({:.2f} s): {}".format(how, len(shots), cuts[0], total, ", ".join("{:.1f} s".format(s["frames"] / H3_FPS) for s in shots))
        if any(s["length"] > 362 for s in shots):
            info += ". Shots over 15 s are outside H3's trained range: add images or move start_at."
        ui = {"timeline": [{"duration": track["waveform"].shape[-1] / track["sample_rate"], "mode": cut_on, "cuts": cuts, "hits": hits,
                            "bpm": grid["bpm"] if grid else None, "downbeat": grid["downbeat"] if grid else None}]}
        n = sum(s["frames"] for s in shots)
        start = round(cuts[0] * H3_FPS)
        covered = slice_audio(guide["waveform"], guide["sample_rate"], cuts[0], total)[0].float().mean(0)
        whole = {"level": frame_levels(covered, guide["sample_rate"], n),
                 "cuts": [round(c * H3_FPS) - start for c in cuts[1:-1]],
                 "hits": [round(h * H3_FPS) - start for h in hits if 0 <= round(h * H3_FPS) - start < n]}
        offsets = [0]
        for s in shots[:-1]:
            offsets.append(offsets[-1] + s["frames"])
        strips = [{**whole, "offset": o} for o in offsets]
        return io.NodeOutput(images, guides, [s["length"] for s in shots], [s["frames"] for s in shots], soundtrack, info, strips, ui=ui)


class SyntaxPreviewStrip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SyntaxPreviewStrip",
            display_name="Syntax Preview Strip",
            category=CATEGORY,
            description="Draw a sync strip along the bottom of a shot: the guide's loudness, the cuts, the hits and a playhead that runs across the whole video. To check that the action lands on the sound.",
            inputs=[
                io.Image.Input("images", tooltip="The shot, trimmed to its length."),
                Strip.Input("strip", tooltip="Syntax Timeline's strip output."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, strip) -> io.NodeOutput:
        return io.NodeOutput(strip_overlay(images, strip["level"], strip["cuts"], strip["hits"], strip["offset"]))


NODES = [SyntaxTimeline, SyntaxPreviewStrip]
