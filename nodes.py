import json

from comfy_api.latest import io

from .timeline import (H3_FPS, HIT_SOUNDS, analyse, auto_bars, bar_cuts, downbeat_at, frame_levels, hits_audio, kick_only, pause_cuts,
                       shots_from_cuts, slice_audio, strip_overlay)

MODES = ["music", "dialogue", "action"]

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
            description="Cut your track into shots, each guided by its part of the sound: the kick for music, the voice for dialogue, the track for action. Most outputs are lists, so the nodes after it run once per shot; join the shot videos with Concatenate Video.",
            inputs=[
                io.Audio.Input("track", optional=True,
                               tooltip="Your song, dialogue or sound effects. It sets the timeline and is the soundtrack of the finished video. Leave it empty to mark hits on a blank timeline."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image", tooltip="Reference images, used by the shots in order and repeated when there are more shots than images."),
                    prefix="image", min=1, max=50)),
                io.Audio.Input("drums", optional=True,
                               tooltip="music: the drums stem (Audio Separation -> Drums). Its kick guides the shots and its beat sets the grid. Default: the track."),
                io.Audio.Input("voice", optional=True,
                               tooltip="dialogue: the voice (Audio Separation -> Vocals). It guides the lips and places the cuts. Default: the track."),
                io.Combo.Input("mode", options=MODES, default="music",
                               tooltip="music: shots of whole bars, guided by the kick. dialogue: cuts in the pauses, guided by the voice. action: cuts between hits, guided by the track."),
                io.Float.Input("start_at", default=0.0, min=0.0, max=36000.0, step=0.01,
                               tooltip="Where the first shot starts, in seconds. music snaps it to the next downbeat."),
                io.Float.Input("duration", default=0.0, min=0.0, max=36000.0, step=0.1,
                               tooltip="How long the video runs from start_at, in seconds, split across the images. 0: about 5.5 s per image for music, the rest of the sound otherwise. Without a track, the length of the blank timeline (0: 10 s)."),
                io.Float.Input("bpm", default=0.0, min=0.0, max=300.0, step=0.01, tooltip="music only. 0 finds the tempo; type it in if you know it."),
                io.Combo.Input("hit_sound", options=HIT_SOUNDS, default="thump",
                               tooltip="The sound each marked hit makes in the guide: thump (punches, blows), crash (cymbals), boom (thunder, explosions)."),
                io.String.Input("edits", default="", tooltip="Shots and hits from the timeline editor, in seconds. Written by the editor."),
            ],
            is_output_node=True,
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
    def execute(cls, images, start_at, duration, bpm, mode="music", hit_sound="thump", edits="",
                track=None, drums=None, voice=None) -> io.NodeOutput:
        images = [images[k] for k in sorted(images, key=lambda k: int(k[len("image"):])) if images[k] is not None]
        edits = json.loads(edits) if edits.strip() else {}
        hits = sorted(float(h) for h in edits.get("hits", []))
        end = None
        blank = track is None
        if blank:
            if not hits:
                raise ValueError("Syntax Timeline: connect a track, or double-click the timeline to mark hits")
            sr = 44100
            end = duration or 10.0
            track = {"waveform": hits_audio(hits, end, sr, hit_sound).view(1, 1, -1), "sample_rate": sr}
            guide = track
            if mode == "music":
                mode = "action"
        else:
            if duration:
                end = start_at + duration
            if mode == "music":
                drums = drums or track
                guide = {"waveform": kick_only(drums["waveform"], drums["sample_rate"]), "sample_rate": drums["sample_rate"]}
            else:
                guide = (voice if mode == "dialogue" else None) or track
            if hits:
                sr = guide["sample_rate"]
                drawn = hits_audio(hits, guide["waveform"].shape[-1] / sr, sr, hit_sound)
                peak = float(guide["waveform"].abs().max()) or 1.0
                guide = {"waveform": guide["waveform"] + drawn.to(guide["waveform"]) * peak, "sample_rate": sr}

        grid = None
        if mode == "music":
            grid = analyse(_mono(drums), drums["sample_rate"])
            if bpm > 0:
                grid["bpm"] = bpm
            bar = 240.0 / grid["bpm"]
            bars = max(len(images), round(duration / bar)) if duration else len(images) * auto_bars(grid["bpm"])
            cuts = bar_cuts(downbeat_at(grid, start_at), grid["bpm"], bars, len(images))
            how = "{:.2f} BPM, {} bars".format(grid["bpm"], bars)
        else:
            cuts = pause_cuts(_mono(guide), guide["sample_rate"], start_at, len(images), end, move_end=end is not None and not blank)
            how = "cut in the pauses"
        edited = edits.get("cuts")
        if edited and len(edited) > 1:
            cuts = sorted(float(c) for c in edited)
            how = "cuts from the editor"

        shots = shots_from_cuts(cuts)
        guides = [{"waveform": slice_audio(guide["waveform"], guide["sample_rate"], s["start"], s["length"] / H3_FPS),
                   "sample_rate": guide["sample_rate"]} for s in shots]
        total = sum(s["frames"] for s in shots) / H3_FPS
        soundtrack = {"waveform": slice_audio(track["waveform"], track["sample_rate"], cuts[0], total), "sample_rate": track["sample_rate"]}
        info = "{}, {} shots from {:.2f} s ({:.2f} s): {}".format(how, len(shots), cuts[0], total, ", ".join("{:.1f} s".format(s["frames"] / H3_FPS) for s in shots))
        if any(s["length"] > 362 for s in shots):
            info += ". Shots over 15 s go past H3's trained range: keep MiniMax H3 Context Windows on the model."
        ui = {"timeline": [{"duration": track["waveform"].shape[-1] / track["sample_rate"], "mode": mode, "cuts": cuts, "hits": hits,
                            "images": len(images), "bpm": grid["bpm"] if grid else None, "downbeat": grid["downbeat"] if grid else None}]}
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
        shot_images = [images[i % len(images)] for i in range(len(shots))]
        return io.NodeOutput(shot_images, guides, [s["length"] for s in shots], [s["frames"] for s in shots], soundtrack, info, strips, ui=ui)


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
