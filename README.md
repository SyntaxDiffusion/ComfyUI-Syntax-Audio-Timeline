# ComfyUI Syntax Audio Timeline

Turn a track and a few images into a cut, synced MiniMax H3 video: shots cut on the beat or in the pauses, with the action in each shot timed to its part of the sound. Music, dialogue or action, and action needs no audio at all: mark where you want the hits.

Two nodes. Everything else in the workflows is core ComfyUI.

| Music: cut on the beat | Dialogue: cut in the pauses | Action: cut between hits |
| --- | --- | --- |
| ![music](docs/music.gif) | ![dialogue](docs/dialogue.gif) | ![action](docs/action.gif) |

## Quick start

1. **Install**: clone this repo into `ComfyUI/custom_nodes/` and restart ComfyUI. No extra Python packages. For music, also install [audio-separation-nodes-comfyui](https://github.com/christian-byrne/audio-separation-nodes-comfyui) to get the drums stem (or load a stem file).
2. **Models** (from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)):

   | File | Folder |
   | --- | --- |
   | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models` |
   | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (or `..._nvfp4_awq` on RTX 50-series) | `models/text_encoders` |
   | `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` |
   | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (fast drafts only) | `models/loras` |

3. **Run**: drag a workflow from `example_workflows/` into ComfyUI, load your track and images, and press Run. Each workflow has a note explaining what to change.

| You want | Workflow | mode |
| --- | --- | --- |
| Start here: any track, switch between music, dialogue and action | `timeline_base` | any |
| A music video, hits that land hard | `timeline_music_hits` | music |
| A quick music draft | `timeline_music_fast` | music |
| Characters speaking a voice track | `timeline_dialogue` | dialogue |
| Action timed to sound effects | `timeline_action` | action |
| Action where you place it, no audio | `timeline_action_drawn_hits` | action |

## The timeline

**Syntax Timeline** takes your track and any number of images and cuts the track into shots. Pick a **mode**:

- **music**: shots of whole bars from the downbeat after *start_at*, each guided by the **kick** of its part of the song. Connect the drums stem to **drums** (Audio Separation -> Drums); the node keeps only the kick, and finds the tempo and downbeat from the full kit. *bpm* 0 finds the tempo.
- **dialogue**: cuts moved into the pauses, so no line is cut in half, each shot guided by its part of the **voice** (Audio Separation -> Vocals, or the track if it is only voice).
- **action**: cuts moved between hits, each shot guided by its part of the track.

*duration* is how long the video runs from *start_at*, split across the images; 0 gives about 5.5 s per image for music and runs to the end of the sound otherwise. Each shot hears its part of the guide from its first frame, so H3 times the action to it. The finished video carries your original track.

The editor on the node shows the track's waveform as soon as a Load Audio is connected, and the shots right away. Hover it for hints; **?** lists every control.

- **Range**: drag the yellow handles to set where the video starts and ends.
- **Shots**: drag a white cut to move it, double-click a shot to split it, right-click a cut to merge the two shots. *One shot* makes a single shot over the whole range; *By images* goes back to one shot per image. More shots than images reuse the images in order, so one image can fill 30 s as one shot or several.
- **Hits** (the top lane): double-click to mark a hit, drag to move it, right-click to delete it. Each hit adds a *hit_sound* (thump, crash or boom) to the guide. With no track, set *duration* for a blank timeline; the hits drive the action and are the soundtrack.
- music snaps to bars; hold Alt for beats, Shift to move freely.
- **Analyse** runs just the Timeline (and what feeds it) to find the beat and the exact cuts without rendering.

**Syntax Preview Strip** is optional: it draws the guide's loudness, the cuts, the hits and a playhead along the bottom of each shot, to check that the action lands. Delete it for a clean video.

## Getting good results

- **CFG lands the hits.** The full model at 30 steps and CFG 5, with the negative zeroed *before* the guide (*ConditioningZeroOut* on the Reference to Video output), pushes the picture onto the sound: motion vs drum hits +0.89 on a 5 s test clip. The 4-step turbo LoRA at CFG 1 follows only loosely (+0.27), and raising CFG on the LoRA burns the image.
- **Guide with the stem, not the mix.** A full mix never goes quiet, so H3 finds no hits in it, and beat detection on a busy mix can land the downbeat on the off-beat.
- **Ask for stillness between hits.** A prompt that implies constant motion ignores the guide.
- **Anchor every shot.** Each shot sees its own image as `<Picture 1>` and one shared *Anchor* image as `<Picture 2>` (a character or style sheet, or the first image), which keeps people and style the same across cuts.
- **Name the speaker** when a shot has two people, or the wrong one may talk.
- **Long shots need Context Windows.** Past 15 s, H3's trained range, keep *MiniMax H3 Context Windows* on the model (as in `timeline_base`): the look holds for 30 s and more, but motion follows the beat more loosely than in short shots.

[recipes.md](recipes.md) has the prompts, settings and timings behind the clips above.

## Nodes

All under `SyntaxNodes/Timeline`.

| Node | What it does |
| --- | --- |
| **Syntax Timeline** | Cuts the track into shots for music, dialogue or action. Outputs lists (image, guide audio, frames to generate, frames to keep, strip), so the nodes after it run once per shot, plus the soundtrack and an *info* line. |
| **Syntax Preview Strip** | Draws a shot's part of the sync strip along the bottom of its frames. |

Join the shots with core nodes: *Get Image from Batch* (length from *frames*) and *Create Video* per shot, then *Concatenate Video* with the Timeline's *soundtrack* in *complete_audio*.

## Credits

The idea of steering H3's action with a guide sound comes from [Nekodificador](https://x.com/Nekodificador)'s event curves in [ComfyUI-NKD-Basic-Tools](https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools). The earlier curve-based nodes of this pack are on the `v1` branch.

## Tests

From `ComfyUI/custom_nodes`:

```
python -m pytest ComfyUI-SyntaxActionCurves/tests --import-mode=importlib
```
