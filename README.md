# ComfyUI Syntax Audio Timeline

Turn a track and a few images into a cut, synced MiniMax H3 video: one shot per image, cut on the beat or in the pauses, with the action in each shot timed to its part of the sound. Music, dialogue or action, and action needs no audio at all: mark where you want the hits.

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

| You want | Workflow | Cut on |
| --- | --- | --- |
| A music video, hits that land hard | `timeline_music_hits` | beats |
| A quick music draft | `timeline_music_fast` | beats |
| Characters speaking a voice track | `timeline_dialogue` | pauses |
| Action timed to sound effects | `timeline_action` | pauses |
| Action where you place it, no audio | `timeline_action_drawn_hits` | pauses |

## The timeline

**Syntax Timeline** takes your track, an optional guide and any number of images, and gives every image a shot:

- **beats** (music): shots of whole bars from a downbeat. The tempo and downbeat are found from the guide; give it the drums stem. *start_at* snaps to the next downbeat, *bpm* 0 finds the tempo, *bars_per_shot* 0 picks the bars closest to 5.5 s.
- **pauses** (dialogue, action): the track from *start_at* to the end of the sound, split across the images, each cut moved into the middle of a pause so no line or hit is cut in half.
- **even**: equal shots.

Each shot hears only its part of the **guide** (the drums for music, the voice for dialogue, the effects for action), anchored at its first frame, so H3 times the action to it. The finished video carries your original track.

The editor on the node shows the track's waveform as soon as a Load Audio is connected. After a run it shows the shots, the beat grid and the cuts:

- **Drag a cut** to move it. With *beats* it snaps to bars; hold Alt for beats, Shift to move freely. *Reset cuts* goes back to automatic.
- **Double-click** to mark a hit, drag to move it, right-click to delete it. Each hit adds a *hit_sound* (thump, crash or boom) to the guide.
- **No track**: leave *track* empty and set *duration* for a blank timeline; the hits you mark drive the action and are the soundtrack.
- To see the cuts without rendering, run only the *Timeline info* node (right-click > Queue Selected Output Nodes).

**Syntax Preview Strip** is optional: it draws the guide's loudness, the cuts, the hits and a playhead along the bottom of each shot, to check that the action lands. Delete it for a clean video.

## Getting good results

- **CFG lands the hits.** The full model at 30 steps and CFG 5, with the negative zeroed *before* the guide (*ConditioningZeroOut* on the Reference to Video output), pushes the picture onto the sound: motion vs drum hits +0.89 on a 5 s test clip. The 4-step turbo LoRA at CFG 1 follows only loosely (+0.27), and raising CFG on the LoRA burns the image.
- **Guide with the drums, not the mix.** A full mix never goes quiet, so H3 finds no hits in it, and beat detection on a busy mix can land the downbeat on the off-beat.
- **Ask for stillness between hits.** A prompt that implies constant motion ignores the guide.
- **Name the speaker** when a shot has two people, or the wrong one may talk.
- **One reference image per shot** holds the character, setting and palette (`<Picture 1>` in the prompt).
- **Keep shots under 15 s**, H3's trained range. *info* warns when one is longer.
- Short shots cut on the beat follow the music better than one long render with *MiniMax H3 Context Windows*, where motion stops following the drums.

[recipes.md](recipes.md) has the prompts, settings and timings behind the clips above.

## Nodes

All under `SyntaxNodes/Timeline`.

| Node | What it does |
| --- | --- |
| **Syntax Timeline** | Cuts the track into one shot per image. Outputs lists (image, guide audio, frames to generate, frames to keep, strip), so the nodes after it run once per shot, plus the soundtrack and an *info* line. |
| **Syntax Preview Strip** | Draws a shot's part of the sync strip along the bottom of its frames. |

Join the shots with core nodes: *Get Image from Batch* (length from *frames*) and *Create Video* per shot, then *Concatenate Video* with the Timeline's *soundtrack* in *complete_audio*.

## Credits

The idea of steering H3's action with a guide sound comes from [Nekodificador](https://x.com/Nekodificador)'s event curves in [ComfyUI-NKD-Basic-Tools](https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools). The earlier curve-based nodes of this pack are on the `v1` branch.

## Tests

From `ComfyUI/custom_nodes`:

```
python -m pytest ComfyUI-SyntaxActionCurves/tests --import-mode=importlib
```
