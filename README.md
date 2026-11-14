# ComfyUI Syntax Action Curves

Draw a timing/intensity curve and make a video model's action follow it: peaks where the action happens, valleys where it rests. No training. Nodes live under `SyntaxNodes/Action Curves`.

## Nodes

| Node | What it does |
| --- | --- |
| **Syntax Action Curve Editor** | Canvas curve editor. Click to add a key, drag to move, right-click to delete. Presets (pulse, ramp + peak, gallop, single hit...), copy/paste JSON. Outputs an `ACTION_CURVE` (one 0..1 value per frame). |
| **Syntax Multi Curve Mixer** | max / add / multiply / average / min / subtract over any number of curves. |
| **Syntax Curve to Control Video** | Renders the curve as control frames: `bounce` (ball height), `pulse` (disk size), `intensity`, `meter`. |
| **Syntax Curve Preview Overlay** | Burns the curve and a yellow playhead into the bottom of a video, to check the action lands on the curve. |
| **Syntax Action Curve Apply (Conditioning Blend)** | Blends a base prompt and an action prompt frame by frame. Any video model (MiniMax H3, LTX, Wan...). |
| **Syntax Action Curve Apply (Guidance)** | Per-frame CFG that follows the curve. Needs CFG > 1. |
| **Syntax Action Curve Apply (H3 Fun ControlNet)** | Drop-in for *Apply MiniMax H3 Fun ControlNet* with per-frame strength from the curve. |
| **Syntax Curve to Guide Audio** | Turns the curve into a guide sound: thump, crash or boom on every rise past 0.5, a wind or speech-murmur bed as loud as the curve, or your own `sound_clip` on every rise. |
| **Syntax H3 Sound Guide** | Anchors the guide sound at frame 0 of a MiniMax H3 video, for all steps or only the first part of sampling. |
| **Syntax Audio to Curve** | Curve from a soundtrack's loudness or onsets, to time action to music or a voice track. |

## How each Apply works

- **Conditioning Blend** gives the base and action positives complementary per-frame cond masks (`1 - curve`, `curve`), so ComfyUI's sampler averages the two predictions frame by frame. On multi-stream latents (H3 and LTX audio+video), the audio stream is weighted along its own time axis. The action prompt inherits the base positive's guides and references, so encode it with the same I2V/reference node. Connect the latent that goes into the sampler. *strength* above 1 pushes peaks past the action prompt (the base weight goes negative there, like CFG between the two prompts). Keep it at 1 for spikes and short hits: on MiniMax H3 at CFG 5, strength above 1 on a hit shorter than ~0.5 s decodes as corrupted green/magenta frames. A single-frame spike always lands on one latent frame at full value, and at strength 1 that is enough to trigger the action. *hard_switch* turns the blend into prompt segments split where the curve crosses 0.5. Costs one extra positive pass per step.
- **Guidance** sets CFG to `1 + (cfg - 1) * curve` on each latent frame. Distilled H3/LTX workflows that run at CFG 1 get no effect from it.
- **H3 Fun ControlNet** scales the control skip on each target video frame's tokens by the curve. Pair it with *Curve to Control Video* for a fully curve-driven control signal.

- **Sound guide** (MiniMax H3, the strongest method on H3): *Curve to Guide Audio* → *H3 Sound Guide*. H3 syncs its picture to its soundtrack, so a sound on each hit makes the action land there. Keep one plain prompt that agrees with the sound; a prompt that fights it wins. Make the negative from the positive before the guide (*ConditioningZeroOut*). *guide_steps* 1 keeps the guide sound as the soundtrack, so pick a sound that suits the scene. About 0.3 lets the guide set the timing and then the model writes its own audio, which is how dialogue gets real voices with true silences between lines (0.2 lets speech leak into the gaps, 0.4 copies the murmur into the soundtrack). Prompt stillness between hits: a prompt that implies constant action ("a drummer playing") ignores them. Measured on full H3 at CFG 5: punches +0.81/+0.60 motion correlation, dialogue +0.88 loudness correlation, against +0.61/+0.74 for the best Conditioning Blend setup.

Curves are stretched over each model's latent frames on normalized time, so one curve works for any length.

## Credits

The sound-guide method (curve → synthetic sound → MiniMax H3 audio guide, released after part of the steps) comes from [Nekodificador](https://x.com/Nekodificador)'s event curves in [ComfyUI-NKD-Basic-Tools](https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools). The sound synthesis here is a separate implementation.

## Tests

```
python -m pytest tests --import-mode=importlib
```
