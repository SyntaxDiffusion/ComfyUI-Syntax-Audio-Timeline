# Recipes

Settings behind the clips in the README. All use MiniMax H3 ref2va at 864x480 and 24 fps, four images, seed 1, timed on a DGX Spark (GB10). The audio is from Epidemic Sound and is not included; use your own.

These clips predate two changes: the music guide now keeps only the kick of the drums stem, and every shot also sees a shared *Anchor* image as \<Picture 2\>. The prompts below are as they were run; the workflows now start with *set in \<Picture 1\>, with the exact characters and style of \<Picture 2\>*.

## Reference images

One still per shot, made with GPT Image at 2048x1152. The pattern that worked:

*Still frame from a 1990s anime OVA: hand-drawn cel animation with richly painted backgrounds and sharp, expressive character design. [character, outfit, and a setting packed with retro electronics]. The scene is lit almost entirely by the glowing screens and LEDs: deep shadows, saturated hot red and electric blue light, soft film grain. A few thin white abstract line-art accents float lightly over the scene (a sine wave, a dotted orbit, a fragment of perspective grid), subtle and never covering the character. 16:9 widescreen composition, no text, no logos, no watermark.*

Keep overlays thin: heavy graphics in the reference get animated instead of the scene. Give every character a distinct look.

## Music

*Beat Culture* by V.V. Campos, with its drums stem (Audio Separation) on *drums*. `timeline_music_hits`: mode music, start_at 1.0 (snapped to the downbeat at 1.75 s), 85.00 BPM found, 2 bars (5.6 s) per shot, 22.6 s in all. Full model, 30 steps, CFG 5: 1 h 03 min. The fast version (turbo LoRA, 8 steps, CFG 1): 11 min.

- **prompt**: A music video in the exact style, characters and setting of \<Picture 1\>. Between drum hits everything holds still. On every drum hit the lights and screens flash, graphic elements pulse outward in one hard burst and the characters move sharply to the beat, then instantly settle back. Static camera, no text.
- **result**: screens flash, speed-line bursts and light flares land on the drums; the fast version flashes more gently.
- **finding the grid**: tempo and downbeat come from the drums stem. On this track a beat tracker on the full mix reads 113 BPM, and the mix puts the downbeat on the off-beat; the stem gives 85 BPM and the downbeat within 13 ms.

## Dialogue

Four lines from two voices with ~0.9 s pauses, 13.9 s. `timeline_dialogue`: mode dialogue, the voice guides each shot. All three cuts land in the middle of the pauses. Full model, 30 steps, CFG 5: 38 min.

- **prompt**: A scene in the exact style, characters and setting of \<Picture 1\>. The character speaks the line heard in the audio, mouth moving in sync with the voice, with natural expressions and small gestures, and holds still in the silences. Static camera, no text.
- **result**: the speakers' mouths move on their lines. In a shot with two people the wrong one spoke: name the speaker ("the young man in the parka says...").

## Action

Eight punches and impacts with gaps, 17.5 s. `timeline_action`: mode action, the effects guide each shot. Cuts fall between impacts. Full model, 30 steps, CFG 5: 44 min.

- **prompt**: An action scene in the exact style, characters and setting of \<Picture 1\>. On every impact sound the character strikes a hard, fast punch and the whole frame jolts, then instantly freezes again until the next impact. Static camera, no text.
- **result**: characters throw punches with impact flashes on the hits, even from calm reference images; one shot only showed a fist entering frame.

## Action without audio

`timeline_action_drawn_hits`: mode action, no track, duration 12 s, hits marked at 1.6, 3.4, 6.2, 8.0 and 10.5 s with *hit_sound* thump. The cuts fall between the hits (2.7, 5.6 and 9.1 s). Same prompt as Action.
