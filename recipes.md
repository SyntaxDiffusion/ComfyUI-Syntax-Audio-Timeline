# Recipes

Settings behind the example clips. All use MiniMax H3 fl2va or ref2va, 141 frames at 24 fps, 30 steps, CFG 5, res_multistep / simple, with the start frame made by MiniMax H3 Image (6 steps, CFG 1) and connected to *first_frame*.

To reuse one: paste the curve JSON into the editor's *Paste* button (or the `curve_data` widget), set *interpolation*, pick the *sound* and *guide_steps*, and use the prompts. Results vary by seed, so try two or three.

## Blacksmith forge

*Hits, 1344x768.* All 4 blows land on the curve, sparks on 2 of them. Seed 1. [Watch with sound](docs/forge.mp4)

- **sound**: `thump`, **guide_steps**: `1.0`, **interpolation**: `linear`
- **start frame prompt**: Cinematic film still. A muscular bearded blacksmith in a scorched leather apron stands at a heavy iron anvil in a dark stone forge, gripping a large hammer at shoulder height, a glowing orange-hot blade lying on the anvil, a few embers drifting, the furnace behind him throwing warm orange firelight, deep shadows, shallow depth of field, anamorphic 35mm, rich contrast, film grain.
- **video prompt**: The blacksmith swings the heavy hammer down onto the glowing blade on the anvil. Each strike lands with a loud clang and throws a bright burst of sparks across the dark forge, then he lifts the hammer again and pauses. Warm firelight, handheld cinematic camera, shallow depth of field.
- **curve**:

```json
{"points": [[0, 0], [0.138, 0], [0.14, 1], [0.19, 1], [0.192, 0], [0.328, 0], [0.33, 1], [0.38, 1], [0.382, 0], [0.468, 0], [0.47, 1], [0.52, 1], [0.522, 0], [0.778, 0], [0.78, 1], [0.83, 1], [0.832, 0], [1, 0]]}
```

## Bass drop

*Hits, vertical 768x1344.* All 5 beats land in the soundtrack. Big splashes on beats 1 and 2, smaller pulses on 3 to 5. Seed 1.

- **sound**: `thump`, **guide_steps**: `1.0`, **interpolation**: `linear`
- **start frame prompt**: Extreme macro photograph, vertical framing. A large black subwoofer speaker cone lying face up, a glossy puddle of electric-blue cornstarch slime resting in the center of the cone, droplets glistening, pitch dark background, neon magenta rim light and a cyan light from the side, razor sharp detail, shallow depth of field.
- **video prompt**: The bass drops and the blue slime on the speaker cone leaps up into wild dancing tendrils and splashes on every beat, then slumps back into a calm puddle until the next beat. Deep booming bass. Extreme macro, static camera, neon lighting.
- **curve**:

```json
{"points": [[0, 0], [0.178, 0], [0.18, 1], [0.23, 1], [0.232, 0], [0.378, 0], [0.38, 1], [0.43, 1], [0.432, 0], [0.578, 0], [0.58, 1], [0.63, 1], [0.632, 0], [0.658, 0], [0.66, 1], [0.71, 1], [0.712, 0], [0.838, 0], [0.84, 1], [0.89, 1], [0.892, 0], [1, 0]]}
```

## Grandma cymbal crash

*Hits, vertical 768x1344.* She holds still and crashes the cymbal on each hit, about 0.17 s after the sound. Prompting her as a drummer failed: she drummed non-stop. Seed 1.

- **sound**: `crash`, **guide_steps**: `1.0`, **interpolation**: `linear`
- **start frame prompt**: Vertical photo. A seventy-year-old grandmother with silver hair in a bun, round glasses and a floral cardigan sits behind a sparkly red drum kit in a cozy living room with lace doilies, a floral sofa and framed family photos on the wall, drumsticks raised high, fierce grin, warm lamp light, realistic, candid.
- **video prompt**: The grandmother sits perfectly still behind the drum kit with both sticks raised, staring deadpan at the camera. Suddenly she smashes the crash cymbal once with full power, the cymbal swinging and shimmering, then instantly freezes again with the sticks raised, motionless, until the next crash. Loud cymbal crashes with silence between them. Static phone camera on a tripod, warm living room light.
- **curve**:

```json
{"points": [[0, 0], [0.118, 0], [0.12, 1], [0.17, 1], [0.172, 0], [0.278, 0], [0.28, 1], [0.33, 1], [0.332, 0], [0.358, 0], [0.36, 1], [0.41, 1], [0.412, 0], [0.598, 0], [0.6, 1], [0.65, 1], [0.652, 0], [0.818, 0], [0.82, 1], [0.87, 1], [0.872, 0], [1, 0]]}
```

## Lightning storm

*Hits that ring, 1344x768.* Both strikes land in their windows, with some flicker between them. Seed 1.

- **sound**: `boom`, **guide_steps**: `1.0`, **interpolation**: `linear`
- **start frame prompt**: Cinematic wide film still at dusk. A lone weathered wooden farmhouse with a lit porch lamp on a vast flat desert plain under an enormous dark thunderstorm, heavy low clouds rolling in, cold blue light, dry grass in the foreground, a tiny figure standing on the porch looking at the sky, anamorphic 35mm, film grain.
- **video prompt**: Jagged bolts of lightning strike the desert plain near the farmhouse. Each strike lights the whole landscape in a blinding white flash, followed by rolling thunder, then the dusk darkness returns. Static wide cinematic shot.
- **curve**:

```json
{"points": [[0, 0], [0.198, 0], [0.2, 1], [0.32, 1], [0.322, 0], [0.618, 0], [0.62, 1], [0.74, 1], [0.742, 0], [1, 0]]}
```

## Cockpit dialogue

*Dialogue, 1344x768.* She speaks, a true silent beat, he answers. Seed 1. [Watch with sound](docs/cockpit.mp4)

- **sound**: `murmur`, **guide_steps**: `0.3`, **interpolation**: `smooth`
- **start frame prompt**: Cinematic film still. Two astronauts in worn orange flight suits sit side by side in a cramped dim spacecraft cockpit, helmets off, a woman in her thirties on the left and a man in his forties on the right, both facing each other in profile, lit by glowing amber and green instrument panels, Earth visible through the small window behind them, medium shot, anamorphic 35mm, film grain.
- **video prompt**: The woman on the left turns to the man and says quietly: "The oxygen line is leaking, we have maybe an hour." He looks at her in silence for a moment, then answers calmly: "Then we fix it together, starting now." Dim cockpit, glowing instrument panels, cinematic.
- **curve**:

```json
{"points": [[0, 0], [0.05, 0], [0.09, 1], [0.34, 1], [0.38, 0], [0.56, 0], [0.6, 1], [0.88, 1], [0.92, 0], [1, 0]]}
```

## Talking cat

*Dialogue, vertical 768x1344.* Her line, a silent beat, the cat answers deadpan on cue. Her voice can come out partly as the murmur. Seed 1. [Watch with sound](docs/cat.mp4)

- **sound**: `murmur`, **guide_steps**: `0.3`, **interpolation**: `smooth`
- **start frame prompt**: Candid vertical smartphone photo in a cozy sunlit kitchen in the morning. A fluffy orange tabby cat sits upright on a wooden kitchen table next to a tipped-over white coffee mug and a puddle of spilled coffee, the cat looking up at a young woman in a grey hoodie standing beside the table with her hands on her hips, natural window light, realistic, slightly messy home.
- **video prompt**: The woman points at the spilled coffee and asks the cat: "Did you knock my coffee over?" The cat stares at her in silence for a moment, then turns its head away and answers in a small deadpan voice: "I have no idea what you are talking about." Candid handheld phone video, natural morning light.
- **curve**:

```json
{"points": [[0, 0], [0.05, 0], [0.09, 1], [0.34, 1], [0.38, 0], [0.56, 0], [0.6, 1], [0.88, 1], [0.92, 0], [1, 0]]}
```

## Clifftop wind

*Continuous, 1344x768.* Partial: gust 1 lands well, gust 2 is weak. Wind is the least reliable sound so far.

- **sound**: `wind`, **guide_steps**: `1.0`, **interpolation**: `smooth`
- **start frame prompt**: Cinematic film still. A woman in a long red wool coat stands on a grassy clifftop above a grey ocean, seen from the side in a medium-wide shot, her long dark hair and coat hanging still, tall green grass around her knees, overcast sky, soft diffuse light, anamorphic 35mm, muted colors with the red coat standing out, film grain.
- **video prompt**: Strong gusts of wind sweep across the clifftop, whipping the woman's long dark hair and her red coat and bending the tall grass in rolling waves, then the air falls calm again. Overcast, cinematic, static camera.
- **curve**:

```json
{"points": [[0, 0.05], [0.12, 0.08], [0.32, 0.6], [0.42, 1.0], [0.55, 0.12], [0.7, 0.08], [0.76, 1.0], [0.88, 0.7], [1, 0.25]]}
```
