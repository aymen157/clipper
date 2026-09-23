# Clipper

Clipper (formely tinyvideo), is a library for composing and editing videos/audios

Similar to adobe premiere or after effects, but it's purpose is automatisation through scripting.
it doesn't replace these DCC software, it has a different purpose that boosts productivity and flexibility

Advantages include access to scripted effects, which means access to ALL python functions and libraries, such as open cv filters, PIL effects, AI filters, and so on.

# structure

`clipper.py` is the lightweight core module, and it contains main structures: image/text/video/audio/export functions.
`clipper_tools.py` contains filters to be applied on a clip with pixels (ie image/text/videos)
`clipper_tools_audio.py` contains filters to be applied on an audio clip.

Both `tools` files depend on the core, but they are completely decoupled. and every function inside them, is decoupled from the other functions (ie. you can delete any function at any time and the script won't break)


# Example usage (video)

#### Trimming & Time Stretching
```python
clip_a = trim(clip, start=2.0, end=10.0) # Slice from 2s to 10s
# Extend duration by looping or reversing
clip_wrap = extend(clip, additional_duration=5.0, mode="wrap") # Seamless loop
clip_yoyo = extend(clip, additional_duration=5.0, mode="yoyo") # Forward-reverse loop
```
#### Resizing, Aspect Ratio & Fitting
```python
# Static resize (fixed height, auto aspect ratio)
hd_clip = resize(clip, height=720, keep_aspect=True)
# Dynamic zoom-in over time
zooming = rescale(clip, scale=lambda t: 1.0 + 0.1 * t)
# Aspect ratio fitting (Letterboxing / Crop to fill canvas)
letterboxed = object_fit(clip, container_width=1920, container_height=1080, mode="contain")
cropped = object_fit(clip, container_width=1080, container_height=1920, mode="cover") # 9:16 vertical
```
#### Fades & Color Effects
```python
# Fades (handles transparency automatically for RGBA)
faded = fade_in(clip, duration=1.0)
faded = fade_out(faded, duration=1.5)
# Color inversion
inverted = invert_colors(clip)
``` 

#### Concatenation & Multi-Layer Compositing
```python
# Sequential stitching (fast)
video = concat([intro_clip, main_clip, outro_clip])

# Multi-layer canvas compositing with positioning & blend modes
overlay_clip.x, overlay_clip.y = 100, 50
overlay_clip.delay = 2.0
overlay_clip.blend_mode = "screen"

composited = blend_clips(
    videos=[background_clip, overlay_clip],
    size=(1920, 1080),
    default_blend_mode="normal"
)
```

# Example usage (audio)

#### Trimming audio
```python
clip = audio_file_clip('file.mp3', channels=2) # load audio clip into RAM
# Cut 2 seconds off the start
clip_a = trim_start(clip, amount=2.0)
# Cut 10% off the end
clip_b = trim_end(clip, amount=10.0, percent=True)
# Slice between 5s and 15s
clip_c = trim(clip, start=5.0, end=15.0)
# Slice from 10% to 80% mark
clip_d = trim(clip, start=10.0, end=80.0, percent=True)
```
#### Blending / Mixing Tracks
```python
# Mix multiple tracks together (supports optional .delay attribute per clip)
clip1.delay = 0.0   # play instantly
clip2.delay = 2.5   # start 2.5s later
mixed = blend([clip1, clip2], normalize=True) # blended result
```
#### Volume & Dynamic Fades
```python
# Fixed gain (e.g., +50% volume or -6 dB)
louder = volume(clip, factor=1.5)
quieter = volume(clip, factor=-6.0, db=True)
fade_in = volume(clip, factor=lambda t: min(1.0, t / 3.0)) # fade-in over 3 seconds
```
### Sequential transformation
```python
trimmed = trim(clip, start=2.0, end=20.0)
boosted = volume(trimmed, factor=3.0, db=True)
processed = volume(boosted, factor=lambda t: min(1.0, t / 2.0))
```

# technical details

Video is <b>streamed</b> from disk, frame by frame. the filters applied over frame at time t.

Audios are <b>loaded entirely into memory</b>. (because small memory footprint, but also not loading it leads to Codec Pre-roll & Windowing which in turn leads to jagged/bumps in output if we edit it on the fly from reads from disk. Sample-Exact Slicing vs. Keyframe Seeking etc..)

`audio_file_clip` func decodes the source audio into 32-bit float (float32) PCM, resampled to 44.1 kHz stereo, stored as a NumPy array shaped (samples, 2). (its interleaved, ie flt, not planar fltp)
