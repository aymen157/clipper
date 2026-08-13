# Clipper

Clipper (formely tinyvideo), is a library for composing and editing videos.

similar to adobe premiere or after effects, but it's purpose is automatisation through scripting.
it doesn't replace these DCC software, it has a different purpose that boosts productivity and flexibility

advantages include access to scripted effects, which means access to ALL python functions and libraries, such as open cv filters, PIL effects, AI filters, and so on.


# structure

`clipper.py` is the lightweight core module, and it contains main structures: image/text/video/audio/export functions.
`clipper_tools.py` contains filters to be applied on a clip with pixels (ie image/text/videos)
`clipper_tools_audio.py` contains filters to be applied on an audio clip.

both `tools` files depend on the core, but they are completely decoupled. and every function inside them, is decoupled from the other functions (ie. you can delete any function at any time and the script won't break)


# technical details

video is streamed from disk, frame by frame. the filters applied over frame at time t.

audios are loaded into memory. (because small memory footprint, but also not loading it leadss to Codec Pre-roll & Windowing which leads to jagged/bumps in output if we edit it on the fly from reads from disk. Sample-Exact Slicing vs. Keyframe Seeking etc..)

audio_file_clip func decodes the source audio into 32-bit float (float32) PCM, resampled to 44.1 kHz stereo, stored as a NumPy array shaped (samples, 2). (its interleaved, ie flt, not planar fltp)