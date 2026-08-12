

video is streamed from disk, frame by frame. the filters applied over frame at time t.

audios are loaded into memory. (because small memory footprint, but also not loading it leadss to Codec Pre-roll & Windowing which leads to jagged/bumps in output if we edit it on the fly from reads from disk. Sample-Exact Slicing vs. Keyframe Seeking etc..)