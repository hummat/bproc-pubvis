# BlenderProc Publication Visualization
Publication-ready visualization of 3D objects and point clouds in seconds.

## Installation
```bash
pip install bproc-pubvis
blenderproc pip install fire loguru
```

## Basic Usage
```bash
blenderproc run main.py --obj_path /path/to/3d.obj --save /path/to/output.png
```

<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<model-viewer src="suzanne.glb" camera-controls tone-mapping="neutral" shadow-intensity="1" auto-rotate></model-viewer>