# BlenderProc Publication Visualization
Publication-ready visualization of 3D objects and point clouds in seconds.

| Mesh                            | Point Cloud                   | Mesh + Depth                           |
|---------------------------------|-------------------------------|----------------------------------------|
| ![mesh](examples/mesh.png) | ![pcd](examples/pcd.png) | ![mesh_depth](examples/mesh.png) |

_Head over to the repository's [**GitHub** Pages site](https://hummat.com/bproc-pubvis) for a prettier and more
interactive version of this README!_

## Installation
```bash
pip install bproc-pubvis
blenderproc pip install fire loguru
```

The first call of `blenderproc` will download [`Blender`](https://blender.org). If you already have a local 
installation, you can use 
`--custom-blender-path /path/to/blender` (this also needs to be used for all subsequent calls of `blenderproc`).

## Basic Usage
To render a mesh (or point cloud if the input is one), simply run:
```bash
blenderproc run main.py path/to/3d.obj
```
The following options can be added to:
* **save** the rendered image: `--save path/to/output.png`
* **export** the object: `--export path/to/output.obj` (use `.glb` for a web-friendly format)
* render the mesh as **point cloud**: `--pcd`
* render the mesh as **depth** image: `--depth`
* render the mesh as **point cloud** from projected **depth** image: `--pcd --depth`

| Mesh                            | Point cloud                                         | Depth                                          |
|---------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| ![mesh](examples/mesh.png) | ![pcd](examples/pcd.png) | ![mesh_depth](examples/mesh.png) |
| `--obj_path suzanne`            | `--mesh_as_pcd`                                     | `--depth`                                             |

## Examples

## Credits

| [**BlenderProc 2**](https://github.com/DLR-RM/BlenderProc)                                                                                     | [**Blender**](https://www.blender.org)                                    |
|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| <img src="https://user-images.githubusercontent.com/6104887/137109535-275a2aa3-f5fd-4173-9d16-a9a9b86f66e7.gif" alt="blenderproc" widht="512"> | ![blender](https://download.blender.org/branding/blender_logo_socket.png) | |