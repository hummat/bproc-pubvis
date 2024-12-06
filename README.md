# BlenderProc Publication Visualization
Publication-ready visualization of 3D objects and point clouds in seconds.

| Mesh                       | Point Cloud              | Mesh + Depth                     |
|----------------------------|--------------------------|----------------------------------|
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

You can test you render settings using any of the `Blender` primitives (`monkey`, `cube`, `sphere`, `cone`, 
`cylinder`, ...) as the first argument.

| Mesh                       | Point cloud              | Depth                            |
|----------------------------|--------------------------|----------------------------------|
| ![mesh](examples/mesh.png) | ![pcd](examples/pcd.png) | ![mesh_depth](examples/mesh.png) |
| `--obj_path suzanne`       | `--pcd`                  | `--pcd --depth`                  |

## Basic Options

* `--resolution`: Change the resolution of the rendered image (default: `512x512`)
* `--normalize`: Normalize and center the object to fit into a unit cube (`True` by default)
* `--rotate`: Rotate the object using `XYZ` Euler angles (default: `0,0,-35`)
* `--show`: Show the rendered image in a window (`True` if `--save` is not provided)

## Additional Options

Some examples of additional options to customize the rendering are shown below.

### Color

To change the color of the rendered object, use the `--color` option using either any of the predefined colors (e.g. 
`pale_violet`), choosing from those at random (`random_color`), a completely random color (`random`), or a 
three-tuple of RGB values in range 0-1 (e.g. `(0.8,0.5,0.2)`). 
Point clouds can additionally be colored using any of the 
[`matplotlib` colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html). The background color can be
changed using the `--bg_color` option.

| Mesh                             | Point cloud                    | Background                           |
|----------------------------------|--------------------------------|--------------------------------------|
| ![mesh](examples/mesh_color.png) | ![pcd](examples/pcd_color.png) | ![mesh_depth](examples/bg_color.png) |
| `--color bright_blue`            | `--pcd --color viridis`        | `--bg_color pale_turquoise`          |

### Background

By default, the background is transparent. To change this, use the `--bg_color` option as shown above. Additionally, 
`--notransparent` can be used to render the backdrop object.

| Backdrop                       | Colored backdrop                      |
|--------------------------------|---------------------------------------|
| ![mesh](examples/backdrop.png) | ![pcd](examples/backdrop_colored.png) |
| `--notransparent`              | `--notransparent --bg_color pale_red` |

## Light

The default light intensity for meshes is `bright` (`0.7`) and `very_bright` (`1.0`) for point clouds. Use a value 
between 0 and 1 or `very_dark`, `dark`, `medium`, `bright`, or `very_bright` to change the light intensity.

| Very Dark                       | Dark                      | Medium                             |
|---------------------------------|---------------------------|------------------------------------|
| ![mesh](examples/very_dark.png) | ![pcd](examples/dark.png) | ![mesh_depth](examples/medium.png) |
| `--light very_dark`             | `--light dark`            | `--light medium`                   |

### Shadow

Shadows are rendered by default. To disable them, use the `--noshadow` option. To make the shadow softer, use 
`--shadow soft` or --shadow=hard` for a harder shadow.

| Soft shadow                       | Hard shadow                      | No shadow                            |
|-----------------------------------|----------------------------------|--------------------------------------|
| ![mesh](examples/shadow_soft.png) | ![pcd](examples/shadow_hard.png) | ![mesh_depth](examples/noshadow.png) |
| `--shadow soft`                   | `--shadow hard`                  | `--noshadow`                         |

### Shading

The default shading is `flat` for meshes and `smooth` for point clouds. To change this, use the `--shade` option.

| Smooth shading               | Auto-smooth shading              |
|------------------------------|----------------------------------|
| ![mesh](examples/smooth.png) | ![pcd](examples/auto-smooth.png) |
| `--shade smooth`             | `--shade auto`                   |

### Animations

To create an animation, use the `--animate` option. The `--frames` option can be used to specify the number of frames
(default: `72`). To keep transparency, which is not supported by GIF, use `.mp4` as file extension.

| Turn (default, loops)      | Tumble                      |
|----------------------------|-----------------------------|
| ![mesh](examples/turn.gif) | ![pcd](examples/tumble.gif) |
| `--animate`                | `--animate tubmle`          |

### Further Options

Some additional useful options include:

* `--roughness`: Change the roughness of the object. Meshes use `0.5` and point clouds use `0.9` by default.
* `--ao`: Apply ambient occlusion
* `--fstop`: Enable depth of field with a given f-stop
* `--keep_material`: Keep your custom material (only works for `.blend` files)
* `--verbose`: Enable verbose logging during execution
* `--seed`: Set a seed for the random number generator. Useful for random colors or the tumble animation.

## Debugging

`BlenderProc` supports visual debugging inside `Blender` using `blenderproc debug` instead of `blenderproc run`. 
Adding `--debug` will further disable rendering and only setup the scene.

## Credits

| [**BlenderProc 2**](https://github.com/DLR-RM/BlenderProc)                                                                                     | [**Blender**](https://www.blender.org)                                                              |
|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| <img src="https://user-images.githubusercontent.com/6104887/137109535-275a2aa3-f5fd-4173-9d16-a9a9b86f66e7.gif" alt="blenderproc" width="512"> | <img src="https://download.blender.org/branding/blender_logo_socket.png" alt="blender" widht="512"> | |
