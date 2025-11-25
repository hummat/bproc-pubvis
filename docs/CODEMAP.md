# bproc-pubvis Code Map

This document complements `README.md` by showing **where** CLI options end up in the code and how the main pieces fit together.  
It’s aimed at contributors who want to change behavior without re-discovering the structure each time.

---

## Modules at a Glance

- `bproc_pubvis/main.py`
  - Tyro CLI entrypoint; defines the `Config` dataclass and `run(cfg)` orchestration.
  - Contains no Blender-specific logic beyond high-level pipeline wiring.
- `bproc_pubvis/constants.py`
  - Enum definitions for everything user-facing:
    - Appearance: `Color`, `Shading`, `Shape`, `Look`, `Shadow`, `Engine`, `Light`.
    - Scene/behavior: `Primitive`, `Animation`, `Strength`.
- `bproc_pubvis/utils.py`
  - BlenderProc / `bpy` integration and all “real work”:
    - Renderer / camera: `init_renderer`, `make_camera`, `render_color`, `render_depth`, `make_animation`.
    - Object loading & setup: `load_data`, `make_obj`, `setup_obj`, `init_mesh`, `init_pointcloud`, `rotate_obj`.
    - Environment: `setup_backdrop`, `make_lights`, `add_ambient_occlusion`.
    - Color helpers: `get_color`, `set_color`, `set_background_color`, `set_look`.
    - Sampling utilities: `_target_count`, `_subsample_indices`, `normalize`, `depth_to_image`.
    - Export / video: `export_obj`, `create_mp4_with_ffmpeg`.
- `tests/`
  - `test_main.py`: verifies `run(cfg)` and `main()` wiring using a stubbed pipeline.
  - `test_utils_helpers.py`: unit tests for sampling, normalization, color and image helpers.
  - `test_smoke.py`: basic import/enum sanity check.
  - `test_integration.py`: opt-in BlenderProc CLI integration tests.

---

## CLI → Config → Implementation

Tyro turns the `Config` dataclass in `bproc_pubvis/main.py` into CLI flags.  
Hyphenated flags map to fields 1:1 (e.g. `--bg-color` → `Config.bg_color`).

| Feature                    | CLI flag(s)                | `Config` field          | Main implementation points                                     |
|---------------------------|----------------------------|-------------------------|-----------------------------------------------------------------|
| Input object              | `--data`                   | `data`                  | `load_data`, `setup_obj` in `bproc_pubvis/utils.py`            |
| Center / scale            | `--center`, `--scale`      | `center`, `scale`       | `load_data` (centering/scaling logic)                          |
| Rotation                  | `--rotate`                 | `rotate`                | `rotate_obj` via `setup_obj`                                   |
| Gravity                   | `--gravity`                | `gravity`               | Gravity logic in `run`, plus `setup_backdrop` (physics)        |
| Animation                 | `--animate`, `--frames`, `--fps`    | `animate`, `frames`, `fps`     | `run` (Animation resolution), `make_animation`                 |
| Shading                   | `--shade`                  | `shade`                 | `init_mesh`                                                    |
| Keep custom material      | `--keep-material`          | `keep_material`         | Material branch in `setup_obj`                                 |
| Object color              | `--color`                  | `color`                 | `set_color`, `get_color`, `set_look`                           |
| Roughness                 | `--roughness`              | `roughness`             | Material setup in `setup_obj`, pcd material in `render_depth`  |
| Point cloud mode          | `--pcd`                    | `pcd`                   | `load_data`, `setup_obj`, `render_depth`                       |
| Depth rendering           | `--depth`                  | `depth`                 | Depth branch in `run`, `render_depth`                          |
| Wireframe                 | `--wireframe`              | `wireframe`             | Wireframe section in `setup_obj`                               |
| Keep mesh                 | `--keep-mesh`              | `keep_mesh`             | `load_data`, `setup_obj`, `render_depth`                       |
| Point size / shape/color  | `--point-size`, `--point-shape`, `--point-color` | `point_size`, `point_shape`, `point_color` | `init_pointcloud`, `setup_obj`, `render_depth`   |
| Subsampling               | `--subsample`, `--subsample-method` | `subsample`, `subsample_method` | `load_data`, `render_depth`, `_subsample_indices`        |
| Camera location / offset  | `--cam-location`, `--cam-offset` | `cam_location`, `cam_offset` | `make_camera`, `set_color` (distance-based colormaps)  |
| Resolution                | `--resolution`             | `resolution`            | `init_renderer`, `get_camera_resolution`, `render_depth`       |
| Depth-of-field            | `--fstop`                  | `fstop`                 | `make_camera` (calls `add_depth_of_field`)                     |
| Backdrop plane / HDRI     | `--backdrop`, `--transparent False` / `--bg-color` | `backdrop`, `transparent`, `bg_color` | `setup_backdrop`, `render_color`, `set_background_color` |
| Lighting intensity        | `--light`                  | `light`                 | Light resolution in `run`, `make_lights`                       |
| Shadow style              | `--shadow`                 | `shadow`                | Shadow preset in `run`, `setup_backdrop`, `make_lights`        |
| Ambient occlusion         | `--ao`                     | `ao`                    | AO decision in `run`, `add_ambient_occlusion`                  |
| Render engine             | `--engine`                 | `engine`                | `init_renderer` (`CYCLES` vs `EEVEE`)                          |
| Noise / samples           | `--noise-threshold`, `--samples` | `noise_threshold`, `samples` | `init_renderer` (adaptive sampling, max samples)        |
| Exposure / look           | `--exposure`, `--look`     | `exposure`, `look`      | `init_renderer`, `set_output_format`, `set_look`               |
| Transparency              | `--transparent` (use `False` to show backdrop) | `transparent`            | `init_renderer` (enable alpha), `setup_backdrop`               |
| Save / show               | `--save`, `--show`         | `save`, `show`          | `run` (default show), `render_color`, `render_depth`           |
| Export mesh               | `--export`                 | `export`                | `export_obj`                                                   |
| Verbose/debug             | `--verbose`, `--debug`     | `verbose`, `debug`      | Logging setup in `run`, `make_animation` (debug mode)          |
| Random seed               | `--seed`                   | `seed`                  | RNG seeding in `run`, various random utilities                 |

> Note: exact flag spellings are controlled by Tyro; use `blenderproc run main.py --help` to list them.

---

## High-Level Pipeline (Code-Oriented)

Use this as a guide when changing behavior in `run(cfg)`:

1. **Initialization** (`main.py`)
   - Seeds RNGs (`random.seed`, `np.random.seed`).
   - Configures `loguru` logging (debug vs info).
   - Calls `init_renderer(...)` with `resolution`, `transparent`, `look`, `exposure`, `engine`, `noise_threshold`, `samples`.
2. **Object setup** (`bproc_pubvis/utils.py`)
   - `setup_obj(...)`:
     - Delegates to `load_data(...)` (file/primitive/tuple handling, centering, scaling, mesh ↔ pcd).
     - Creates BlenderProc objects via `make_obj(...)`.
     - Applies materials, colors (`set_color`), shading (`init_mesh`), wireframe, or point cloud setup (`init_pointcloud`).
3. **Camera and look**
   - `make_camera(...)`:
     - Builds camera pose from `cam_location` + `cam_offset` and object location.
     - Clears any existing camera animation; enforces exactly one pose (frame range `[0, 1]`).
     - Adds optional depth of field based on `fstop`.
   - `set_look(...)`:
     - If no explicit `look`, chooses a `Look` enum preset depending on `pcd`, `depth`, and `color`.
     - Calls `set_output_format(...)` → drives Blender view transform / exposure.
4. **Optional depth stage**
   - If `cfg.depth`:
     - Calls `render_depth(...)`:
       - For pure depth images: returns `None` after writing a colormapped image (optionally with `bg_color`).
       - For depth→pcd: constructs a point cloud from the depth map, adds noise, sets material/colormap, and configures point cloud visualization.
     - If `render_depth` returns falsy, `run` exits early—no backdrop or lighting.
5. **Gravity and backdrop**
   - `run` decides whether gravity is allowed:
     - Disabled for point clouds and forced off for tumble animation.
   - Light strength is resolved from `Light` enum, string name, or float.
   - `setup_backdrop(...)`:
     - Handles HDRI backdrops (`--backdrop` path) vs plane-based backdrops (`backdrop.ply`).
     - Configures transparency, background color, and optional gradient falloff.
     - When `gravity=True`, sets up rigid bodies and runs physics via BlenderProc.
6. **AO and lighting**
   - `add_ambient_occlusion(...)`:
     - Scene-level compositing when called without object.
   - `make_lights(...)`:
     - Creates key/fill/rim lights, scales light size based on `Shadow` preset, and sets energy from `light_intensity`.
7. **Render or animate**
   - If `cfg.animate`:
     - `make_animation(...)` handles keyframes and rendering to GIF/MP4 using `render_color(...)` internally.
   - Else:
     - Single `render_color(...)` call renders one frame and saves/shows according to `save`/`show`.
   - Optional `export_obj(...)` writes mesh/pcd to `.obj` / `.glb` / `.gltf`.

---

## Where to Change What

- **Add a new CLI option**
  - Add a field to `Config` in `main.py` with a sensible default and type.
  - Thread it into `run(cfg)` and, where appropriate, into helpers in `bproc_pubvis/utils.py`.
  - Prefer adding behavior in `bproc_pubvis/utils.py` rather than in `main.py`.

- **Adjust object loading / normalization**
  - Modify `load_data(...)` for mesh/pcd/tuple inputs, centering, and scaling.
  - Use `_target_count` / `_subsample_indices` for any subsampling logic.

- **Change point cloud rendering**
  - `init_pointcloud(...)` and `_pointcloud_with_geometry_nodes*` / `_pointcloud_with_particle_system`.
  - `render_depth(...)` for depth→pcd behavior and default sampling/resolution.

- **Change look, exposure, or color behavior**
  - `set_look(...)`, `set_output_format(...)`, and `set_color(...)` in `bproc_pubvis/utils.py`.

- **Change lighting or backdrop**
  - `setup_backdrop(...)` for backdrop plane/HDRI and gravity interaction.
  - `make_lights(...)` for light placement, shadow softness, and energy.

- **Change animation behavior**
  - `make_animation(...)` for keyframing, pivot/empty usage, and GIF/MP4 output.
  - `bake_physics(...)` and gravity handling in `run(cfg)` for tumble behavior.

Use this map together with `README.md` (for user-facing behavior) and the tests in `tests/` (to see what is relied upon in CI) before making larger changes.
