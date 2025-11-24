import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, cast

import numpy as np
import tyro
from loguru import logger

sys.path.append(str(Path(__file__).parent.absolute()))
from src.constants import Animation, Color, Engine, Light, Look, Shading, Shadow, Shape, Strength
from src.utils import (
    add_ambient_occlusion,
    export_obj,
    init_renderer,
    make_animation,
    make_camera,
    make_lights,
    render_color,
    render_depth,
    set_look,
    setup_backdrop,
    setup_obj,
)


@dataclass
class Config:
    """Creates and renders publication-ready visualizations of 3D meshes and point clouds.

    This function serves as the main entry point for the bproc-pubvis library, handling
    the complete pipeline from loading 3D objects to final rendering. It supports both
    static renders and animations, with extensive customization options for materials,
    lighting, camera positioning, and rendering quality.
    """

    data: str | Path | Tuple[Path, Path]
    """Path to a single or two 3D object files (e.g. mesh + depth). Or name of BLender primitive."""

    # Object manipulation options
    center: bool = True
    """Whether to center the object at the origin"""
    scale: bool | float = True
    """Whether to scale the object to fit within a unit cube or by how much to scale it"""
    rotate: Optional[Tuple[float, float, float]] = (0, 0, -35)
    """Initial rotation angles (x, y, z) in degrees"""
    gravity: bool = False
    """Whether to enable physics-based gravity simulation"""
    animate: Optional[Union[Animation, str, bool]] = None
    """Animation type to apply (turn, tumble) or False for static render"""

    # Material options
    shade: Union[Shading, str] = Shading.FLAT
    """Shading style to apply to the object"""
    keep_material: bool = False
    """Whether to keep the custom material or apply the default one"""
    color: Optional[Union[Tuple[float, float, float], Color, str]] = None
    """Color for the object in RGB format"""
    roughness: Optional[float] = None
    """Material roughness value"""

    # Visualization options
    pcd: Union[bool, int] = False
    """Create a point cloud by sampling points from the surface of the mesh"""
    depth: Union[Literal["ray_trace", "z_buffer"], bool] = False
    """Visualize the given mesh as a (projected) depth map"""
    wireframe: Union[Tuple[float, float, float], Color, str, bool] = False
    """Whether to render the object as a wireframe"""
    keep_mesh: bool = False
    """Whether to keep the mesh object after creating the point cloud"""
    point_size: Optional[float] = None
    """Size of points when rendering point clouds"""
    point_color: Optional[Union[Tuple[float, float, float], Color, str]] = None
    """Color for the points in RGB format"""
    point_shape: Optional[Union[Shape, str]] = None
    """Shape to use for point cloud visualization"""
    subsample: Optional[int | float] = None
    """Number of points or fraction to subsample the point cloud"""
    subsample_method: Literal["random", "fps"] = "random"
    """Subsampling strategy: 'random' for uniform random, 'fps' for farthest point sampling"""

    # Camera options
    cam_location: Tuple[float, float, float] = (1.5, 0, 1)
    """Camera position in 3D space"""
    cam_offset: Tuple[float, float, float] = (0, 0, 0)
    """Additional offset applied to camera position"""
    resolution: Union[int, Tuple[int, int]] = 512
    """Output resolution (single int for square, tuple for rectangular)"""
    fstop: Optional[float] = None
    """Camera f-stop value for depth of field"""

    # Environment options
    backdrop: Union[bool, str] = True
    """Whether to include a backdrop plane"""
    light: Optional[Union[Light, str, float]] = None
    """Lighting intensity preset or custom value"""
    bg_color: Optional[Union[Tuple[float, float, float], Color, str]] = None
    """Background color in RGB format"""
    bg_light: float = 0.15
    """Background light intensity"""
    transparent: Union[bool, float] = True
    """Whether to render with transparency"""

    # Rendering options
    look: Optional[Union[Look, str]] = None
    """Visual style preset to apply"""
    exposure: float = 0
    """Global exposure adjustment"""
    shadow: Optional[Shadow | str] = None
    """Shadow type and intensity"""
    ao: Optional[Union[bool, float]] = None
    """Ambient occlusion strength or disable"""
    engine: Union[Engine, str] = Engine.CYCLES
    """Rendering engine to use"""
    noise_threshold: Optional[float] = None
    """Cycles render noise threshold"""
    samples: Optional[int] = None
    """Number of render samples"""

    # Output options
    save: Optional[Union[Path, str]] = None
    """Output file path for rendered image"""
    show: bool = False
    """Whether to display the render"""
    export: Optional[Union[Path, str]] = None
    """Output file path for OBJ file export"""

    # Debug options
    verbose: bool = False
    """Enable verbose logging"""
    debug: bool = False
    """Enable debug mode"""
    seed: int = 1337
    """Random seed for reproducibility"""

    def __post_init__(self):
        pass


def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if cfg.verbose or cfg.debug:
        logger.add(sys.stderr, level="DEBUG")

    init_renderer(
        resolution=cfg.resolution,
        transparent=bool(cfg.transparent or cfg.bg_color is not None),
        look=cfg.look,
        exposure=cfg.exposure,
        engine=cfg.engine,
        noise_threshold=cfg.noise_threshold or 0.01,
        samples=cfg.samples or 100,
    )

    animate = (Animation.TURN
               if isinstance(cfg.animate, bool) and cfg.animate else Animation(cfg.animate) if cfg.animate else None)
    point_shape = Shape.SPHERE if animate in [Animation.TURN, Animation.TUMBLE] else cfg.point_shape
    obj = setup_obj(
        obj_path=cfg.data,
        center=cfg.center,
        scale=cfg.scale,
        pcd=cfg.pcd if cfg.pcd > 1 else (cfg.pcd and not cfg.depth),
        wireframe=cfg.wireframe,
        keep_mesh=cfg.keep_mesh,
        set_material=not cfg.keep_material,
        color=cfg.color,
        cam_location=cfg.cam_location,
        roughness=cfg.roughness,
        point_shape=point_shape,
        rotate=cfg.rotate,
        shade=cfg.shade,
        point_size=cfg.point_size,
        point_color=cfg.point_color,
        subsample=cfg.subsample,
        subsample_method=cfg.subsample_method,
    )

    make_camera(obj=obj, location=cfg.cam_location, offset=cfg.cam_offset, fstop=cfg.fstop)

    pcd_for_look = False
    if isinstance(cfg.pcd, bool):
        pcd_for_look = cfg.pcd
    elif isinstance(cfg.pcd, int):
        pcd_for_look = cfg.pcd <= 1

    set_look(
        look=cfg.look,
        color=cfg.color,
        pcd=pcd_for_look,
        depth=bool(cfg.depth),
    )

    default_show = cfg.save is None and not os.getenv("USE_EXTERNAL_BPY_MODULE")
    should_show = cfg.show or default_show

    if cfg.depth:
        obj = render_depth(
            obj=obj,
            pcd=cfg.pcd,
            keep_mesh=cfg.keep_mesh,
            color=cfg.color,
            cam_location=cfg.cam_location,
            roughness=cfg.roughness,
            point_shape=point_shape,
            point_size=cfg.point_size,
            subsample=cfg.subsample,
            subsample_method=cfg.subsample_method,
            bg_color=cfg.bg_color,
            ray_trace=cfg.depth != "z_buffer",
            save=Path(cfg.save).resolve() if cfg.save is not None else None,
            show=should_show,
        )
        if not obj:
            return

    is_mesh = len(obj.get_mesh().polygons) > 0
    if cfg.gravity and not is_mesh:
        logger.warning("Disabling gravity for point clouds.")
        gravity = False
    else:
        gravity = cfg.gravity

    offset = np.array([0, 0, -0.05])
    if animate is Animation.TUMBLE:
        if gravity:
            logger.warning("Disabling gravity for tumble animation.")
            gravity = False
        offset = np.array([0, 0, -0.6])

    base_light = cfg.light or Light.BRIGHT
    logger.debug(f"Setting light intensity to {base_light}")
    if isinstance(base_light, float):
        light = base_light
    elif isinstance(base_light, str):
        light = Light[base_light.upper()].value
    else:
        light = cast(Any, base_light).value
    if gravity or cfg.backdrop:
        setup_backdrop(
            obj=obj,
            shadow_strength=Strength.OFF if cfg.shadow == "off" else Strength.MEDIUM,
            transparent=cfg.transparent,
            color=None if cfg.transparent else cfg.bg_color,
            hdri_path=Path(cfg.backdrop).resolve() if isinstance(cfg.backdrop, str) else None,
            bg_light=cfg.bg_light * light,
            gravity=gravity,
            offset=offset,
        )

    if cfg.ao or (cfg.ao is None and (is_mesh or cfg.depth)):
        ao = 0.5 if isinstance(cfg.ao, bool) or cfg.ao is None else cfg.ao
        logger.debug(f"Setting ambient occlusion strength to {ao}")
        add_ambient_occlusion(strength=ao)

    shadow = (Shadow(cfg.shadow) if cfg.shadow else (Shadow.MEDIUM if (is_mesh or cfg.depth) else Shadow.SOFT))
    logger.debug(f"Setting shadow type to {shadow}")
    make_lights(obj=obj, shadow=shadow, light_intensity=light)

    save_path = Path(cfg.save).resolve() if cfg.save is not None else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    if animate:
        save = Path(animate.value).with_suffix(".gif") if save_path is None else save_path
        make_animation(obj=obj, save=save, animation=animate, bg_color=cfg.bg_color, debug=cfg.debug)
    elif not cfg.debug:
        render_color(
            bg_color=cfg.bg_color,
            save=save_path,
            show=should_show,
        )
    if cfg.export:
        export_obj(obj=obj, path=Path(cfg.export).resolve())


def main():
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
