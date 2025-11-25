# pyright: ignorefile
# ruff: noqa: E402

import math
import os
import subprocess
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import blenderproc as bproc  # noqa: E402
import bpy
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from blenderproc.python.utility.Utility import Utility, stdout_redirected
from loguru import logger
from PIL import Image
from scipy.spatial import KDTree
from trimesh import PointCloud, Trimesh

from .constants import Animation, Color, Engine, Look, Primitive, Shading, Shadow, Shape, Strength


def export_obj(obj: bproc.types.MeshObject, path: Path):
    """Exports a BlenderProc mesh object to an OBJ or GLTF file.

    This function exports the given BlenderProc mesh object to an OBJ or GLTF file
    at the specified path. The export format is determined based on the file extension.

    Args:
        obj: The BlenderProc mesh object to export.
        path: The file path to export the object to.
    """
    for o in bproc.object.get_all_mesh_objects():
        o.deselect()
    obj.select()
    if path.suffix == ".obj":
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    elif path.suffix in [".glb", ".gltf"]:
        bpy.ops.export_scene.gltf(filepath=str(path), use_selection=True)
    else:
        logger.warning(f"Unsupported export format: {path.suffix}")


def set_output_format(
    look: Optional[Look | str] = None,
    exposure: Optional[float] = None,
    gamma: Optional[float] = None,
):
    """Sets the output format for the Blender scene.

    Args:
        look: The look to apply to the scene view settings.
        exposure: The exposure value to set for the scene view settings.
        gamma: The gamma value to set for the scene view settings.
    """
    if look is not None:
        logger.debug(f"Setting look to {look}")
        bpy.context.scene.view_settings.look = Look(look).value
    if exposure is not None:
        logger.debug(f"Setting exposure to {exposure}")
        bpy.context.scene.view_settings.exposure = exposure
    if gamma is not None:
        logger.debug(f"Setting gamma to {gamma}")
        bpy.context.scene.view_settings.gamma = gamma


def apply_modifier(obj: bproc.types.MeshObject, name: str):
    """Applies a specified modifier to a given Blender object.

    This function permanently applies a modifier to the mesh geometry. The modifier
    must already be added to the object before calling this function.

    Args:
        obj: The Blender object to which the modifier will be applied.
        name: The name of the modifier to apply.
    """
    with bpy.context.temp_override(object=obj.blender_obj):
        bpy.ops.object.modifier_apply(modifier=name)


def get_modifier(obj: bproc.types.MeshObject, name: str) -> "bproc.types.Modifier":
    return obj.blender_obj.modifiers.get(name)


def create_icoshpere(radius: float, subdivisions: int = 1) -> bproc.types.MeshObject:
    """Creates an icosphere mesh object in Blender.

    An icosphere is a spherical mesh made of triangular faces. It's often used
    for point cloud visualization and as a base for more complex geometries.
    Higher subdivision levels create smoother spheres but increase vertex count.

    Args:
        radius: The radius of the icosphere in Blender units
        subdivisions: The number of subdivisions (0-5). Higher values create smoother spheres.

    Returns:
        bproc.types.MeshObject: The created icosphere mesh object
    """
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=subdivisions)
    return bproc.types.MeshObject(bpy.context.object)


def get_camera() -> bproc.types.Entity:
    """Retrieves the active camera in the current Blender scene.

    This function returns the currently active camera used for rendering.
    If multiple cameras exist in the scene, only the active one is returned.

    Returns:
        bproc.types.Entity: The active camera entity in the Blender scene.
    """
    return bproc.types.Entity(bpy.context.scene.camera)


def get_camera_resolution() -> Tuple[int, int]:
    return bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y


def get_all_blender_light_objects() -> List["bpy.types.Object"]:
    """Retrieves all light objects in the current Blender scene.

    Returns:
        List[bpy.types.Object]: A list of Blender light objects.
    """
    return [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]


def convert_to_lights(blender_objects: List["bpy.types.Object"]) -> List[bproc.types.Light]:
    """Converts a list of Blender light objects to a list of BlenderProc light objects.

    Args:
        blender_objects: A list of Blender light objects.

    Returns:
        A list of BlenderProc light objects.
    """
    return [bproc.types.Light(blender_obj=obj) for obj in blender_objects]


def get_all_light_objects() -> List[bproc.types.Light]:
    """Retrieves all light objects in the current Blender scene and converts them to BlenderProc light objects.

    Returns:
        List of BlenderProc light objects.
    """
    return convert_to_lights(get_all_blender_light_objects())


def bake_physics(frames_to_bake: int, gravity: bool = True):
    """Bakes the physics simulation for a specified number of frames.

    This function sets up the physics simulation environment in Blender,
    including enabling or disabling gravity, and bakes the simulation
    for the given number of frames. It also frees the bake cache after
    the simulation is complete.

    Args:
        frames_to_bake: The number of frames to bake the physics simulation.
        gravity: A boolean indicating whether to enable gravity in the simulation.
    """
    bpy.context.scene.use_gravity = gravity
    point_cache = bpy.context.scene.rigidbody_world.point_cache
    point_cache.frame_end = frames_to_bake
    with stdout_redirected():
        with bpy.context.temp_override(point_cache=point_cache):
            bpy.ops.ptcache.bake(bake=True)
            bpy.ops.ptcache.free_bake()


def init_renderer(
    resolution: int | Tuple[int, int],
    transparent: bool,
    look: Look | str | None,
    exposure: float,
    engine: Engine | str,
    noise_threshold: float,
    samples: int,
):
    """Initializes the BlenderProc renderer with the specified settings.

    This function sets up the renderer with the given resolution, transparency,
    look, exposure, rendering engine, noise threshold, and sample count. It
    configures the output format, denoiser, and camera resolution based on the
    provided parameters.

    Args:
        resolution: The resolution of the output image.
        transparent: Whether to enable transparency in the output.
        look: The look to apply to the scene view settings.
        exposure: The exposure value for the scene.
        engine: The rendering engine to use (e.g., Cycles, Eevee).
        noise_threshold: The noise threshold for the denoiser.
        samples: The maximum number of samples for rendering.
    """
    # Allow overriding the base temporary directory for BlenderProc via env.
    # Useful in constrained environments or tests.
    temp_dir_base = os.getenv("BPROC_TEMP_DIR")
    try:
        bproc.init(temp_dir=temp_dir_base)
    except TypeError:
        # Older BlenderProc versions do not accept `temp_dir`; fall back.
        bproc.init()
    if hasattr(bproc, "utility") and hasattr(bproc.utility, "set_keyframe_render_interval"):
        bproc.utility.set_keyframe_render_interval(frame_end=1)

    bproc.renderer.set_output_format(enable_transparency=transparent, view_transform="Filmic")
    set_output_format(look=look, exposure=exposure)

    if Engine(engine) is Engine.CYCLES:
        denoiser_set = False
        for name in ("OPTIX", "OPENIMAGEDENOISE"):
            try:
                bproc.renderer.set_denoiser(name)
            except Exception:  # pragma: no cover - hardware/driver dependent
                continue
            else:  # pragma: no cover - hardware/driver dependent
                denoiser_set = True
                break
        if not denoiser_set:  # pragma: no cover - hardware/driver dependent
            logger.warning("No supported denoiser (OPTIX/OPENIMAGEDENOISE); using Blender defaults.")
        bproc.renderer.set_noise_threshold(noise_threshold)
        bproc.renderer.set_max_amount_of_samples(samples)

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    bproc.camera.set_resolution(*resolution)


def setup_obj(
    obj_path: str | Path | Tuple[Path, Path],
    center: bool = True,
    scale: bool | float = True,
    pcd: bool | int = False,
    wireframe: Tuple[float, float, float] | Color | str | bool = False,
    keep_mesh: bool = False,
    set_material: bool = True,
    color: Optional[Tuple[float, float, float] | Color | str] = None,
    cam_location: Tuple[float, float, float] = (1.5, 0, 1),
    roughness: Optional[float] = None,
    point_shape: Optional[Shape | str] = None,
    rotate: Optional[Tuple[float, float, float]] = None,
    shade: Shading | str = Shading.FLAT,
    point_size: Optional[float] = None,
    point_color: Optional[Tuple[float, float, float] | Color | str] = None,
    subsample: Optional[int | float] = None,
    subsample_method: Literal["random", "fps"] = "random",
) -> bproc.types.MeshObject:
    """Sets up a 3D object in BlenderProc.

    This function loads and processes 3D object data from various sources, applies materials,
    and initializes the object as either a mesh or a point cloud. It also handles object
    transformations such as rotation and shading.

    Args:
        obj_path: Path to the object file or a tuple of paths
        center: Whether to center the object at the origin
        scale: Whether to normalize the object to fit within a unit cube
        pcd: Whether transform the mesh into a point cloud by sampling points from the surface
        wireframe: Whether to render the object as a wireframe
        keep_mesh: Whether to keep the mesh object after creating the point cloud
        set_material: Whether to set a material for the object
        color: The color to apply to the object
        cam_location: The location of the camera
        roughness: The roughness value for the material
        point_shape: The shape of the points if the object is a point cloud
        rotate: The rotation to apply to the object
        shade: The shading mode to apply to the object
        point_size: The size of the points if the object is a point cloud
        point_color: The color to apply to the points in the point cloud
        subsample: If provided and the object is a point cloud, reduce to this many points (int) or
               to this fraction of points if 0 < float <= 1.0
        subsample_method: Subsampling strategy: 'random' (uniform random) or 'fps' (farthest point sampling)

    Returns:
        The created and configured BlenderProc mesh object.
    """
    data = load_data(
        obj_path=obj_path,
        center=center,
        scale=scale,
        pcd=pcd,
        keep_mesh=keep_mesh,
        subsample=subsample,
        subsample_method=subsample_method,
    )
    # Treat `colors` as a local list separate from the incoming `color` arg for type-checking clarity.
    colors: list[Tuple[float, float, float] | Color | str | None]
    if isinstance(data, tuple):
        colors = [color, "plasma_r" if point_color is None else point_color]
    else:
        data = [data]
        colors = [color]

    objs = list()
    for d, c in zip(data, colors, strict=True):
        obj = make_obj(mesh_or_pcd=d)
        is_mesh = len(obj.get_mesh().polygons) > 0

        if set_material or not obj.get_materials():
            material = obj.new_material(f"{'mesh' if is_mesh else 'pointcloud'}_material")
            material.set_principled_shader_value("Roughness", roughness or (0.5 if is_mesh else 0.9))
            set_color(
                obj=obj,
                color=c or (Color.PALE_GREEN if is_mesh else "pointflow"),
                camera_location=cam_location,
                instancer=point_shape is not None,
            )

        if is_mesh:
            init_mesh(obj=obj, shade=shade)
            if wireframe:
                if keep_mesh:
                    wf_color = np.zeros(3) if isinstance(wireframe, bool) else get_color(wireframe)
                    wf_color = np.asarray(wf_color).flatten()
                    # Blender expects an RGB triple; guard against accidental RGBA/long vectors.
                    if wf_color.size >= 4:
                        wf_color = wf_color[:3]
                    if wf_color.size == 0:
                        wf_color = np.zeros(3)
                    wireframe = obj.duplicate()
                    wireframe.set_parent(obj)
                    wireframe.clear_materials()
                    material = wireframe.new_material("wireframe_material")
                    material.set_principled_shader_value("Base Color", [*wf_color.tolist(), 1])
                    material.set_principled_shader_value("Roughness", 0.9)
                    wireframe.add_modifier("WIREFRAME")
                    wireframe = get_modifier(wireframe, "Wireframe")
                    wireframe.thickness = 0.03
                else:
                    obj.add_modifier("WIREFRAME")
                    wireframe = get_modifier(obj, "Wireframe")
                    wireframe.thickness = 0.05
                wireframe.use_relative_offset = True
        else:
            init_pointcloud(obj=obj, point_size=point_size, point_shape=point_shape)
        objs.append(obj)

    mesh = objs[0]
    if len(objs) > 1:
        pcd = objs[1]
        pcd.set_parent(mesh)

    if rotate:
        rotate_obj(obj=mesh, rotate=rotate, persistent=True)

    return mesh


def load_data(
    obj_path: str | Path | Tuple[Path, Path] | Primitive,
    center: bool = True,
    scale: bool | float = True,
    pcd: bool | int = False,
    keep_mesh: bool = False,
    subsample: Optional[int | float] = None,
    subsample_method: Literal["random", "fps"] = "random",
) -> Trimesh | PointCloud | Tuple[Trimesh, PointCloud]:
    """Loads and processes 3D object data from various sources.

    This function can load 3D object data from a file path, a tuple of file paths, or a predefined primitive.
    It supports normalizing the object and converting meshes to point clouds if specified.

    Args:
        obj_path: Path to the object file, a tuple of paths, or a predefined primitive
        center: Whether to center the object at the origin
        scale: Whether to normalize the object to fit within a unit cube
        pcd: Whether to treat the mesh as a point cloud. If an integer is provided, it specifies the number of points to sample.
        keep_mesh: Whether to keep the mesh object after creating the point cloud
        subsample: If returning a point cloud, optionally reduce it to this many points (int)
               or to this fraction of points if 0 < float <= 1.0
        subsample_method: Subsampling strategy when reducing point clouds: 'random' or 'fps'

    Returns:
        A Trimesh, PointCloud, or a tuple of Trimesh and PointCloud, depending on the input and options.
    """
    logger.debug(f"Loading object from {obj_path}")
    if not isinstance(obj_path, (str, Primitive)):
        # Tuple input is expected as (mesh, depth/pcd); detect which entry has faces.
        obj_1_path, obj_2_path = obj_path  # type: ignore[misc]
        obj_1 = trimesh.load(str(obj_1_path), force="mesh")
        obj_2 = trimesh.load(str(obj_2_path), force="mesh")
        if obj_1.faces is None:
            p = PointCloud(obj_1.vertices)
            mesh = Trimesh(obj_2.vertices, obj_2.faces)
        else:
            p = PointCloud(obj_2.vertices)
            mesh = Trimesh(obj_1.vertices, obj_1.faces)
        if center:
            offset = -mesh.bounds.mean(axis=0)
            logger.debug(f"Centering object with offset {offset}")
            mesh.apply_translation(offset)
            p.apply_translation(offset)
        if scale == 1:
            scale = 1 / mesh.extents.max()
            logger.debug(f"Scaling object by {scale}")
            mesh.apply_scale(scale)
            p.apply_scale(scale)
        # Optional subsampling of provided point cloud (tuple input)
        if subsample is not None and subsample > 0:
            m = _target_count(len(p.vertices), subsample)
            if m < len(p.vertices):
                idx = _subsample_indices(p.vertices, subsample, subsample_method)
                p = PointCloud(p.vertices[idx])
        elif pcd > 1:
            p = PointCloud(np.random.permutation(p.vertices)[:pcd])
        return mesh, p

    if isinstance(obj_path, str) and hasattr(Primitive, obj_path.upper()):
        prim = Primitive[obj_path.upper()]
        if prim in [Primitive.SUZANNE, Primitive.MONKEY]:
            obj = bproc.object.create_primitive("MONKEY")
            obj.add_modifier("SUBSURF")
            apply_modifier(obj, "Subdivision")
        else:
            obj = bproc.object.create_primitive(prim.name)
            if prim is not Primitive.SPHERE:
                obj.add_modifier("BEVEL")
                apply_modifier(obj, "Bevel")

        geom = obj.mesh_as_trimesh()
        # Rotate primitives so they face the default camera orientation used by BlenderProc.
        geom.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
        obj.delete()
    elif isinstance(obj_path, (str, Path)) and Path(obj_path).suffix == ".blend":
        # TODO: Fix multi-object and material loading
        objs = bproc.loader.load_blend(obj_path, obj_types="mesh")
        geom = objs[0].mesh_as_trimesh()
        for obj in objs:
            obj.delete()
        if geom.faces is None:
            geom = PointCloud(geom.vertices)
    else:
        geom = trimesh.load(str(obj_path), force="mesh")
    if not isinstance(geom, (Trimesh, PointCloud)):
        raise TypeError(f"Invalid object type: {type(geom)}")
    if center:
        offset = -geom.bounds.mean(axis=0)
        logger.debug(f"Centering object with offset {offset}")
        geom.apply_translation(offset)
    if scale == 1:
        scale = 1 / geom.extents.max()
        logger.debug(f"Scaling object by {scale}")
        geom.apply_scale(scale)
    elif scale > 0:
        logger.debug(f"Scaling object by {scale}")
        geom.apply_scale(scale)

    if isinstance(geom, Trimesh) and pcd:
        # Initial random surface sampling
        count = pcd if isinstance(pcd, int) and pcd > 1 else 4096
        pc = PointCloud(geom.sample(count))
        # Optional secondary subsampling using chosen method
        if subsample is not None and subsample > 0:
            m = _target_count(len(pc.vertices), subsample)
            if m < len(pc.vertices):
                idx = _subsample_indices(pc.vertices, subsample, subsample_method)
                pc = PointCloud(pc.vertices[idx])
        if keep_mesh:
            return geom, pc
        return pc
    elif isinstance(geom, PointCloud):
        pc = geom
        # Optional subsampling of provided point cloud (single input)
        if subsample is not None and subsample > 0:
            m = _target_count(len(pc.vertices), subsample)
            if m < len(pc.vertices):
                idx = _subsample_indices(pc.vertices, subsample, subsample_method)
                pc = PointCloud(pc.vertices[idx])
        elif isinstance(pcd, int) and pcd > 1 and len(pc.vertices) > pcd:
            pc = PointCloud(np.random.permutation(pc.vertices)[:pcd])
        return pc
    return geom


def get_color(
    color: Optional[Tuple[float, float, float] | Color | str],
) -> np.ndarray | Tuple[float, float, float]:
    """Returns the RGB color value based on the input.

    This function converts various color input formats into a standardized RGB triple
    (numpy array or tuple).
    Supports direct RGB values, Color enum values, and special string formats.
    Args:
        color: The color specification, which can be:
            - None: Returns white (1,1,1)
            - Tuple[float,float,float]: Direct RGB values between 0 and 1
            - Color: Enum value from the Color class
            - str: Either a Color enum name (e.g., "WHITE") or special values:
                  "random": Returns random RGB values
                  "random_color": Returns a random predefined Color enum value
    Returns:
        np.ndarray | Tuple[float,float,float]: RGB color values between 0 and 1

    Examples:
        >>> get_color(None)
        (1.0, 1.0, 1.0)
        >>> get_color(Color.WHITE)
        (1.0, 1.0, 1.0)
        >>> get_color("WHITE")
        (1.0, 1.0, 1.0)
    """
    if color is None:
        return Color.WHITE.value
    if isinstance(color, str):
        if hasattr(Color, color.upper()):
            return Color[color.upper()].value
        if color == "random":
            return np.random.rand(3)
        if color == "random_color":
            return np.random.choice(list(Color)).value
    if isinstance(color, Color):
        return color.value
    return color  # type: ignore[return-value]


def set_color(
    obj: bproc.types.MeshObject,
    color: Tuple[float, float, float] | Color | str,
    camera_location: Tuple[float, float, float] | np.ndarray,
    instancer: bool = False,
):
    """Sets the color of a BlenderProc mesh object.

    This function sets the color of a given BlenderProc mesh object. If the color is specified as a string and matches
    a colormap or 'pointflow', it calculates vertex colors based on the object's vertices and the camera location.
    Otherwise, it sets the object's base color directly.

    Args:
        obj: The BlenderProc mesh object to set the color for.
        color: The color to apply to the object. Can be an RGB tuple, a Color enum, or a string.
        camera_location: The location of the camera, used for calculating vertex colors if applicable.
        instancer: A boolean indicating whether to use the 'INSTANCER' attribute type for the color node.
    """
    material = obj.get_materials()[0]
    valid_special = ["pointflow", "random_points"]
    if isinstance(color, str) and (color in plt.colormaps() or color in valid_special):
        values = obj.mesh_as_trimesh().vertices
        colors = list()
        if color == "pointflow":

            def cmap(x, y, z):
                vec = np.array([x, y, z])
                vec = np.clip(vec, 0.001, 1.0)
                norm = np.sqrt(np.sum(vec**2))
                vec /= norm
                return [vec[0], vec[1], vec[2], 1]

            for value in values:
                colors.append(cmap(*value))
        elif color == "random_points":
            for _ in values:
                colors.append((*np.random.rand(3), 1))
        else:
            cmap = plt.get_cmap(color)
            distances = normalize(np.linalg.norm(values - np.array(camera_location), axis=1))
            for dist in distances:
                colors.append(cmap(dist))

        # BlenderProc lacks a helper for creating color attributes, so set the raw mesh attribute.
        mesh = obj.get_mesh()
        color_attr_name = "point_color"
        mesh.attributes.new(name=color_attr_name, type="FLOAT_COLOR", domain="POINT")

        color_data = mesh.attributes[color_attr_name].data
        for i, rgba in enumerate(colors):
            color_data[i].color = rgba

        attribute_node = material.new_node("ShaderNodeAttribute")
        attribute_node.attribute_name = color_attr_name
        if instancer:
            attribute_node.attribute_type = "INSTANCER"
        material.set_principled_shader_value("Base Color", attribute_node.outputs["Color"])
    else:
        material.set_principled_shader_value("Base Color", [*get_color(color), 1])


def set_background_color(image: Image.Image, color: Tuple[float, float, float] | Color | str):
    rgb = get_color(color)
    background = Image.new("RGBA", image.size, tuple(int(c * 255) for c in rgb))
    return Image.alpha_composite(background, image)


def set_look(
    look: Optional[Look | str] = None,
    color: Optional[Tuple[float, float, float] | Color | str] = None,
    pcd: bool = False,
    depth: bool = False,
):
    """Sets the visual style for the BlenderProc renderer.

    This function sets the visual style for the BlenderProc renderer based on the given parameters. If no look is
    specified, it automatically determines the look based on the input data and visualization type.

    Args:
        look: The visual style to apply to the renderer
        color: The color to apply to the object
        pcd: Whether the object is being visualized as a point cloud
        depth: Whether the object is being visualized as a depth map
    """
    if not look:
        look = Look.MEDIUM_CONTRAST
        if pcd:
            if depth and (not color or color in plt.colormaps()):
                look = Look.VERY_HIGH_CONTRAST
            elif not color or color == "pointflow":
                look = Look.VERY_LOW_CONTRAST
    set_output_format(look=look)


def setup_backdrop(
    obj: bproc.types.MeshObject,
    shadow_strength: Strength | str = Strength.MEDIUM,
    transparent: bool | float = True,
    color: Optional[Tuple[float, float, float] | Color | str] = None,
    hdri_path: Optional[Path] = None,
    bg_light: float = 0.15,
    gravity: bool = False,
    offset: Optional[np.ndarray] = None,
):
    """Sets up a backdrop for the given object in the Blender scene.

    This function loads a backdrop object, applies materials and shading, and positions it relative to the given object.
    It also configures transparency and shadow settings for the backdrop. If gravity is enabled, the function sets up
    rigid body physics for the object and the backdrop and simulates the physics to fix their final poses.

    Args:
        obj: The BlenderProc mesh object for which the backdrop is being set up
        shadow_strength: The strength of the shadow to be applied to the backdrop
        transparent: Whether the backdrop should be transparent
        color: The color to apply to the backdrop
        hdri_path: The path to an HDRI image to use as backdrop or to the HAVEN dataset
        bg_light: The intensity of the background light
        gravity: Whether to enable gravity for the object and the backdrop
        offset: The offset to apply to the backdrop's position
    """
    if offset is None:
        offset = np.array([0, 0, -0.05])

    if hdri_path:
        resolved_hdri = hdri_path
        if (resolved_hdri / "hdri").exists():
            # Allow passing a HAVEN dataset root instead of a single HDR file.
            resolved_hdri = bproc.loader.get_random_world_background_hdr_img_path_from_haven(str(resolved_hdri))
        logger.debug(f"Setting HDRI backdrop to {Path(resolved_hdri).stem}")
        bproc.world.set_world_background_hdr_img(str(resolved_hdri), strength=bg_light)
        if gravity:
            logger.warning("Gravity is not compatible with an HDRI backdrop.")
        return
    bproc.renderer.set_world_background([1, 1, 1], strength=bg_light)

    with stdout_redirected():
        plane = bproc.loader.load_obj("./backdrop.ply")[0]
    plane.clear_materials()
    material = plane.new_material("backdrop_material")
    material.set_principled_shader_value("Base Color", [*get_color(color), 1])
    material.set_principled_shader_value("Roughness", 1.0)
    material.set_principled_shader_value("Alpha", shadow_strength.value)
    plane.set_shading_mode("SMOOTH")
    plane.set_location(np.array([0, 0, obj.get_bound_box()[:, 2].min()]) + offset)
    if transparent:
        plane.blender_obj.is_shadow_catcher = True
        if not isinstance(transparent, bool):
            tex_coord_node = material.new_node("ShaderNodeTexCoord")
            tex_coord_node.object = obj.blender_obj
            tex_gradient_node = material.new_node("ShaderNodeTexGradient")
            tex_gradient_node.gradient_type = "SPHERICAL"
            material.link(tex_coord_node.outputs["Object"], tex_gradient_node.inputs["Vector"])
            val_to_rgb_node = material.new_node("ShaderNodeValToRGB")
            val_to_rgb_node.color_ramp.elements[1].position = transparent
            material.link(tex_gradient_node.outputs["Color"], val_to_rgb_node.inputs["Fac"])
            math_node = material.new_node("ShaderNodeMath")
            math_node.operation = "MULTIPLY"
            math_node.inputs[1].default_value = shadow_strength.value
            material.link(val_to_rgb_node.outputs["Color"], math_node.inputs[0])
            material.set_principled_shader_value("Alpha", math_node.outputs["Value"])
    if gravity:
        obj.enable_rigidbody(active=True)
        plane.enable_rigidbody(active=False, collision_shape="MESH")
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=4, max_simulation_time=20, check_object_interval=1
        )


def make_obj(mesh_or_pcd: Trimesh | PointCloud) -> bproc.types.MeshObject:
    """Creates a BlenderProc mesh object from a Trimesh or PointCloud.

    This function initializes a BlenderProc mesh object with the provided
    Trimesh or PointCloud data. It sets up the mesh data, validates it,
    and persists the transformation into the mesh.

    Args:
        mesh_or_pcd: The Trimesh or PointCloud data to create the mesh object from.

    Returns:
        The created BlenderProc mesh object.
    """
    obj = bproc.object.create_with_empty_mesh("mesh" if hasattr(mesh_or_pcd, "faces") else "pointcloud")
    obj.get_mesh().from_pydata(mesh_or_pcd.vertices, [], getattr(mesh_or_pcd, "faces", []))
    obj.get_mesh().validate()
    obj.persist_transformation_into_mesh()
    return obj


def rotate_obj(
    obj: bproc.types.MeshObject,
    rotate: Tuple[float, float, float],
    frame: Optional[int] = None,
    persistent: bool = False,
):
    """Rotates a BlenderProc mesh object.

    This function sets the rotation of the given BlenderProc mesh object to the specified Euler angles.
    If the `persistent` flag is set to True, the transformation is persisted into the mesh data.

    Args:
        obj: The BlenderProc mesh object to rotate.
        rotate: A tuple of three floats representing the rotation angles in degrees for the X, Y, and Z axes.
        frame: The frame number at which to set the rotation. If None, the rotation is applied immediately.
        persistent: If True, the transformation is persisted into the mesh data.
    """
    obj.set_rotation_euler(rotation_euler=[np.deg2rad(r) for r in rotate], frame=frame)
    if persistent:
        obj.persist_transformation_into_mesh()


def init_mesh(obj: bproc.types.MeshObject, shade: Shading | str = Shading.FLAT):
    """Initializes the shading mode for a BlenderProc mesh object.

    This function sets the shading mode of the given BlenderProc mesh object based on the specified shading type.
    If the shading type is set to 'AUTO', an auto-smooth modifier is added to the object. Otherwise, the shading
    mode is set to the specified shading type.

    Args:
        obj: The BlenderProc mesh object to initialize.
        shade: The shading type to apply to the object. Can be 'FLAT', 'SMOOTH', or 'AUTO'.
    """
    logger.debug(f"Setting shading mode to {Shading(shade).name}")
    if Shading(shade) is Shading.AUTO:
        obj.add_auto_smooth_modifier()
    obj.set_shading_mode(Shading(shade).name)


def _pointcloud_with_geometry_nodes(links, nodes, set_material_node=None, point_size: float = 0.004):
    """Sets up a point cloud using geometry nodes.

    This function configures a point cloud in Blender using geometry nodes. It converts a mesh to points
    and optionally sets a material for the points.

    Args:
        links: The links between geometry nodes.
        nodes: The geometry nodes to be used.
        set_material_node: An optional node to set the material for the points.
        point_size: The size of the points in the point cloud.
    """
    group_input_node = Utility.get_the_one_node_with_type(nodes, "NodeGroupInput")
    mesh_to_points_node = nodes.new(type="GeometryNodeMeshToPoints")
    mesh_to_points_node.inputs["Radius"].default_value = point_size
    links.new(group_input_node.outputs["Geometry"], mesh_to_points_node.inputs["Mesh"])

    group_output_node = Utility.get_the_one_node_with_type(nodes, "NodeGroupOutput")
    if set_material_node is None:
        links.new(mesh_to_points_node.outputs["Points"], group_output_node.inputs["Geometry"])
    else:
        links.new(mesh_to_points_node.outputs["Points"], set_material_node.inputs["Geometry"])
        links.new(set_material_node.outputs["Geometry"], group_output_node.inputs["Geometry"])


def _pointcloud_with_geometry_nodes_and_instances(
    links,
    nodes,
    set_material_node=None,
    point_size: float = 0.004,
    point_shape: Shape | str = Shape.SPHERE,
):
    """Sets up a point cloud using geometry nodes and instances.

    This function configures a point cloud in Blender using geometry nodes and instances. It converts a mesh to points
    and optionally sets a material for the points. The shape of the points can be specified as a sphere, cube, or diamond.

    Args:
        links: The links between geometry nodes.
        nodes: The geometry nodes to be used.
        set_material_node: An optional node to set the material for the points.
        point_size: The size of the points in the point cloud.
        point_shape: The shape of the points in the point cloud.
    """
    group_input_node = Utility.get_the_one_node_with_type(nodes, "NodeGroupInput")
    instance_on_points_node = nodes.new("GeometryNodeInstanceOnPoints")
    if Shape(point_shape) is Shape.SPHERE:
        mesh_node = nodes.new("GeometryNodeMeshUVSphere")
        mesh_node.inputs["Radius"].default_value = point_size
        set_shade_smooth_node = nodes.new("GeometryNodeSetShadeSmooth")
        links.new(mesh_node.outputs["Mesh"], set_shade_smooth_node.inputs["Geometry"])
        links.new(set_shade_smooth_node.outputs["Geometry"], instance_on_points_node.inputs["Instance"])
    elif Shape(point_shape) is Shape.CUBE:
        mesh_node = nodes.new("GeometryNodeMeshCube")
        mesh_node.inputs["Size"].default_value = [np.sqrt(2 * point_size**2)] * 3
        links.new(mesh_node.outputs["Mesh"], instance_on_points_node.inputs["Instance"])
    elif Shape(point_shape) is Shape.DIAMOND:
        mesh_node = nodes.new("GeometryNodeMeshIcoSphere")
        mesh_node.inputs["Radius"].default_value = point_size
        mesh_node.inputs["Subdivisions"].default_value = 1
        links.new(mesh_node.outputs["Mesh"], instance_on_points_node.inputs["Instance"])
    else:
        raise ValueError(f"Invalid point shape: {point_shape}")
    links.new(group_input_node.outputs["Geometry"], instance_on_points_node.inputs["Points"])

    group_output_node = Utility.get_the_one_node_with_type(nodes, "NodeGroupOutput")
    if set_material_node is None:
        links.new(instance_on_points_node.outputs["Instances"], group_output_node.inputs["Geometry"])
    else:
        links.new(instance_on_points_node.outputs["Instances"], set_material_node.inputs["Geometry"])
        links.new(set_material_node.outputs["Geometry"], group_output_node.inputs["Geometry"])


def _pointcloud_with_particle_system(
    obj: bproc.types.MeshObject,
    point_size: Optional[float] = 0.004,
    point_shape: Optional[Shape | str] = Shape.SPHERE,
):
    """Sets up a point cloud using a particle system.

    This function configures a point cloud in Blender using a particle system. It creates an instance
    of the specified shape (sphere, cube, or diamond) and sets up the particle system to emit from
    the vertices of the given object.

    Args:
        obj: The BlenderProc mesh object to which the particle system will be applied.
        point_size: The size of the points in the point cloud.
        point_shape: The shape of the points in the point cloud.
    """
    if point_size is None:
        point_size = 0.004
    if point_shape is None:
        point_shape = Shape.SPHERE

    if Shape(point_shape) is Shape.SPHERE:
        instance = bproc.object.create_primitive("SPHERE", radius=point_size)
        instance.set_shading_mode("SMOOTH")
    elif Shape(point_shape) is Shape.CUBE:
        instance = bproc.object.create_primitive("CUBE", size=np.sqrt(2 * point_size**2))
        instance.set_shading_mode("FLAT")
    elif Shape(point_shape) is Shape.DIAMOND:
        instance = create_icoshpere(radius=point_size)
        instance.set_shading_mode("FLAT")
    else:
        raise ValueError(f"Invalid point shape: {point_shape}")
    instance.add_material(obj.get_materials()[0])
    instance.hide()

    obj.add_modifier("PARTICLE_SYSTEM")
    settings = obj.blender_obj.modifiers[-1].particle_system.settings
    settings.count = len(obj.get_mesh().vertices)
    settings.particle_size = 1
    settings.frame_end = 0
    settings.lifetime = 9999
    settings.physics_type = "NO"
    settings.emit_from = "VERT"
    settings.use_emit_random = False
    settings.render_type = "OBJECT"
    settings.instance_object = instance.blender_obj


def init_pointcloud(
    obj: bproc.types.MeshObject,
    point_size: Optional[float] = None,
    point_shape: Optional[Shape | str] = None,
    use_instance: bool = False,
    use_particle_system: bool = False,
    subsample: Optional[int | float] = None,
    subsample_method: Literal["random", "fps"] = "random",
):
    """Initializes a point cloud in BlenderProc.

    This function sets up a point cloud for a given BlenderProc mesh object. It estimates the point size if not provided,
    and configures the point cloud using either geometry nodes, instances, or a particle system based on the specified options.

    Args:
        obj: The BlenderProc mesh object to initialize as a point cloud.
        point_size: The size of the points in the point cloud. If None, it is estimated based on the nearest neighbor distance.
        point_shape: The shape of the points in the point cloud. If specified, instances are used.
        use_instance: Whether to use instances for the point cloud.
        use_particle_system: Whether to use a particle system for the point cloud.
    """
    if point_size is None:
        logger.debug("Estimating point size based on nearest neighbor distance")
        points = obj.mesh_as_trimesh().vertices
        distances, _ = KDTree(points).query(points, k=2)
        nearest_neighbor_distances = distances[:, 1]
        point_size = np.quantile(nearest_neighbor_distances, 0.8)
        logger.debug(f"Using point size: {point_size}")
    if point_shape is not None:
        use_instance = True

    if use_particle_system:
        logger.debug("Point cloud setup with particle system")
        _pointcloud_with_particle_system(obj=obj, point_size=point_size, point_shape=point_shape)
        return

    node_group = obj.add_geometry_nodes()
    links = node_group.links
    nodes = node_group.nodes

    set_material_node = None
    if obj.get_materials():
        set_material_node = nodes.new(type="GeometryNodeSetMaterial")
        set_material_node.inputs["Material"].default_value = obj.get_materials()[0].blender_obj

    if use_instance:
        logger.debug("Point cloud setup with geometry nodes and instances")
        _pointcloud_with_geometry_nodes_and_instances(
            links=links,
            nodes=nodes,
            set_material_node=set_material_node,
            point_size=point_size or 0.004,  # type: ignore[arg-type]
            point_shape=point_shape or Shape.SPHERE,  # type: ignore[arg-type]
        )
    else:
        logger.debug("Point cloud setup with geometry nodes")
        _pointcloud_with_geometry_nodes(
            links=links,
            nodes=nodes,
            set_material_node=set_material_node,
            point_size=point_size or 0.004,  # type: ignore[arg-type]
        )


def add_ambient_occlusion(obj: Optional[bproc.types.MeshObject] = None, distance: float = 0.2, strength: float = 0.5):
    """Adds ambient occlusion to the Blender scene or a specific object.

    If no object is provided, it configures the scene to use ambient occlusion
    with the specified distance and strength. If an object is provided, it adds
    an ambient occlusion shader node to the object's material.

    Args:
        obj: The BlenderProc mesh object to which ambient occlusion will be applied.
        distance: The distance for the ambient occlusion effect.
        strength: The strength of the ambient occlusion effect.
    """
    if obj is None:
        bpy.context.scene.view_layers["ViewLayer"].use_pass_ambient_occlusion = True
        bpy.context.scene.world.light_settings.distance = distance
        bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes
        render_layers_node = Utility.get_the_one_node_with_type(nodes, "CompositorNodeRLayers")
        mix_rgb_node = nodes.new("CompositorNodeMixRGB")
        mix_rgb_node.blend_type = "MULTIPLY"
        mix_rgb_node.inputs[0].default_value = strength

        links = bpy.context.scene.node_tree.links
        try:
            denoise_node = Utility.get_the_one_node_with_type(nodes, "CompositorNodeDenoise")
            Utility.insert_node_instead_existing_link(
                links,
                render_layers_node.outputs["Image"],
                mix_rgb_node.inputs["Image"],
                mix_rgb_node.outputs["Image"],
                denoise_node.inputs["Image"],
            )
        except RuntimeError:
            composite_node = Utility.get_the_one_node_with_type(nodes, "CompositorNodeComposite")
            Utility.insert_node_instead_existing_link(
                links,
                render_layers_node.outputs["Image"],
                mix_rgb_node.inputs["Image"],
                mix_rgb_node.outputs["Image"],
                composite_node.inputs["Image"],
            )
        links.new(render_layers_node.outputs["AO"], mix_rgb_node.inputs[2])
    else:
        material = obj.get_materials()[0]
        ao_node = material.new_node("ShaderNodeAmbientOcclusion")
        ao_node.inputs["Distance"].default_value = distance * strength
        ao_node.samples = 8
        ao_node.only_local = True
        try:
            attribute_node = material.get_the_one_node_with_type("ShaderNodeAttribute")
            bsdf_node = material.get_the_one_node_with_type("ShaderNodeBsdfPrincipled")
            material.insert_node_instead_existing_link(
                attribute_node.outputs["Color"],
                ao_node.inputs["Color"],
                ao_node.outputs["Color"],
                bsdf_node.inputs["Base Color"],
            )
        except RuntimeError:
            ao_node.inputs["Color"].default_value = material.get_principled_shader_value("Base Color")
            material.set_principled_shader_value("Base Color", ao_node.outputs["Color"])


def make_lights(
    obj: bproc.types.MeshObject,
    shadow: Shadow | str = Shadow.MEDIUM,
    light_intensity: float = 0.2,
    fill_light: bool = False,
    rim_light: bool = False,
):
    """Sets up lighting for the given object in the Blender scene.

    This function configures the lighting environment for a Blender scene, including the key light, fill light, and rim light.
    It sets the world background, positions the lights, and adjusts their properties based on the provided parameters.

    Args:
        obj: The BlenderProc mesh object for which the lighting is being set up
        shadow: The type of shadow to apply to the key light
        light_intensity: The intensity of the key light
        fill_light: Whether to add a fill light to the scene
        rim_light: Whether to add a rim light to the scene
    """
    key_light = bproc.types.Light("AREA", name="key_light")
    key_light.set_location([1, 0.5, 2])
    key_light.set_rotation_mat(bproc.camera.rotation_from_forward_vec(obj.get_location() - key_light.get_location()))
    scale = 1
    if Shadow(shadow) is Shadow.VERY_HARD:
        scale = 0.01
    elif Shadow(shadow) is Shadow.HARD:
        scale = 0.1
    elif Shadow(shadow) is Shadow.SOFT:
        scale = 3
    elif Shadow(shadow) is Shadow.VERY_SOFT:
        scale = 5
    key_light.set_scale([scale] * 3)
    key_light.set_energy(int(150 * light_intensity))

    if fill_light:
        fill_light_obj = bproc.types.Light(name="fill_light")
        fill_light_obj.set_location([0, -2, 0])
        fill_light_obj.set_energy(int(10 * light_intensity))
        fill_light_obj.blender_obj.data.use_shadow = False

    if rim_light:
        rim_light_obj = bproc.types.Light(name="rim_light")
        rim_light_obj.set_location([-1, 0, 0.4])
        rim_light_obj.set_energy(int(10 * light_intensity))
        rim_light_obj.blender_obj.data.use_shadow = False


def make_camera(
    obj: bproc.types.MeshObject,
    location: Tuple[float, float, float] | np.ndarray = (1.5, 0, 1),
    offset: Optional[Tuple[float, float, float] | np.ndarray] = None,
    fstop: Optional[float] = None,
):
    """Sets up the camera in the Blender scene.

    Args:
        obj: The BlenderProc mesh object to focus the camera on.
        location: The location of the camera in the scene.
        offset: The offset to apply to the camera's position.
        fstop: The f-stop value for depth of field.

    """
    if offset is None:
        offset = [0, 0, 0]
    location = np.array(location)
    offset = np.array(offset)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() + offset - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    # Ensure only a single camera pose is present; clear any default keyframes and
    # render exactly one frame to avoid duplicated renders.
    camera = bpy.context.scene.camera
    if camera and camera.animation_data:
        camera.animation_data_clear()
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    bproc.camera.add_camera_pose(cam2world_matrix, frame=bpy.context.scene.frame_start)
    bpy.context.scene.frame_end = bpy.context.scene.frame_start + 1
    if fstop:
        focal_distance = np.linalg.norm(obj.mesh_as_trimesh().vertices - location, axis=1).min()
        bproc.camera.add_depth_of_field(focal_point_obj=None, fstop_value=fstop, focal_distance=focal_distance)


def normalize(values: np.ndarray, a: float = 0, b: float = 1) -> np.ndarray:
    min_val = values.min()
    max_val = values.max()
    if min_val == max_val:
        return (values / max_val) * a
    return (values - min_val) / (max_val - min_val) * (b - a) + a


def _target_count(n: int, count_or_fraction: int | float) -> int:
    """Compute target sample size from absolute count or fraction.

    - If 0 < float <= 1: use ceil(n * fraction)
    - Else: treat as absolute count
    Clip result to [1, n].
    """
    if isinstance(count_or_fraction, float):
        if 0 < count_or_fraction <= 1:
            m = int(math.ceil(n * count_or_fraction))
        else:
            m = int(count_or_fraction)
    else:
        m = int(count_or_fraction)
    return int(max(1, min(m, n)))


def _subsample_indices(
    points: np.ndarray, count_or_fraction: int | float, method: Literal["random", "fps"] = "random"
) -> np.ndarray:
    """Return indices to subsample points by the chosen method.

    Args:
        points: (N, 3) array of point positions
        count_or_fraction: desired number of points (int) or fraction (0,1]
        method: 'random' for uniform random, 'fps' for farthest point sampling

    Returns:
        (M,) integer array of selected indices, where M = min(target, N)
    """
    n = int(points.shape[0])
    if n == 0:
        return np.array([], dtype=int)
    m = _target_count(n, count_or_fraction)
    if m >= n:
        return np.arange(n, dtype=int)
    if method == "random":
        return np.random.permutation(n)[:m]

    # Farthest Point Sampling (greedy, O(N*M))
    # Initialize with a random point for robustness
    first = int(np.random.randint(0, n))
    selected = np.empty(m, dtype=int)
    selected[0] = first
    # Maintain distances to the nearest selected point
    dists = np.full(n, np.inf, dtype=np.float64)
    # Update distances with the first selection
    diff = points - points[first]
    dists = np.minimum(dists, np.einsum("ij,ij->i", diff, diff))
    for i in range(1, m):
        idx = int(np.argmax(dists))
        selected[i] = idx
        diff = points - points[idx]
        d2 = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, d2)
    return selected


def depth_to_image(depth: np.ndarray, cmap_name: str = "Greys_r") -> Image.Image:
    empty_mask = depth == 0
    depth[~empty_mask] = normalize(depth[~empty_mask], 0.1, 1)
    cmap = plt.get_cmap(cmap_name)
    depth = cmap(depth)
    depth[empty_mask] = 0
    depth[~empty_mask] *= 255
    return Image.fromarray(depth.astype(np.uint8))


def render_depth(
    obj: bproc.types.MeshObject,
    pcd: bool | int = False,
    keep_mesh: bool = False,
    color: Optional[Tuple[float, float, float] | Color | str] = None,
    cam_location: Tuple[float, float, float] = (1.5, 0, 1),
    roughness: Optional[float] = None,
    point_shape: Optional[Shape | str] = None,
    point_size: Optional[float] = None,
    subsample: Optional[int | float] = None,
    subsample_method: Literal["random", "fps"] = "random",
    noise: float = 0.002,
    bg_color: Optional[Tuple[float, float, float] | Color | str] = None,
    ray_trace: bool = True,
    save: Optional[Path] = None,
    show: bool = False,
) -> Optional[bproc.types.MeshObject]:
    """Renders a depth map of the given object and optionally converts it to a point cloud.

    This function renders a depth map of the provided mesh object using ray tracing. It can either
    return the depth map directly or convert it into a point cloud by sampling points from the
    depth information. Various parameters control the appearance and processing of the output.

    Args:
        obj: The BlenderProc mesh object to render the depth map from
        pcd: Whether to convert depth to point cloud and sample points (if int, specifies number of points)
        keep_mesh: Whether to keep the original mesh when converting to point cloud
        color: Color to apply to the generated point cloud
        cam_location: Position of the camera in world space
        roughness: Material roughness value for the point cloud
        point_shape: Shape to use for point cloud visualization
        point_size: Size of points in the point cloud
        subsample: Optional point-cloud subsampling target (count or fraction) for depth renders
        subsample_method: Subsampling strategy ('random' or 'fps') when reducing depth point clouds
        noise: Amount of random noise to add to point positions
        bg_color: Background color for the rendered depth map
        ray_trace: Whether to use ray tracing for depth map rendering
        save: Path to save the rendered depth map as an image file
        show: Whether to display the rendered depth map

    Returns:
        The generated point cloud object if pcd=True, None if depth map only or input is point cloud

    Raises:
        NotImplementedError: When attempting to render depth map without converting to point cloud
    """
    is_mesh = len(obj.get_mesh().polygons) > 0
    if not is_mesh:
        logger.warning("Depth rendering not supported for point clouds.")
        return

    def get_depth() -> np.ndarray:
        if ray_trace:
            bvh_tree = bproc.object.create_bvh_tree_multi_objects([obj])
            return bproc.camera.depth_via_raytracing(bvh_tree)
        else:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
            data = bproc.renderer.render(load_keys={"depth"})
            depth = data["depth"][0]
            depth[depth == depth.max()] = np.nan
            return depth

    if pcd:
        mesh = obj
        resolution = get_camera_resolution()

        # Temporarily render depth at low resolution to keep sampling fast, then restore.
        bproc.camera.set_resolution(128, 128)
        depth = get_depth()
        points = bproc.camera.pointcloud_from_depth(depth).reshape(-1, 3)
        points = points[~np.isnan(points).any(axis=1)]

        if subsample is not None and subsample > 0:
            idx = _subsample_indices(points, subsample, subsample_method)
        elif isinstance(pcd, int) and pcd > 1:
            target = min(len(points), pcd)
            idx = _subsample_indices(points, target, subsample_method)
        else:
            target = min(len(points), 2048)
            idx = _subsample_indices(points, target, subsample_method)
        points = points[idx]

        pcd = PointCloud(points + np.random.randn(*points.shape) * noise)
        obj = make_obj(mesh_or_pcd=pcd)

        material = obj.new_material("pointcloud_material")
        material.set_principled_shader_value("Roughness", roughness or 0.9)
        set_color(
            obj=obj,
            color=color or "plasma_r",
            camera_location=cam_location,
            instancer=point_shape is not None,
        )
        init_pointcloud(
            obj=obj,
            point_size=point_size,
            point_shape=point_shape,
            subsample=subsample,
            subsample_method=subsample_method,
        )
        bproc.camera.set_resolution(*resolution)
        if keep_mesh:
            obj.set_parent(mesh)
            return mesh
        mesh.delete()
        return obj

    depth = get_depth()
    depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
    cmap_name = color if isinstance(color, str) else "plasma_r"
    image = depth_to_image(depth=depth, cmap_name=cmap_name)
    if bg_color:
        image = set_background_color(image, bg_color)
    if save:
        image.save(Path(save).resolve())
    if show:
        image.show()


def create_mp4_with_ffmpeg(image_folder: Path, output_path: Path, fps: int = 20):
    """Creates an MP4 video from a sequence of images using FFmpeg.

    This function takes a folder containing image files, sorts them, and uses FFmpeg to create an MP4 video
    with the specified frames per second (fps). The resulting video is saved to the specified output path.

    Args:
        image_folder: The folder containing the image files.
        output_path: The path where the output MP4 video will be saved.
        fps: The frames per second for the output video.
    """
    image_files: List[Path] = sorted(image_folder.glob("*.png"))
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    ffmpeg_command = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        str(image_folder / "image_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    subprocess.run(ffmpeg_command, check=True)


def make_animation(
    obj: bproc.types.MeshObject,
    animation: Animation,
    frames: int = 72,
    fps: int = 20,
    save: Path = Path("animation.gif"),
    bg_color: Optional[Tuple[float, float, float] | Color | str] = None,
    debug: bool = False,
):
    """Creates an animation of the given object.

    This function sets up and renders an animation for the specified object using BlenderProc.
    It supports different types of animations such as turning, swiveling, and tumbling.
    The animation can be saved as a GIF or MP4 file, and various rendering options can be configured.

    Args:
        obj: The BlenderProc mesh object to animate.
        animation: The type of animation to perform.
        frames: The number of frames in the animation.
        fps: The frames per second for the animation.
        save: The path where the animation will be saved.
        bg_color: The background color for the animation.
        debug: If True, the function runs in debug mode without rendering the final animation.
    """
    bproc.utility.set_keyframe_render_interval(frame_end=frames)
    if animation in [Animation.TURN, Animation.SWIVEL]:
        pivot = obj
        if animation is Animation.SWIVEL:
            pivot = bproc.object.create_empty("pivot")
            pivot.set_location(obj.get_location())
            # Parent camera and lights to pivot so they orbit with the swivel instead of staying static.
            get_camera().set_parent(pivot)
            for light in get_all_light_objects():
                light.set_parent(pivot)
        for frame, angle in enumerate(np.linspace(0, 360 - 360 // frames, frames)):
            rotate_obj(
                obj=pivot,
                rotate=(0, 0, -angle if animation is Animation.TURN else angle),
                frame=frame,
            )
    elif animation is Animation.TUMBLE:
        obj.enable_rigidbody(active=True)
        obj.set_rotation_euler([0, 0, 0], frame=1)
        obj.get_rigidbody().kinematic = True
        obj.get_rigidbody().keyframe_insert(data_path="kinematic", frame=1)

        obj.set_rotation_euler(
            [np.deg2rad(angle) for angle in np.random.uniform(-10, 10, 3)],
            frame=4,
        )
        obj.get_rigidbody().kinematic = False
        obj.get_rigidbody().keyframe_insert(data_path="kinematic", frame=4)

        bake_physics(frames_to_bake=frames, gravity=False)

    if not debug:
        save = Path(save).resolve()
        images = render_color(
            bg_color=bg_color or Color.WHITE if save.suffix == ".gif" else bg_color,
        )
        images[0].save(save.with_suffix(".png"))

        if save.suffix == ".gif":
            images[0].save(
                save,
                save_all=True,
                append_images=images[1:],
                format="GIF",
                duration=1000 // fps,
                loop=0,
            )
        elif save.suffix == ".mp4":
            image_folder = save.parent / "images"
            image_folder.mkdir(exist_ok=True)
            for i, image in enumerate(images):
                image.save(image_folder / f"image_{i:04d}.png")
            create_mp4_with_ffmpeg(image_folder=image_folder, output_path=save, fps=fps)
            for image in image_folder.glob("*.png"):
                image.unlink()
            image_folder.rmdir()


def render_color(
    bg_color: Optional[Tuple[float, float, float] | Color | str] = None,
    save: Optional[Path] = None,
    show: bool = False,
    progress: bool = True,
) -> List[Image]:
    """Renders the scene and returns a list of images.

    This function renders the current BlenderProc scene and returns a list of images.
    It supports optional background color, saving the images to disk or displaying the images.

    Args:
        bg_color: Optional background color for the images.
        save: Optional path to save the rendered images.
        show: Whether to display the rendered images.
        progress: Whether to show the rendering progress.

    Returns:
        A list of rendered images.
    """
    with stdout_redirected(enabled=not progress):
        data = bproc.renderer.render()

    images: List[Image] = list()
    images_np: List[np.ndarray] = data["colors"]
    for image_np in images_np:
        image = Image.fromarray(image_np.astype(np.uint8))
        if bg_color:
            image = set_background_color(image, bg_color)
        if save:
            image.save(Path(save).resolve())
        if show:
            image.show()
        images.append(image)
    return images
