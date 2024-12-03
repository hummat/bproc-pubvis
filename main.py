import blenderproc as bproc
from blenderproc.python.types import MeshObjectUtility
from blenderproc.python.utility.Utility import Utility

from typing import Tuple, Optional
from pathlib import Path
import random
from enum import Enum
import fire
from PIL import Image
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class Color(Enum):
    WHITE = (1, 1, 1)
    BLACK = (0, 0, 0)
    PALE_VIOLET = (0.342605, 0.313068, 0.496933)
    PALE_TURQUOISE = (0.239975, 0.426978, 0.533277)
    PALE_GREEN = (0.165398, 0.558341, 0.416653)
    BRIGHT_BLUE = (0.0419309, 0.154187, 0.438316)
    PALE_RED = (0.410603, 0.101933, 0.0683599)
    BEIGE = (0.496933, 0.472623, 0.331984)
    WARM_GREY = (0.502887, 0.494328, 0.456411)


class Strength(Enum):
    OFF = 0.0
    WEAK = 0.6
    MEDIUM = 0.7
    STRONG = 1.0


class Shading(Enum):
    FLAT = 'flat'
    SMOOTH = 'smooth'
    AUTO = 'auto'


class Shape(Enum):
    SPHERE = 'sphere'
    CUBE = 'cube'
    DIAMOND = 'diamond'


class Look(Enum):
    VERY_LOW_CONTRAST = 'Very Low Contrast'
    LOW_CONTRAST = 'Low Contrast'
    MEDIUM_CONTRAST = 'Medium Contrast'
    HIGH_CONTRAST = 'High Contrast'
    VERY_HIGH_CONTRAST = 'Very High Contrast'


class Shadow(Enum):
    VERY_HARD = 'very_hard'
    HARD = 'hard'
    MEDIUM = 'medium'
    SOFT = 'soft'
    VERY_SOFT = 'very_soft'
    NONE = 'none'


class Engine(Enum):
    CYCLES = 'cycles'
    EEVEE = 'eevee'


class Primitive(Enum):
    SUZANNE = 'suzanne'
    MONKEY = 'monkey'
    CUBE = 'cube'
    SPHERE = 'sphere'
    CYLINDER = 'cylinder'
    CONE = 'cone'


class Animation(Enum):
    TURN = 'turn'
    SWIVEL = 'swivel'
    TUMBLE = 'tumble'


def run(obj_path: str,
        normalize: bool = True,
        rotate: Optional[Tuple[float, float, float]] = (0, 0, -35),
        gravity: bool = False,
        animate: Optional[Animation | str] = None,
        shade: Shading | str = Shading.FLAT,
        set_material: bool = True,
        color: Optional[Tuple[float, float, float] | Color | str] = None,
        roughness: Optional[float] = None,
        mesh_as_pcd: bool = False,
        point_size: Optional[float] = None,
        point_shape: Shape | str = Shape.SPHERE,
        cam_location: Tuple[float, float, float] = (1.5, 0, 1),
        cam_offset: Tuple[float, float, float] = (0, 0, 0),
        resolution: int | Tuple[int, int] = 512,
        fstop: Optional[float] = None,
        backdrop: bool = True,
        bg_light: float = 0.2,
        bg_color: Optional[Tuple[float, float, float] | Color | str] = None,
        transparent: bool | float = True,
        look: Optional[Look | str] = None,
        exposure: float = 0,
        shadow: Optional[Shadow | str] = None,
        ao: Optional[bool | float] = None,
        engine: Engine | str = Engine.CYCLES,
        noise_threshold: Optional[float] = None,
        samples: Optional[int] = None,
        show: bool = True,
        seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)

    init_renderer(resolution=resolution,
                  transparent=transparent or bg_color is not None,
                  look=look or Look.MEDIUM_CONTRAST,
                  exposure=exposure,
                  engine=engine,
                  noise_threshold=noise_threshold or 0.01,
                  samples=samples or 100)

    data = load_data(obj_path=obj_path,
                     normalize=normalize,
                     mesh_as_pcd=mesh_as_pcd)
    obj = make_obj(mesh_or_pcd=data)

    if set_material or not obj.get_materials():
        obj.new_material('obj_material')
    modify_material(obj=obj,
                    color=color or (Color.BRIGHT_BLUE if isinstance(data, trimesh.Trimesh) else 'pointflow'),
                    camera_location=cam_location,
                    roughness=roughness or (0.5 if isinstance(data, trimesh.Trimesh) else 0.9),
                    instancer=Shape(point_shape) is not Shape.SPHERE)

    if rotate:
        rotate_obj(obj=obj,
                   rotate=rotate)

    if isinstance(data, trimesh.Trimesh):
        init_mesh(obj=obj,
                  shade=shade)
    elif isinstance(data, trimesh.PointCloud):
        init_pointcloud(obj=obj,
                        point_size=point_size,
                        point_shape=point_shape)
        if look is None and (color is None or color == 'pointflow'):
            set_output_format(look=Look.VERY_LOW_CONTRAST)
    else:
        raise ValueError(f"Invalid object type: {type(data)}")

    if gravity or (backdrop and not (transparent and shadow and Shadow(shadow) is Shadow.NONE)):
        shadow_strength = Strength.OFF if shadow and Shadow(shadow) is Shadow.NONE else Strength.MEDIUM
        plane = load_and_init_backdrop(obj=obj,
                                       shadow_strength=shadow_strength,
                                       transparent=transparent)
        if gravity:
            obj.enable_rigidbody(active=True)
            plane.enable_rigidbody(active=False, collision_shape='MESH')
            bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4,
                                                              max_simulation_time=20,
                                                              check_object_interval=1)

    if ao or (ao is None and isinstance(data, trimesh.Trimesh)):
        add_ambient_occlusion(strength=0.5 if isinstance(ao, bool) or ao is None else ao)

    make_lights(obj=obj,
                shadow=shadow or (Shadow.MEDIUM if isinstance(data, trimesh.Trimesh) else Shadow.SOFT),
                background_light=bg_light)
    make_camera(obj=obj,
                location=cam_location,
                offset=cam_offset,
                fstop=fstop)
    if Animation(animate) is Animation.TURN:
        images = list()
        for z in range(0, 360, 10):
            rotate_obj(obj=obj,
                       rotate=(0, 0, z))
            image = render(background_color=bg_color,
                           show=False)
            images.append(image)
        images[0].save(
            './out.gif',
            save_all=True,
            append_images=images[1:],
            format='GIF',
            duration=200,
            loop=1
        )
    else:
        render(background_color=bg_color,
               show=show)


def set_output_format(look: Optional[Look | str] = None,
                      exposure: Optional[float] = None):
    import bpy
    if look is not None:
        bpy.context.scene.view_settings.look = Look(look).value
    if exposure is not None:
        bpy.context.scene.view_settings.exposure = exposure


def apply_modifier(obj: MeshObjectUtility.MeshObject, mame: str):
    import bpy
    with bpy.context.temp_override(object=obj.blender_obj):
        bpy.ops.object.modifier_apply(modifier=mame)


def create_icoshpere(radius: float, subdivisions: int = 1) -> MeshObjectUtility.MeshObject:
    import bpy
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=subdivisions)
    return MeshObjectUtility.MeshObject(bpy.context.object)


def init_renderer(resolution: int | Tuple[int, int],
                  transparent: bool = True,
                  look: Look | str = Look.MEDIUM_CONTRAST,
                  exposure: float = 0,
                  engine: Engine | str = Engine.CYCLES,
                  noise_threshold: float = 0.01,
                  samples: int = 100):
    bproc.init()
    bproc.renderer.set_output_format(enable_transparency=transparent,
                                     view_transform='Filmic')
    set_output_format(look=look,
                      exposure=exposure)
    if Engine(engine) is Engine.CYCLES:
        bproc.renderer.set_denoiser('OPTIX')
        bproc.renderer.set_noise_threshold(noise_threshold)
        bproc.renderer.set_max_amount_of_samples(samples)
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    bproc.camera.set_resolution(*resolution)


def load_data(obj_path: Path | str | Primitive,
              normalize: bool = True,
              mesh_as_pcd: bool | int = False) -> trimesh.Trimesh | trimesh.PointCloud:
    if hasattr(Primitive, obj_path.upper()):
        if Primitive(obj_path) in [Primitive.SUZANNE, Primitive.MONKEY]:
            obj = MeshObjectUtility.create_primitive('MONKEY')
            obj.add_modifier('SUBSURF')
            apply_modifier(obj, 'Subdivision')
        else:
            obj = MeshObjectUtility.create_primitive(obj_path.upper())
            if Primitive(obj_path) is not Primitive.SPHERE:
                obj.add_modifier('BEVEL')
                apply_modifier(obj, 'Bevel')

        mesh = obj.mesh_as_trimesh()
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
        obj.delete()
    elif Path(obj_path).suffix == '.blend':
        # TODO: Fix multi-object and material loading
        objs = bproc.loader.load_blend(obj_path, obj_types='mesh')
        mesh = objs[0].mesh_as_trimesh()
        for obj in objs:
            obj.delete()
        if mesh.faces is None:
            mesh = trimesh.PointCloud(mesh.vertices)
    else:
        mesh = trimesh.load(obj_path, force='mesh')
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.PointCloud)):
        raise TypeError(f"Invalid object type: {type(mesh)}")
    if normalize:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(1 / mesh.extents.max())

    if isinstance(mesh, trimesh.Trimesh) and mesh_as_pcd:
        return trimesh.PointCloud(mesh.sample(mesh_as_pcd if mesh_as_pcd > 1 else 4096))
    return mesh


def set_color(obj: MeshObjectUtility.MeshObject,
              color: Tuple[float, float, float] | Color | str,
              camera_location: Optional[Tuple[float, float, float] | np.ndarray] = None,
              instancer: bool = False) -> MeshObjectUtility.Material:
    material = obj.get_materials()[0]
    if isinstance(color, str):
        if hasattr(Color, color.upper()):
            material.set_principled_shader_value('Base Color', [*Color[color.upper()].value, 1])
        elif color == 'random':
            material.set_principled_shader_value('Base Color', [*np.random.rand(3), 1])
        elif color == 'random_color':
            material.set_principled_shader_value('Base Color',
                                                 [*np.random.choice(list(Color)).value, 1])
        else:
            values = obj.mesh_as_trimesh().vertices
            colors = list()
            if color == 'pointflow':
                def cmap(x, y, z):
                    vec = np.array([x, y, z])
                    vec = np.clip(vec, 0.001, 1.0)
                    norm = np.sqrt(np.sum(vec ** 2))
                    vec /= norm
                    return [vec[0], vec[1], vec[2], 1]

                for value in values:
                    colors.append(cmap(*value))
            else:
                cmap = plt.get_cmap(color)
                distances = np.linalg.norm(values - np.array(camera_location), axis=1)
                distances = (distances - distances.min()) / (distances.max() - distances.min())
                for d in distances:
                    colors.append(cmap(d))

            # TODO: Is this possible with BlenderProc, i.e. obj.new_attribute?
            mesh = obj.get_mesh()
            color_attr_name = "point_color"
            mesh.attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')

            color_data = mesh.attributes[color_attr_name].data
            for i, color in enumerate(colors):
                color_data[i].color = color

            attribute_node = material.new_node('ShaderNodeAttribute')
            attribute_node.attribute_name = color_attr_name
            if instancer:
                attribute_node.attribute_type = 'INSTANCER'
            material.set_principled_shader_value('Base Color', attribute_node.outputs['Color'])
    elif isinstance(color, Color):
        material.set_principled_shader_value('Base Color', [*color.value, 1])
    else:
        material.set_principled_shader_value('Base Color', [*color, 1])
    return material


def modify_material(obj: MeshObjectUtility.MeshObject,
                    color: Tuple[float, float, float] | Color | str = Color.BRIGHT_BLUE,
                    camera_location: Optional[Tuple[float, float, float] | np.ndarray] = None,
                    roughness: float = 0.9,
                    instancer: bool = False):
    material = set_color(obj=obj, color=color, camera_location=camera_location, instancer=instancer)
    material.set_principled_shader_value('Roughness', roughness)


def load_and_init_backdrop(obj: MeshObjectUtility.MeshObject,
                           shadow_strength: Strength | str = Strength.MEDIUM,
                           transparent: bool | float = True,
                           offset: np.ndarray = np.array([0, 0, -0.05])) -> MeshObjectUtility.MeshObject:
    plane = bproc.loader.load_obj('backdrop.ply')[0]
    plane.clear_materials()
    material = plane.new_material('backdrop_material')
    material.set_principled_shader_value('Base Color', [1, 1, 1, 1])
    material.set_principled_shader_value('Roughness', 1.0)
    plane.set_shading_mode('SMOOTH')
    plane.set_location(np.array([0, 0, obj.get_bound_box()[:, 2].min()]) + offset)
    if transparent and Strength(shadow_strength) is not Strength.OFF:
        plane.blender_obj.is_shadow_catcher = True
        material.set_principled_shader_value('Alpha', shadow_strength.value)
        if not isinstance(transparent, bool):
            tex_coord_node = material.new_node('ShaderNodeTexCoord')
            tex_coord_node.object = obj.blender_obj
            tex_gradient_node = material.new_node('ShaderNodeTexGradient')
            tex_gradient_node.gradient_type = 'SPHERICAL'
            material.link(tex_coord_node.outputs['Object'], tex_gradient_node.inputs['Vector'])
            val_to_rgb_node = material.new_node('ShaderNodeValToRGB')
            val_to_rgb_node.color_ramp.elements[1].position = transparent
            material.link(tex_gradient_node.outputs['Color'], val_to_rgb_node.inputs['Fac'])
            math_node = material.new_node('ShaderNodeMath')
            math_node.operation = 'MULTIPLY'
            math_node.inputs[1].default_value = shadow_strength.value
            material.link(val_to_rgb_node.outputs['Color'], math_node.inputs[0])
            material.set_principled_shader_value('Alpha', math_node.outputs['Value'])
    return plane


def make_obj(mesh_or_pcd: trimesh.Trimesh | trimesh.PointCloud) -> MeshObjectUtility.MeshObject:
    obj = MeshObjectUtility.create_with_empty_mesh('mesh' if hasattr(mesh_or_pcd, 'faces') else 'pointcloud')
    obj.get_mesh().from_pydata(mesh_or_pcd.vertices, [], getattr(mesh_or_pcd, 'faces', []))
    obj.get_mesh().validate()
    return obj


def rotate_obj(obj: MeshObjectUtility.MeshObject,
               rotate: Tuple[float, float, float]):
    obj.set_rotation_euler([np.deg2rad(r) for r in rotate])
    obj.persist_transformation_into_mesh()


def init_mesh(obj: MeshObjectUtility.MeshObject,
              shade: Shading | str = Shading.FLAT):
    if Shading(shade) is Shading.AUTO:
        obj.add_auto_smooth_modifier()
    else:
        obj.set_shading_mode(Shading(shade).name)


def _pointcloud_with_geometry_nodes(links,
                                    nodes,
                                    set_material_node=None,
                                    point_size: float = 0.004):
    group_input_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupInput')
    mesh_to_points_node = nodes.new(type='GeometryNodeMeshToPoints')
    mesh_to_points_node.inputs['Radius'].default_value = point_size
    links.new(group_input_node.outputs['Geometry'], mesh_to_points_node.inputs['Mesh'])

    group_output_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupOutput')
    if set_material_node is None:
        links.new(mesh_to_points_node.outputs['Points'], group_output_node.inputs['Geometry'])
    else:
        links.new(mesh_to_points_node.outputs['Points'], set_material_node.inputs['Geometry'])
        links.new(set_material_node.outputs['Geometry'], group_output_node.inputs['Geometry'])


def _pointcloud_with_geometry_nodes_and_instances(links,
                                                  nodes,
                                                  set_material_node=None,
                                                  point_size: float = 0.004,
                                                  point_shape: Shape | str = Shape.SPHERE):
    group_input_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupInput')
    instance_on_points_node = nodes.new('GeometryNodeInstanceOnPoints')
    if Shape(point_shape) is Shape.SPHERE:
        mesh_node = nodes.new('GeometryNodeMeshUVSphere')
        mesh_node.inputs['Radius'].default_value = point_size
        set_shade_smooth_node = nodes.new('GeometryNodeSetShadeSmooth')
        links.new(mesh_node.outputs['Mesh'], set_shade_smooth_node.inputs['Geometry'])
        links.new(set_shade_smooth_node.outputs['Geometry'], instance_on_points_node.inputs['Instance'])
    elif Shape(point_shape) is Shape.CUBE:
        mesh_node = nodes.new('GeometryNodeMeshCube')
        mesh_node.inputs['Size'].default_value = [np.sqrt(2 * point_size ** 2)] * 3
        links.new(mesh_node.outputs['Mesh'], instance_on_points_node.inputs['Instance'])
    elif Shape(point_shape) is Shape.DIAMOND:
        mesh_node = nodes.new('GeometryNodeMeshIcoSphere')
        mesh_node.inputs['Radius'].default_value = point_size
        mesh_node.inputs['Subdivisions'].default_value = 1
        links.new(mesh_node.outputs['Mesh'], instance_on_points_node.inputs['Instance'])
    else:
        raise ValueError(f"Invalid point shape: {point_shape}")
    links.new(group_input_node.outputs['Geometry'], instance_on_points_node.inputs['Points'])

    group_output_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupOutput')
    if set_material_node is None:
        links.new(instance_on_points_node.outputs['Instances'], group_output_node.inputs['Geometry'])
    else:
        links.new(instance_on_points_node.outputs['Instances'], set_material_node.inputs['Geometry'])
        links.new(set_material_node.outputs['Geometry'], group_output_node.inputs['Geometry'])


def _pointcloud_with_particle_system(obj: MeshObjectUtility.MeshObject,
                                     point_size: float = 0.004,
                                     point_shape: Shape | str = Shape.SPHERE):
    if Shape(point_shape) is Shape.SPHERE:
        instance = MeshObjectUtility.create_primitive('SPHERE', radius=point_size)
        instance.set_shading_mode('SMOOTH')
    elif Shape(point_shape) is Shape.CUBE:
        instance = MeshObjectUtility.create_primitive('CUBE', size=np.sqrt(2 * point_size ** 2))
        instance.set_shading_mode('FLAT')
    elif Shape(point_shape) is Shape.DIAMOND:
        instance = create_icoshpere(radius=point_size)
        instance.set_shading_mode('FLAT')
    else:
        raise ValueError(f"Invalid point shape: {point_shape}")
    instance.add_material(obj.get_materials()[0])
    instance.hide()

    obj.add_modifier('PARTICLE_SYSTEM')
    settings = obj.blender_obj.modifiers[-1].particle_system.settings
    settings.count = len(obj.get_mesh().vertices)
    settings.particle_size = 1
    settings.frame_end = 0
    settings.lifetime = 9999
    settings.physics_type = 'NO'
    settings.emit_from = 'VERT'
    settings.use_emit_random = False
    settings.render_type = 'OBJECT'
    settings.instance_object = instance.blender_obj


def init_pointcloud(obj: MeshObjectUtility.MeshObject,
                    point_size: Optional[float] = None,
                    point_shape: Shape | str = Shape.SPHERE,
                    use_instance: bool = False,
                    use_particle_system: bool = False):
    if point_size is None:
        points = obj.mesh_as_trimesh().vertices
        distances, _ = KDTree(points).query(points, k=2)
        nearest_neighbor_distances = distances[:, 1]
        point_size = np.quantile(nearest_neighbor_distances, 0.8)
    if Shape(point_shape) is not Shape.SPHERE:
        use_instance = True

    if use_particle_system:
        _pointcloud_with_particle_system(obj=obj,
                                         point_size=point_size,
                                         point_shape=point_shape)
        return obj

    node_group = obj.add_geometry_nodes()
    links = node_group.links
    nodes = node_group.nodes

    set_material_node = None
    if obj.get_materials():
        set_material_node = nodes.new(type='GeometryNodeSetMaterial')
        set_material_node.inputs['Material'].default_value = obj.get_materials()[0].blender_obj

    if use_instance:
        _pointcloud_with_geometry_nodes_and_instances(links=links,
                                                      nodes=nodes,
                                                      set_material_node=set_material_node,
                                                      point_size=point_size,
                                                      point_shape=point_shape)
        return obj
    _pointcloud_with_geometry_nodes(links=links,
                                    nodes=nodes,
                                    set_material_node=set_material_node,
                                    point_size=point_size)


def add_ambient_occlusion(obj: Optional[MeshObjectUtility.MeshObject] = None,
                          distance: float = 0.2,
                          strength: float = 0.5):
    if obj is None:
        import bpy

        bpy.context.scene.view_layers['ViewLayer'].use_pass_ambient_occlusion = True
        bpy.context.scene.world.light_settings.distance = distance
        bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes
        render_layers_node = Utility.get_the_one_node_with_type(nodes, 'CompositorNodeRLayers')
        mix_rgb_node = nodes.new('CompositorNodeMixRGB')
        mix_rgb_node.blend_type = 'MULTIPLY'
        mix_rgb_node.inputs[0].default_value = strength

        links = bpy.context.scene.node_tree.links
        try:
            denoise_node = Utility.get_the_one_node_with_type(nodes, 'CompositorNodeDenoise')
            Utility.insert_node_instead_existing_link(links,
                                                      render_layers_node.outputs['Image'],
                                                      mix_rgb_node.inputs['Image'],
                                                      mix_rgb_node.outputs['Image'],
                                                      denoise_node.inputs['Image'])
        except RuntimeError:
            composite_node = Utility.get_the_one_node_with_type(nodes, 'CompositorNodeComposite')
            Utility.insert_node_instead_existing_link(links,
                                                      render_layers_node.outputs['Image'],
                                                      mix_rgb_node.inputs['Image'],
                                                      mix_rgb_node.outputs['Image'],
                                                      composite_node.inputs['Image'])
        links.new(render_layers_node.outputs['AO'], mix_rgb_node.inputs[2])
    else:
        material = obj.get_materials()[0]
        ao_node = material.new_node('ShaderNodeAmbientOcclusion')
        ao_node.inputs['Distance'].default_value = distance * strength
        ao_node.samples = 8
        ao_node.only_local = True
        try:
            attribute_node = material.get_the_one_node_with_type('ShaderNodeAttribute')
            bsdf_node = material.get_the_one_node_with_type('ShaderNodeBsdfPrincipled')
            material.insert_node_instead_existing_link(attribute_node.outputs['Color'],
                                                       ao_node.inputs['Color'],
                                                       ao_node.outputs['Color'],
                                                       bsdf_node.inputs['Base Color'])
        except RuntimeError:
            ao_node.inputs['Color'].default_value = material.get_principled_shader_value('Base Color')
            material.set_principled_shader_value('Base Color', ao_node.outputs['Color'])


def make_lights(obj: MeshObjectUtility.MeshObject,
                shadow: Shadow | str = Shadow.MEDIUM,
                background_light: float = 0.2,
                fill_light: bool = False,
                rim_light: bool = False):
    bproc.renderer.set_world_background([1, 1, 1], strength=background_light)

    key_light = bproc.types.Light('AREA', name='key_light')
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
    key_light.set_energy(150)

    if fill_light:
        fill_light = bproc.types.Light(name='fill_light')
        fill_light.set_location([0, -2, 0])
        fill_light.set_energy(10)
        fill_light.blender_obj.data.use_shadow = False

    if rim_light:
        rim_light = bproc.types.Light(name='rim_light')
        rim_light.set_location([-1, 0, 0.4])
        rim_light.set_energy(10)
        rim_light.blender_obj.data.use_shadow = False


def make_camera(obj: MeshObjectUtility.MeshObject,
                location: Tuple[float, float, float] | np.ndarray = (1.5, 0, 1),
                offset: Optional[Tuple[float, float, float] | np.ndarray] = None,
                fstop: Optional[float] = None):
    if offset is None:
        offset = [0, 0, 0]
    location = np.array(location)
    offset = np.array(offset)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() + offset - location)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
    if fstop:
        focal_distance = np.linalg.norm(obj.mesh_as_trimesh().vertices - location, axis=1).min()
        bproc.camera.add_depth_of_field(focal_point_obj=None,
                                        fstop_value=fstop,
                                        focal_distance=focal_distance)


def render(background_color: Optional[Tuple[float, float, float] | Color | str] = None,
           save: Optional[Path | str] = None,
           show: bool = False) -> Image:
    data = bproc.renderer.render()
    images = data['colors']
    if len(images) == 1:
        image = Image.fromarray(images[0].astype(np.uint8))
    else:
        for image in images:
            Image.fromarray(image.astype(np.uint8)).show()

    if background_color:
        if isinstance(background_color, str):
            background_color = Color[background_color.upper()]
        background_color = background_color.value if isinstance(background_color, Color) else background_color
        background = Image.new('RGBA', image.size, tuple(int(c * 255) for c in background_color))
        image = Image.alpha_composite(background, image)

    if save:
        image.save(save)
    if show:
        image.show()

    return image


fire.Fire(run)
