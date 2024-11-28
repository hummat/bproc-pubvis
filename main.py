import blenderproc as bproc
from blenderproc.python.types import MeshObjectUtility

from typing import Iterable, Tuple, Literal, Optional
from pathlib import Path
import fire
from PIL import Image
import numpy as np
import trimesh
import matplotlib.pyplot as plt

COLORS = {'pale_blue': (0.2, 0.3, 1)}


def run(obj_path: str,
        normalize: bool = True,
        rotate: Optional[Tuple[float, float, float]] = None,
        shade: Literal['flat', 'smooth', 'auto'] = 'flat',
        set_material: bool = True,
        color: Optional[Tuple[float, float, float] | Literal['pale_blue'] | str] = None,
        roughness: float = 0.9,
        mesh_as_pcd: bool = False,
        point_size: float = 0.004,
        light: Optional[Literal['simple', 'three_point']] = 'simple',
        cam_location: Iterable[float] = (1.6, 1, 1.2),
        resolution: int | Tuple[int, int] = 512,
        backdrop: bool = True,
        background_light: float = 0.1,
        transparent: bool = True,
        shadow: Optional[Literal['hard', 'medium', 'soft', 'none']] = None,
        ambient_occlusion: Optional[float] = None,
        engine: Literal['cycles, eevee'] = 'cycles',
        noise_threshold: float = 0.01,
        samples: int = 100,
        show: bool = True):
    init_renderer(resolution=resolution,
                  transparent=transparent,
                  engine=engine,
                  noise_threshold=noise_threshold,
                  samples=samples)

    data = load_data(obj_path=obj_path,
                     normalize=normalize,
                     rotate=rotate,
                     mesh_as_pcd=mesh_as_pcd)
    obj = make_obj(mesh_or_pcd=data)

    if set_material or not obj.get_materials():
        obj.new_material('obj_material')
    modify_material(obj=obj,
                    color=color or ('pale_blue' if isinstance(data, trimesh.Trimesh) else 'pointflow'),
                    roughness=roughness)

    if isinstance(data, trimesh.Trimesh):
        init_mesh(obj=obj,
                  shade=shade)
    elif isinstance(data, trimesh.PointCloud):
        init_pointcloud(obj=obj,
                        point_size=point_size,
                        use_instance=False,
                        use_particle_system=False)
    else:
        raise ValueError(f"Invalid object type: {type(data)}")

    if backdrop and not (transparent and shadow == 'none'):
        load_and_init_backdrop(obj=obj,
                               shadow=shadow != 'none',
                               transparent=transparent)

    if ambient_occlusion:
        add_ambient_occlusion(obj=obj,
                              distance=ambient_occlusion or (0.5 if isinstance(data, trimesh.Trimesh) else 0))

    if light:
        make_lights(light_setup=light,
                    shadow=shadow or ('medium' if isinstance(data, trimesh.Trimesh) else 'soft'),
                    background_light=background_light)
    make_camera(cam_location=np.array(cam_location),
                obj=obj)
    render(show=show)


def init_renderer(resolution: int | Tuple[int, int],
                  transparent: bool = True,
                  engine: Literal['cycles, eevee'] = 'cycles',
                  noise_threshold: float = 0.01,
                  samples: int = 100):
    bproc.init()
    bproc.renderer.set_output_format(enable_transparency=transparent,
                                     view_transform='Filmic',
                                     look='Very Low Contrast')
    if engine == 'cycles':
        bproc.renderer.set_noise_threshold(noise_threshold)
        bproc.renderer.set_max_amount_of_samples(samples)
    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    bproc.camera.set_resolution(*resolution)


def load_data(obj_path: Path | str,
              normalize: bool = True,
              rotate: Optional[Tuple[float, float, float]] = None,
              mesh_as_pcd: bool | int = False) -> trimesh.Trimesh | trimesh.PointCloud:
    if Path(obj_path).suffix == '.blend':
        mesh = bproc.loader.load_blend(obj_path, obj_types='mesh')[0].mesh_as_trimesh()
        if not mesh.faces:
            mesh = trimesh.PointCloud(mesh.vertices)
    else:
        mesh = trimesh.load(obj_path, force='mesh')
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.PointCloud)):
        raise TypeError(f"Invalid object type: {type(mesh)}")
    if normalize:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        mesh.apply_scale(1 / mesh.extents.max())
    if rotate is not None:
        for i, angle in enumerate(rotate):
            direction = np.zeros(3)
            direction[i] = 1
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(angle), direction))
    if isinstance(mesh, trimesh.Trimesh) and mesh_as_pcd:
        return trimesh.PointCloud(mesh.sample(mesh_as_pcd if mesh_as_pcd > 1 else 2048))
    return mesh


def set_color(obj: MeshObjectUtility.MeshObject,
              color: Tuple[float, float, float] | Literal['pale_blue'] | str) -> MeshObjectUtility.Material:
    material = obj.get_materials()[0]
    if isinstance(color, str):
        if color in COLORS:
            material.set_principled_shader_value('Base Color', [*COLORS[color], 1])
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
                for value in values:
                    x_values = values[:, 0]
                    x_value_norm = (value[0] - x_values.min()) / (x_values.max() - x_values.min())
                    colors.append(cmap(x_value_norm))

            # Todo: Is this possible with BlenderProc?
            mesh = obj.get_mesh()
            color_attr_name = "point_color"
            mesh.attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')

            color_data = mesh.attributes[color_attr_name].data
            for i, color in enumerate(colors):
                color_data[i].color = color

            attribute_node = material.new_node('ShaderNodeAttribute')
            attribute_node.attribute_name = color_attr_name
            material.set_principled_shader_value('Base Color', attribute_node.outputs['Color'])
    else:
        material.set_principled_shader_value('Base Color', [*color, 1])
    return material


def modify_material(obj: MeshObjectUtility.MeshObject,
                    color: Tuple[float, float, float] | Literal['pale_blue'] | str = COLORS['pale_blue'],
                    roughness: float = 0.9):
    material = set_color(obj=obj, color=color)
    material.set_principled_shader_value('Roughness', roughness)


def load_and_init_backdrop(obj: MeshObjectUtility.MeshObject,
                           shadow: bool = True,
                           transparent: bool = True):
    plane = bproc.loader.load_obj('backdrop.ply')[0]
    plane.clear_materials()
    plane.set_shading_mode('SMOOTH')
    plane.set_location([0, 0, 0.3 + obj.get_bound_box()[:, 2].min()])  # Fixme: 0.45 is hardcoded
    if transparent and shadow:
        plane.blender_obj.is_shadow_catcher = True


def make_obj(mesh_or_pcd: trimesh.Trimesh | trimesh.PointCloud):
    obj = MeshObjectUtility.create_with_empty_mesh(object_name='pointcloud')
    obj.get_mesh().from_pydata(mesh_or_pcd.vertices, [], getattr(mesh_or_pcd, 'faces', []))
    obj.get_mesh().validate()
    return obj


def init_mesh(obj: MeshObjectUtility.MeshObject,
              shade: Literal['flat', 'smooth', 'auto'] = 'flat'):
    if shade == 'auto':
        obj.add_auto_smooth_modifier()
    else:
        obj.set_shading_mode(shade.upper())


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
                                                  point_size: float = 0.004):
    group_input_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupInput')
    mesh_ico_sphere_node = nodes.new('GeometryNodeMeshUVSphere')
    mesh_ico_sphere_node.inputs['Radius'].default_value = point_size
    instance_on_points_node = nodes.new('GeometryNodeInstanceOnPoints')
    set_shade_smooth_node = nodes.new('GeometryNodeSetShadeSmooth')
    links.new(mesh_ico_sphere_node.outputs['Mesh'], set_shade_smooth_node.inputs['Geometry'])
    links.new(set_shade_smooth_node.outputs['Geometry'], instance_on_points_node.inputs['Instance'])
    links.new(group_input_node.outputs['Geometry'], instance_on_points_node.inputs['Points'])

    group_output_node = MeshObjectUtility.Utility.get_the_one_node_with_type(nodes, 'NodeGroupOutput')
    if set_material_node is None:
        links.new(instance_on_points_node.outputs['Instances'], group_output_node.inputs['Geometry'])
    else:
        links.new(instance_on_points_node.outputs['Instances'], set_material_node.inputs['Geometry'])
        links.new(set_material_node.outputs['Geometry'], group_output_node.inputs['Geometry'])


def _pointcloud_with_particle_system(obj: MeshObjectUtility.MeshObject,
                                     point_size: float = 0.004):
    sphere = MeshObjectUtility.create_primitive('SPHERE', radius=point_size)
    sphere.add_material(obj.get_materials()[0])
    sphere.set_shading_mode('SMOOTH')
    sphere.hide()

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
    settings.instance_object = sphere.blender_obj


def init_pointcloud(obj: MeshObjectUtility.MeshObject,
                    point_size: float = 0.004,
                    use_instance: bool = False,
                    use_particle_system: bool = False):
    if use_particle_system:
        _pointcloud_with_particle_system(obj, point_size)
        return obj

    node_group = obj.add_geometry_nodes()
    links = node_group.links
    nodes = node_group.nodes

    set_material_node = None
    if obj.get_materials():
        set_material_node = nodes.new(type='GeometryNodeSetMaterial')
        set_material_node.inputs['Material'].default_value = obj.get_materials()[0].blender_obj

    if use_instance:
        _pointcloud_with_geometry_nodes_and_instances(links, nodes, set_material_node, point_size)
        return obj
    _pointcloud_with_geometry_nodes(links, nodes, set_material_node, point_size)


def add_ambient_occlusion(obj: MeshObjectUtility.MeshObject,
                          distance: float = 0.5):
    material = obj.get_materials()[0]
    ao_node = material.new_node('ShaderNodeAmbientOcclusion')
    ao_node.inputs['Distance'].default_value = distance
    ao_node.samples = 8
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


def make_lights(light_setup: Literal['simple', 'three_point'] = 'simple',
                shadow: Literal['hard', 'medium', 'soft'] = 'medium',
                background_light: float = 0.1):
    bproc.renderer.set_world_background([1, 1, 1], strength=background_light)
    if light_setup == 'simple':
        light = bproc.types.Light()
        light.set_type('AREA')
        light.set_location([0.5, 0, 1.5])
        light.set_rotation_euler([0, np.deg2rad(20), 0])
        light.set_scale([1 if shadow == 'soft' else 0.5 if shadow == 'medium' else 0.05] * 3)
        light.set_energy(50)
    elif light_setup == 'three_point':
        raise NotImplementedError("Three point light setup is not implemented.")
    else:
        raise ValueError(f"Invalid light setup: {light_setup}")


def make_camera(cam_location: np.ndarray,
                obj: MeshObjectUtility.MeshObject):
    poi = bproc.object.compute_poi([obj])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - cam_location)
    cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def render(show: bool):
    data = bproc.renderer.render()
    if show:
        Image.fromarray(data['colors'][0].astype(np.uint8)).show()


fire.Fire(run)
