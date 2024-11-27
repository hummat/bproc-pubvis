import blenderproc as bproc
from blenderproc.python.types import MeshObjectUtility

from typing import Iterable, Tuple, Literal, Optional
from pathlib import Path
import fire
from PIL import Image
import numpy as np
import trimesh


def run(obj_path: str,
        normalize: bool = True,
        rotate: Optional[Tuple[float, float, float]] = None,
        shade: Literal['flat', 'smooth', 'auto'] = 'flat',
        set_material: bool = True,
        point_size: float = 0.004,
        cam_location: Iterable[float] = (1.6, 0, 1.2),
        resolution: Tuple[int, int] = (512, 512),
        backdrop: bool = True,
        background_light: float = 0.1,
        transparent: bool = True,
        shadow: bool = True,
        engine: Literal['cycles, eevee'] = 'cycles',
        noise_threshold: float = 0.01,
        samples: int = 100,
        show: bool = True):
    init_renderer(resolution, transparent, engine, noise_threshold, samples)

    data = load_data(obj_path, normalize, rotate)
    if isinstance(data, trimesh.Trimesh):
        obj = make_mesh(data, set_material, shade)
    elif isinstance(data, trimesh.PointCloud):
        obj = make_pointcloud(data.vertices, point_size, set_material, use_instance=False, use_particle_system=True)
    else:
        raise ValueError(f"Invalid object type: {type(data)}")

    if backdrop and not (transparent and not shadow):
        make_backdrop(obj, shadow, transparent)

    make_lights(background_light)
    make_camera(np.array(cam_location), obj)
    render(show)


def init_renderer(resolution: Tuple[int, int],
                  transparent: bool = True,
                  engine: Literal['cycles, eevee'] = 'cycles',
                  noise_threshold: float = 0.01,
                  samples: int = 100):
    bproc.init()
    if transparent:
        bproc.renderer.set_output_format(enable_transparency=True)
    if engine == 'cycles':
        bproc.renderer.set_noise_threshold(noise_threshold)
        bproc.renderer.set_max_amount_of_samples(samples)
    bproc.camera.set_resolution(*resolution)


def load_data(obj_path: Path | str,
              normalize: bool = True,
              rotate: Optional[Tuple[float, float, float]] = None) -> trimesh.Trimesh | trimesh.PointCloud:
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
    return mesh


def make_backdrop(obj: MeshObjectUtility.MeshObject,
                  shadow: bool = True,
                  transparent: bool = True):
    backdrop_path = Path(__file__).parent / 'backdrop.ply'
    plane = bproc.loader.load_obj(str(backdrop_path))[0]
    plane.clear_materials()
    plane.set_shading_mode('SMOOTH')
    plane.set_location([0, 0, obj.get_bound_box()[:, 2].min()])
    if transparent and shadow:
        plane.blender_obj.is_shadow_catcher = True


def make_mesh(mesh: trimesh.Trimesh,
              set_material: bool = True,
              shade: Literal['flat', 'smooth', 'auto'] = 'flat') -> MeshObjectUtility.MeshObject:
    obj = MeshObjectUtility.create_with_empty_mesh(object_name='mesh')
    obj.get_mesh().from_pydata(mesh.vertices, [], mesh.faces)
    obj.get_mesh().validate()
    if set_material:
        material = obj.new_material('mesh_material')
        material.set_principled_shader_value('Roughness', 0.8)
        material.set_principled_shader_value('Base Color', np.ones(4))
    if shade == 'auto':
        obj.add_auto_smooth_modifier()
    else:
        obj.set_shading_mode(shade.upper())
    return obj


def _pointcloud_with_geometry_nodes(links,
                                    nodes,
                                    set_material_node = None,
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
                                                  set_material_node = None,
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
                                     material: Optional[MeshObjectUtility.Material] = None,
                                     point_size: float = 0.004):
    sphere = MeshObjectUtility.create_primitive('SPHERE', radius=point_size)
    sphere.set_shading_mode('SMOOTH')
    if material is not None:
        sphere.add_material(material)
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


def make_pointcloud(points: np.ndarray,
                    point_size: float = 0.004,
                    set_material: bool = True,
                    use_instance: bool = False,
                    use_particle_system: bool = False) -> MeshObjectUtility.MeshObject:
    obj = MeshObjectUtility.create_with_empty_mesh(object_name='pointcloud')
    obj.get_mesh().from_pydata(points, [], [])
    obj.get_mesh().validate()

    material = None
    if set_material:
        material = obj.new_material('pointcloud_material')
        material.set_principled_shader_value('Roughness', 0.2)
        material.set_principled_shader_value('Base Color', [0, 0, 1, 1])

    if use_particle_system:
        _pointcloud_with_particle_system(obj, material, point_size)
        return obj

    node_group = obj.add_geometry_nodes()
    links = node_group.links
    nodes = node_group.nodes

    set_material_node = None
    if set_material:
        set_material_node = nodes.new(type='GeometryNodeSetMaterial')
        set_material_node.inputs['Material'].default_value = material.blender_obj

    if use_instance:
        _pointcloud_with_geometry_nodes_and_instances(links, nodes, set_material_node, point_size)
        return obj
    _pointcloud_with_geometry_nodes(links, nodes, set_material_node, point_size)
    return obj




def make_lights(background_light: float = 0.1):
    bproc.renderer.set_world_background([1, 1, 1], strength=background_light)
    light = bproc.types.Light()
    light.set_type('POINT')
    light.set_location([-1, 0, 1])
    light.set_energy(100)


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
