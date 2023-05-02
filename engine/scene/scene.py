import numpy as np
import moderngl as mgl
import glm
import pygame as pg
from .light import Light
from .camera import Camera
from ..allocated_resources import Resources
from .. import mesh
from .. import vertex_approximator
from .scene_object import SceneObject
import math


class Scene:
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        self.resources:Resources = resources
        self.camera:Camera = camera
        self.camera.position=glm.vec3((0, 4, 7))
        self.light = Light(position=(6, 4, 7))
        self.gl_context:mgl.Context = gl_context
        self.scene_objects:list[SceneObject] = []

    def initiate_resources(self):
        shader = mesh.Shaders(self.gl_context)
        shader.read_from_file(file_name="default")
        self.resources.shaders["default"] = shader

        texture_1 = mesh.Texture(self.gl_context, file_path="textures/test.png")
        self.resources.textures["debuging"] = texture_1
        texture_2 = mesh.Texture(self.gl_context, file_path="textures/img.png")
        self.resources.textures["wood"] = texture_2
        texture_3 = mesh.Texture(self.gl_context, file_path="textures/img_1.png")
        self.resources.textures["iron"] = texture_3
        texture_4 = mesh.Texture(self.gl_context, file_path="textures/img_2.png")
        self.resources.textures["gold"] = texture_4
        
        texture_4 = mesh.Texture(self.gl_context, fill_color="red")
        self.resources.textures["red"] = texture_4

        vertexes = mesh.Vertexes(self.gl_context)
        cube_vertex  = vertexes.get_cube_model()
        vertexes.model_data = cube_vertex
        vertexes.allocate_vertex_buffer(cube_vertex)
        self.resources.vertexes["cube"] = vertexes

        vertexes_2 = mesh.Vertexes(self.gl_context)
        circle_vertex  = vertexes_2.get_circle_model()
        vertexes_2.model_data = circle_vertex
        vertexes_2.allocate_vertex_buffer(circle_vertex)
        self.resources.vertexes["circle"] = vertexes_2

        vertexes_3 = mesh.Vertexes(self.gl_context)
        cone_vertex  = vertexes_3.get_cone_model()
        vertexes_3.model_data = cone_vertex
        vertexes_3.allocate_vertex_buffer(cone_vertex)
        self.resources.vertexes["cone"] = vertexes_3

        mesh_object = mesh.Mesh(self.gl_context, vertexes, self.resources.textures["debuging"], shader)
        mesh_object.create_object()
        self.resources.meshes["basic_cube"] = mesh_object

        mesh_object = mesh.Mesh(self.gl_context, vertexes, self.resources.textures["wood"], shader)
        mesh_object.create_object()
        self.resources.meshes["wooden_cube"] = mesh_object
        
        mesh_object = mesh.Mesh(self.gl_context, vertexes, self.resources.textures["iron"], shader)
        mesh_object.create_object()
        self.resources.meshes["iron_cube"] = mesh_object
        
        mesh_object = mesh.Mesh(self.gl_context, vertexes, self.resources.textures["gold"], shader)
        mesh_object.create_object()
        self.resources.meshes["gold_cube"] = mesh_object

        mesh_object = mesh.Mesh(self.gl_context, vertexes, self.resources.textures["red"], shader)
        mesh_object.create_object()
        self.resources.meshes["red_cube"] = mesh_object

        mesh_object = mesh.Mesh(self.gl_context, vertexes_2, self.resources.textures["gold"], shader)
        mesh_object.create_object()
        self.resources.meshes["circle"] = mesh_object

        mesh_object = mesh.Mesh(self.gl_context, vertexes_3, self.resources.textures["gold"], shader)
        mesh_object.create_object()
        self.resources.meshes["cone"] = mesh_object

    def create_objects_instances(self):
        pass

    def update(self, time:int):
        pass

    def render(self, time:int):
        self.update(time)
        for object in self.scene_objects:
            object.render(self.light, self.camera)


class Scene1(Scene):
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        super().__init__(resources, camera, gl_context)
        self.left_wall_position = np.array((-2,0,0), dtype=np.float32)
        self.left_wall = None
        self.right_wall_position = np.array((2,0,0), dtype=np.float32)
        self.right_wall = None
        self.circle_position = np.array((0,3,0), dtype=np.float32)
        self.circle = None
        self.circle_moving_direction = np.array([1,0,0], dtype=np.float32)
        self.circle_moving_direction = self.circle_moving_direction/np.linalg.norm(self.circle_moving_direction)
        self.circle_speed = 0.01
        # self.camera.locked = True

    def create_objects_instances(self):
        mesh_2 = self.resources.meshes["wooden_cube"]
        self.left_wall = SceneObject(mesh_2, self.light, position=self.left_wall_position, scale=(0.1,4,0.5))

        mesh_3 = self.resources.meshes["iron_cube"]
        self.right_wall = SceneObject(mesh_3, self.light, position=self.right_wall_position, scale=(0.1,4,0.5))

        mesh_5 = self.resources.meshes["circle"]
        self.circle = SceneObject(mesh_5, self.light, position=self.circle_position)

        self.scene_objects.append(self.left_wall)
        self.scene_objects.append(self.right_wall)
        self.scene_objects.append(self.circle)

        for x in range(-15,15, 2):
            for z in range(-15,15, 2):
                object = SceneObject(mesh_2, self.light, position=(x, -1, z))
                self.scene_objects.append(object)

    def update(self, time: int):
        if (self.right_wall.free_transformation is not None and
            self.right_wall.free_transformation[1,0] < 0.01 and 
            self.circle_moving_direction[0] < 0):
            return
        self.circle.rotate((0,0,0.005))
        self.circle.move(self.circle_moving_direction * self.circle_speed)
        self.circle_position = self.circle.position
        if (self.circle_position[0] +1.1 > self.right_wall_position[0]):
            share_transform = np.identity(4, dtype=np.float32)
            share_transform[1,0] = ((self.circle_position[0]-self.right_wall_position[0]+1.1)
            /self.circle_position[1])
            self.right_wall.free_transformation = share_transform
            if (self.right_wall.free_transformation[1,0]
                    /self.right_wall.free_transformation[1,1] > 0.5):
                self.circle_moving_direction *= -1


class Scene2(Scene):
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        super().__init__(resources, camera, gl_context)
        
        self.cone = None
        self.cube = None
        self.step = 1
        self.cube_speed = 0.01
        self.cone_tilt = 0.0

    def create_objects_instances(self):
        mesh_1 = self.resources.meshes["red_cube"]
        self.cube = SceneObject(mesh_1, self.light, position=(5,4,0), scale=(0.4, 0.2, 0.2))
        mesh_5 = self.resources.meshes["cone"]
        self.cone = SceneObject(mesh_5, self.light, position=(0,0,0), scale=(1, 3, 1))
        self.scene_objects.append(self.cube)
        self.scene_objects.append(self.cone)
        mesh_2 = self.resources.meshes["wooden_cube"]
        for x in range(-15,15, 2):
            for z in range(-15,15, 2):
                object = SceneObject(mesh_2, self.light, position=(x, -1, z))
                self.scene_objects.append(object)

    def update(self, time: int):
        target = self.cone.position.copy()
        target[1] +=(self.cone.scaling[1]/2)
        target[2] += 0.5
        distance = (target - self.cube.position)**2
        distance = np.power(np.sum(distance), 1/2)
        if distance < 0.1:
            self.step += 1
        if self.step == 1:
            movement_direction = self.cone.position - self.cube.position
            movement_direction[1] += (self.cone.scaling[1]/2)
            movement_direction[2] += 0.5
            movement_direction /= np.linalg.norm(movement_direction)
            cube_shift = movement_direction * self.cube_speed
            self.cube.move(cube_shift)
        else:
            self.cone_tilt += 0.03
            tilt_angle = math.radians(self.cone_tilt)
            quaternion = np.array([tilt_angle, 0.0, 0.0, 1.0],dtype=np.float32)
            self.cone.free_transformation = self.cone.rotate_around_vector(quaternion)

            rot = np.array([0.00, -0.01, 0.00], dtype=np.float32)
            self.cone.rotate(rot)
            self.cone.position[1] = math.sin(math.radians(self.cone_tilt))

            self.cube.position = np.zeros(3, dtype=np.float32)
            cube_transform = np.identity(4, dtype=np.float32)
            cube_transform[3,0:3] = np.array([0, 2, 0.5], dtype=np.float32)
            cube_around_cone_rotation = self.cube.rotate_around_vector(np.array([self.cone.rotation[1], 0, 1, 0], dtype=np.float32))
            cube_transform = cube_transform @ cube_around_cone_rotation @ self.cone.free_transformation @ cube_around_cone_rotation
            self.cube.position[1] = math.sin(math.radians(self.cone_tilt))

            self.cube.free_transformation = cube_transform


class Scene3(Scene):
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        super().__init__(resources, camera, gl_context)
        self.fan_scaling = 0.001
        self.fan_rotation = 0.005
        
        self.project:bool = False
        self.projection_point = np.array([1,1,1], dtype=np.float32)
        self.projection_forward = np.array([-1,-1,-1], dtype=np.float32)
        self.projection_right = np.cross(self.projection_forward, np.array([0,1,], dtype=np.float32))
        self.projection_up = np.cross(self.projection_forward, self.projection_right)
        self.projection_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix[:3, :3] = np.vstack([self.projection_right, self.projection_up, -1*self.projection_forward]).T
        self.projection_matrix[3,:3] = (-1*self.projection_matrix[:3, :3].T)@self.projection_point

    def initiate_resources(self):
        super().initiate_resources()
        vertexes = mesh.Vertexes(self.gl_context)
        vertexes.model_data = vertexes.get_fan_model()
        vertexes.allocate_vertex_buffer(vertexes.model_data)
        self.resources.vertexes["fan"] = vertexes
        texture = self.resources.textures["wood"]
        shaders = self.resources.shaders["default"]
        fan_mesh = mesh.Mesh(self.gl_context, vertexes, texture, shaders)
        fan_mesh.create_object()
        self.resources.meshes["wooden_fan"] = fan_mesh

    def create_objects_instances(self):
        self.fan = SceneObject( self.resources.meshes["wooden_fan"],self.light,
         position=(0,3,0), scale=(1, 1, 1))
        self.scene_objects.append(self.fan)
        mesh_2 = self.resources.meshes["wooden_cube"]
        for x in range(-15,15, 2):
            for z in range(-15,15, 2):
                object = SceneObject(mesh_2, self.light, position=(x, -1, z))
                self.scene_objects.append(object)

    def update(self, time: int):
        self.fan.rotate(np.array([0, self.fan_rotation, 0], dtype=np.float32))
        self.fan.scale(np.array([self.fan_scaling, self.fan_scaling, self.fan_scaling], dtype=np.float32))
        if self.fan.scaling[0] < 0.5:
            self.fan_scaling *= -1
            self.fan_rotation *= -1
            self.fan.rotate(np.array([np.pi, 0, 0], dtype=np.float32))
        elif self.fan.scaling[0] > 1:
            self.fan_scaling *= -1


class Scene4(Scene):
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        super().__init__(resources, camera, gl_context)
        self.project:bool = False
        self.projection_point = np.array([1,1,1], dtype=np.float32)
        self.projection_forward = np.array([-1,-1,-1], dtype=np.float32)
        self.projection_right = np.cross(self.projection_forward, np.array([0,1,], dtype=np.float32))
        self.projection_up = np.cross(self.projection_forward, self.projection_right)
        self.projection_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix[:3, :3] = np.vstack([self.projection_right, self.projection_up, -1*self.projection_forward]).T
        self.projection_matrix[3,:3] = (-1*self.projection_matrix[:3, :3].T)@self.projection_point

    def initiate_resources(self):
        super().initiate_resources()
        texture = self.resources.textures["red"]
        shaders = self.resources.shaders["default"]

        original_vertexes = mesh.Vertexes(self.gl_context)
        original_vertexes.model_data = original_vertexes.read_stl_model(
            "../lab3/T1/T1_1.stl")
        original_vertexes.allocate_vertex_buffer(original_vertexes.model_data)
        self.resources.vertexes["original_vertexes"] = original_vertexes
        
        original_mesh = mesh.Mesh(self.gl_context, original_vertexes, texture, shaders)
        original_mesh.create_object()
        self.resources.meshes["original_red"] = original_mesh

        converter = vertex_approximator.MeshConverter()
        square_grid = converter.triangle_to_square(original_vertexes.model_data)
        triangled_grid = converter.square_to_triangle(square_grid)
        self.initiate_from_vertexes(triangled_grid, "approximated_vertexes", "approximated_wood")

        lagrange = vertex_approximator.Lagrange()
        lagrange_interpolated_grid = lagrange.interpolate(square_grid, 3)
        lagrange_interpolated_triangles = converter.square_to_triangle(lagrange_interpolated_grid)
        self.initiate_from_vertexes(lagrange_interpolated_triangles,
                                     "lagrange_vertexes", "lagrange_wood")


    def initiate_from_vertexes(self, vertexes_array:np.ndarray, vertex_name:str, mesh_name:str) -> None:
        interpolated_vertexes = mesh.Vertexes(self.gl_context)
        interpolated_vertexes.model_data = vertexes_array
        interpolated_vertexes.allocate_vertex_buffer(interpolated_vertexes.model_data)
        self.resources.vertexes[vertex_name] = interpolated_vertexes
        shaders = self.resources.shaders["default"]
        interpolated_mesh = mesh.Mesh(self.gl_context, interpolated_vertexes, 
                                      self.resources.textures["wood"], shaders)
        interpolated_mesh.create_object()
        self.resources.meshes[mesh_name] = interpolated_mesh

    def create_objects_instances(self):
        self.original_object = SceneObject(self.resources.meshes["original_red"], self.light,
                        position=(0,1,0), scale=(4, 4, 4))
        self.scene_objects.append(self.original_object)

        self.approximated_object = SceneObject(self.resources.meshes["approximated_wood"], self.light,
                        position=(2,1,2), scale=(4, 4, 4))
        self.scene_objects.append(self.approximated_object)
        
        self.lagrange_object = SceneObject(self.resources.meshes["lagrange_wood"], self.light,
                        position=(3,1,3), scale=(4, 4, 4))
        self.scene_objects.append(self.lagrange_object)


        mesh_2 = self.resources.meshes["wooden_cube"]
        for x in range(-15,15, 2):
            for z in range(-15,15, 2):
                object = SceneObject(mesh_2, self.light, position=(x, -1, z))
                self.scene_objects.append(object)

    def update(self, time: int):
        self.import_rotation = 0.005
        self.original_object.rotate(np.array([0, self.import_rotation, 0], dtype=np.float32))
        self.approximated_object.rotate(np.array([0, self.import_rotation, 0], dtype=np.float32))
        self.lagrange_object.rotate(np.array([0, self.import_rotation, 0], dtype=np.float32))
        pass
        # self.imported.scale(np.array([self.fan_scaling, self.fan_scaling, self.fan_scaling], dtype=np.float32))
        # if self.imported.scaling[0] < 0.5:
        #     self.fan_scaling *= -1
        #     self.fan_rotation *= -1
        #     self.imported.rotate(np.array([np.pi, 0, 0], dtype=np.float32))
        # elif self.imported.scaling[0] > 1:
        #     self.fan_scaling *= -1