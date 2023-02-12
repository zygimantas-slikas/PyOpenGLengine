import numpy as np
import moderngl as mgl
import glm
import pygame as pg
from .light import Light
from .camera import Camera
from ..allocated_resources import Resources
from .. import mesh
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
        self.left_wall = SceneObject(mesh_2, self.light, position=self.left_wall_position)
        self.left_wall.scale((0.1,4,0.5))

        mesh_3 = self.resources.meshes["iron_cube"]
        self.right_wall = SceneObject(mesh_3, self.light, position=self.right_wall_position)
        self.right_wall.scale((0.1,4,0.5))

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
        self.circle_position = self.circle.position_transform[3,:3]
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
        self.cube_moving_direction = np.array([1,-1,-0.1], dtype=np.float32)
        self.cube_moving_direction = self.cube_moving_direction/np.linalg.norm(self.cube_moving_direction)
        self.cube_speed = 0.01

    def create_objects_instances(self):
        mesh_1 = self.resources.meshes["red_cube"]
        self.cube = SceneObject(mesh_1, self.light, position=(0,0,0), scale=(0.4, 0.2, 0.2))

        mesh_5 = self.resources.meshes["cone"]
        self.cone = SceneObject(mesh_5, self.light, position=(0,0,0))

        self.scene_objects.append(self.cube)
        self.scene_objects.append(self.cone)

        mesh_2 = self.resources.meshes["wooden_cube"]
        for x in range(-30,30, 2):
            for z in range(-30,30, 2):
                object = SceneObject(mesh_2, self.light, position=(x, -1, z))
                self.scene_objects.append(object)

    def update(self, time: int):
        distance = (self.cone.position - self.cube.position)**2
        distance = np.power(np.sum(distance), 1/2)
        if (distance > 2):
            scaler = (2+math.sin(math.radians(time*100)))/4
            self.cube.scale = (2*scaler,scaler,scaler)
            self.cube.move(self.cube_moving_direction*self.cube_speed)
        # else :
        #     self.cube.set_position((0,0,0))
        #     self.cube.rotate((0.1,0.1,0))
        #     # self.cube.free_transformation = np.array([[1,1,0,0],
        #     #                                         [0,1,0,0],
        #     #                                         [0,0,1,0],
        #     #                                         [0,0,0,1]])
        #     position = self.cone_position + np.array([0,2,0])


