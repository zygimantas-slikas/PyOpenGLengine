import numpy as np
import moderngl as mgl
import glm
import pygame as pg
from .light import Light
from .camera import Camera
from ..allocated_resources import Resources
from .. import mesh
from .scene_object import SceneObject


class Scene:
    def __init__(self, resources:Resources, camera:Camera, gl_context:mgl.Context):
        self.resources = resources
        self.camera = camera
        self.light = Light()
        self.gl_context = gl_context
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
        
        vertexes = mesh.Vertexes(self.gl_context)
        cube_vertex  = vertexes.get_cube_model()
        vertexes.model_data = cube_vertex
        vertexes.allocate_vertex_buffer(cube_vertex)
        self.resources.vertexes["cube"] = vertexes

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

    def create_objects_instances(self):
        mesh_1 = self.resources.meshes["basic_cube"]
        object_1 = SceneObject(mesh_1, self.light, position=(0,2,0))
        
        mesh_2 = self.resources.meshes["wooden_cube"]
        object_2 = SceneObject(mesh_2, self.light, position=(1,3,1))
        
        mesh_3 = self.resources.meshes["iron_cube"]
        object_3 = SceneObject(mesh_3, self.light, position=(-3,2,0))
        
        mesh_4 = self.resources.meshes["gold_cube"]
        object_4 = SceneObject(mesh_4, self.light, position=(0,2,-3))
        self.scene_objects.append(object_1)
        self.scene_objects.append(object_2)
        self.scene_objects.append(object_3)
        self.scene_objects.append(object_4)

        for x in range(-10,10, 2):
            for z in range(-10,10, 2):
                object = SceneObject(mesh_2, self.light, position=(x, 0, z))
                self.scene_objects.append(object)

    def update(self, time:int):
        for object in self.scene_objects[:4]:
            object.rotate((0,0.01,0))

    def render(self, time:int):
        self.update(time)
        for object in self.scene_objects:
            object.render( self.light, self.camera)
