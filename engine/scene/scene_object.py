import glm
from ..mesh import *
from . import light, camera
import numpy as np

class SceneObject:
    def __init__(self, mesh:mesh.Mesh, light:light.Light, scale:tuple[float,float,float]=(1,1,1),
        rotation:tuple[float,float,float]=(0,0,0), position:tuple[float,float,float]=(0,0,0))->None:
        self.mesh:mesh.Mesh = mesh
        self.scaling:np.ndarray = np.array(scale, dtype=np.float32)
        self.rotation:np.ndarray = np.array(rotation, dtype=np.float32)
        self.position:np.ndarray = np.array(position, dtype=np.float32)
        self.free_transformation:np.ndarray = np.identity(4, dtype=np.float32)
        self.update_lighting(light)

    def scale(self, scale:np.ndarray)->np.ndarray:
        self.scaling += scale
        return self.scaling

    def rotate(self, rotation:np.ndarray)->np.ndarray:
        self.rotation += rotation
        return self.rotation

    def move(self, offset:np.ndarray)->np.ndarray:
        self.position += offset
        return self.position

    def update_lighting(self, light:light.Light)->None:
        self.mesh.shaders.program["light.position"].write(light.position)
        self.mesh.shaders.program["light.ambient_intensity"].write(light.ambient_intensity)
        self.mesh.shaders.program["light.diffusion_intensity"].write(light.diffuse_intensity)
        self.mesh.shaders.program["light.specular_intensity"].write(light.specular_intensity)

    def update(self, view_matrix, camera_projection_matrix, camera_position)->None:
        self.mesh.shaders.program['scaling_factors'].write(self.scaling)
        self.mesh.shaders.program['shear_transform'].write(self.free_transformation)
        self.mesh.shaders.program['rotation_angles'].write(self.rotation)
        self.mesh.shaders.program['translation_vector'].write(self.position)

        self.mesh.shaders.program['view_matrix'].write(view_matrix)
        self.mesh.shaders.program["camera_position"].write(camera_position)
        self.mesh.shaders.program['camera_projection_matrix'].write(camera_projection_matrix)

    def render(self, light:light.Light, camera:camera.Camera)->None:
        self.update(camera.view_matrix, camera.projection_matrix,camera.position)
        self.mesh.render()
