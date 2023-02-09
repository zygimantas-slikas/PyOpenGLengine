import glm
from ..mesh import *
from . import light, camera
import numpy as np

class SceneObject:
    def __init__(self, mesh:mesh.Mesh, light:light.Light, scale:tuple[float,float,float]=(1,1,1),
        rotation:tuple[float,float,float]=(0,0,0), position:tuple[float,float,float]=(0,0,0))->None:
        self.mesh:mesh.Mesh = mesh

        self.scaleing_transform:np.ndarray = np.identity(4, dtype=np.float32)
        self.rotation_transform:np.ndarray = np.identity(4, dtype=np.float32)
        self.position_transform:np.ndarray = np.identity(4, dtype=np.float32)
        
        self.scaling:list[float,float,float] = list(scale)
        self.rotation:list[float,float,float] = list(rotation)
        self.position:list[float,float,float] = list(position)

        self.set_scale(scale)
        self.set_rotation(rotation)
        self.set_position(position)

        self.update_lighting(light)

    def set_scale(self, scale:tuple[float,float,float])->None:
        self.scaling = [1,1,1]
        self.scaleing_transform = np.identity(4, dtype=np.float32)
        self.scale(scale)
    
    def set_rotation(self, rotation:tuple[float,float,float])->None:
        self.rotation = [0,0,0]
        self.rotation_transform = np.identity(4, dtype=np.float32)
        self.rotate(rotation)
    
    def set_position(self, position:tuple[float,float,float])->None:
        self.position = np.identity(4, dtype=np.float32)
        self.position = list(position)
        self.move(position)

    def scale(self, scale:tuple[float,float,float])->tuple[float,float,float]:
        scaling_matrix = np.identity(4, dtype=np.float32)
        for dimension in range(3):
            scaling_matrix[dimension, dimension] = scale[dimension]
            self.scaling[dimension] *= scale[dimension]
        self.scaleing_transform = self.scaleing_transform @ scaling_matrix
        return tuple(self.scaling)

    def rotate(self, rotation:tuple[float,float,float])->tuple[float,float,float]:
        x_rotation_matrix = np.identity(4, dtype=np.float32)
        x_rotation_matrix[1,1] = np.cos(rotation[0])
        x_rotation_matrix[1,2] = np.sin(rotation[0])
        x_rotation_matrix[2,1] = np.sin(rotation[0])*-1
        x_rotation_matrix[2,2] = np.cos(rotation[0])

        y_rotation_matrix = np.identity(4, dtype=np.float32)
        y_rotation_matrix[0,0] = np.cos(rotation[1])
        y_rotation_matrix[0,2] = np.sin(rotation[1])
        y_rotation_matrix[2,0] = np.sin(rotation[1])*-1
        y_rotation_matrix[2,2] = np.cos(rotation[1])

        z_rotation_matrix = np.identity(4, dtype=np.float32)
        z_rotation_matrix[0,0] = np.cos(rotation[2])
        z_rotation_matrix[0,1] = np.sin(rotation[2])
        z_rotation_matrix[1,0] = np.sin(rotation[2])*-1
        z_rotation_matrix[1,1] = np.cos(rotation[2])

        rotation_matrix = x_rotation_matrix @ y_rotation_matrix @ z_rotation_matrix
        self.rotation_transform = self.rotation_transform @ rotation_matrix
        self.rotation = [self.rotation[dimension] + rotation[dimension] for dimension in range(3)]
        return self.rotation

    def move(self, offset:tuple[float,float,float])->tuple[float,float,float]:
        offset_matrix:np.ndarray = np.identity(4, dtype=np.float32)
        offset_matrix[3,0] = offset[0]
        offset_matrix[3,1] = offset[1]
        offset_matrix[3,2] = offset[2]
        self.position_transform = self.position_transform @ offset_matrix
        self.position = [self.position[index] + offset[index] for index in range(3)]
        return self.position

    def update_lighting(self, light:light.Light)->None:
        self.mesh.shaders.program["light.position"].write(light.position)
        self.mesh.shaders.program["light.ambient_intensity"].write(light.ambient_intensity)
        self.mesh.shaders.program["light.diffusion_intensity"].write(light.diffuse_intensity)
        self.mesh.shaders.program["light.specular_intensity"].write(light.specular_intensity)

    def update(self, view_matrix, camera_projection_matrix, camera_position)->None:
        model_matrix = self.scaleing_transform @ self.rotation_transform @ self.position_transform
        self.mesh.shaders.program['model_matrix'].write(model_matrix)
        self.mesh.shaders.program['view_matrix'].write(view_matrix)
        self.mesh.shaders.program["camera_position"].write(camera_position)
        self.mesh.shaders.program['camera_projection_matrix'].write(camera_projection_matrix)

    def render(self, light:light.Light, camera:camera.Camera)->None:
        self.update(camera.view_matrix, camera.projection_matrix,camera.position)
        self.mesh.render()
