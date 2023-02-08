import glm
from ..mesh import *
from . import light, camera

class SceneObject:
    def __init__(self, mesh:mesh.Mesh, light:light.Light, position=(0,0,0)):
        self.mesh:mesh.Mesh = mesh
        self.model_transformation_matrix = glm.mat4()
        self.update_lighting(light)

    def update_lighting(self, light:light.Light):
        self.mesh.shaders.program["light.position"].write(light.position)
        self.mesh.shaders.program["light.ambient_intensity"].write(light.ambient_intensity)
        self.mesh.shaders.program["light.diffusion_intensity"].write(light.diffuse_intensity)
        self.mesh.shaders.program["light.specular_intensity"].write(light.specular_intensity)

    def update(self, time:int, view_matrix, camera_projection_matrix, camera_position, transform:glm.mat4 = None):
        model_matrix = glm.rotate(self.model_transformation_matrix, time*0.5, glm.vec3(0,1,0))
        self.mesh.shaders.program['model_matrix'].write(model_matrix)
        self.mesh.shaders.program['view_matrix'].write(view_matrix)
        self.mesh.shaders.program["camera_position"].write(camera_position)
        self.mesh.shaders.program['camera_projection_matrix'].write(camera_projection_matrix)

    def render(self, time:int, light:light.Light, camera:camera.Camera):
        self.update(time, camera.view_matrix, camera.projection_matrix,camera.position)
        self.mesh.mesh_in_shader.render()
