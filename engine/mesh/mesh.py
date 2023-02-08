import numpy as np
import moderngl as mgl
from .texture import Texture
from .vertexes import Vertexes
from .shaders import Shaders

class Mesh:
    def __init__(self, gl_context:mgl.Context, vertex:Vertexes=None,
     texture:Texture=None, shaders:Shaders=None)->None:
        """Creates object for holding mesh objects data, and reference to allocated space on GPU 
        buffer."""
        self.gl_context:mgl.Context = gl_context
        self.vertex:Vertexes = vertex
        self.texture:Texture = texture
        self.shaders:Shaders = shaders
        self.mesh_in_shader:mgl.VertexArray = None
        self.texture.texture_data.use(location = 0)

    def create_objects_shader(self)->mgl.VertexArray:
        if (self.vertex.buffer is not None
            and self.texture.texture_data is not None
            and self.shaders.program is not None):
            self.mesh_in_shader = self.gl_context.vertex_array(self.shaders.program, 
            [(self.vertex.buffer, '2f 3f 3f', 'in_texture_coordinates_0', 'in_normal', 'in_position')])
            return self.mesh_in_shader
        else:
            print("Failed to create a mesh. Some data missing.")
            return None
    
    def delete(self):
        """Deallocates resources from OpenGL."""
        self.mesh_in_shader.release()
        self.vertex.delete()
        self.shaders.delete()

    def read_shaders(self, file_name:str)->mgl.Program:
        ...

    def load_object_from_file(self, file_anme:str)->list:
        ...

    def generate_polygons(self, vertex_locations:np.ndarray, texture_locations:np.ndarray):
        ...