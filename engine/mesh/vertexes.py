import numpy as np
import moderngl as mgl
import glm

class Vertexes:
    def __init__(self, gl_context:mgl.Context):
        self.gl_context:mgl.Context = gl_context
        self.model_data:np.ndarray = None
        self.model_data = self.set_model_data()
        self.buffer = self.allocate_vertex_buffer(self.model_data)

    def allocate_vertex_buffer(self, vertex_data:np.ndarray)->mgl.Buffer:
        vertex_buffer = self.gl_context.buffer(vertex_data)
        return vertex_buffer

    def delete(self):
        """Deallocates resources from OpenGL."""
        self.buffer.release()

    def set_model_data(self):
        vertices = [(-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1),
                    (-1,1,-1), (-1,-1,-1), (1,-1,-1), (1,1,-1)]
        indices = [(0,2,3), (0,1,2),
                    (1,7,2), (1,6,7),
                    (6,5,4), (4,7,6),
                    (3,4,5), (3,5,0),
                    (3,7,4), (3,2,7),
                    (0,6,1), (0,5,6)]
        vertex_data:list[tuple[int,int,int]]=[]
        for triangle in indices:
            for index in triangle:
                vertex_data.append(vertices[index])
        vertex_data:np.ndarray = np.array(vertex_data, dtype=np.float32)

        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        texture_coordinates_indices = [(0,2,3), (0,1,2),
                                       (0,2,3), (0,1,2),
                                       (0,1,2), (2,3,0),
                                       (2,3,0), (2,0,1),
                                       (0,2,3), (0,1,2),
                                       (3,1,2), (3,0,1)]
        texture_coordinates_data = []
        texture_coordinates_data:list[tuple[int,int,int]]=[]
        for triangle in texture_coordinates_indices:
            for index in triangle:
                texture_coordinates_data.append(texture_coordinates[index])
        texture_coordinates_data:np.ndarray = np.array(texture_coordinates_data, dtype=np.float32)

        normals = [(0,0,1)*6,
                   (1,0,0)*6,
                   (0,0,-1)*6,
                   (-1,0,0)*6,
                   (0,1,0)*6,
                   (0,-1,0)*6]
        normals = np.array(normals, dtype = np.float32).reshape(36, 3)

        vertex_data = np.hstack([normals, vertex_data])
        model_data = np.hstack([texture_coordinates_data, vertex_data])
        return model_data