import numpy as np
import moderngl as mgl
import glm
import math

class Vertexes:
    def __init__(self, gl_context:mgl.Context):
        self.gl_context:mgl.Context = gl_context
        self.model_data:np.ndarray = None
        self.buffer:mgl.Buffer = None

    def allocate_vertex_buffer(self, vertex_data:np.ndarray)->mgl.Buffer:
        self.buffer = self.gl_context.buffer(vertex_data)
        return self.buffer

    def delete(self):
        """Deallocates resources from OpenGL."""
        self.buffer.release()

    def combine_points(self, points:list[tuple], combinations:list[tuple])->np.ndarray:
        """Combines separate points to triangles by a given order."""
        points_list:list[tuple[float,float,float]] = []
        for combination in combinations:
            for index in combination:
                points_list.append(points[index])
        array = np.array(points_list, dtype=np.float32)
        return array

    def get_cube_model(self):
        vertices = [(-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1),
                    (-1,1,-1), (-1,-1,-1), (1,-1,-1), (1,1,-1)]
        indices = [(0,2,3), (0,1,2),
                    (1,7,2), (1,6,7),
                    (6,5,4), (4,7,6),
                    (3,4,5), (3,5,0),
                    (3,7,4), (3,2,7),
                    (0,6,1), (0,5,6)]
        vertex_data = self.combine_points(vertices, indices)

        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        texture_coordinates_indices = [(0,2,3), (0,1,2),
                                       (0,2,3), (0,1,2),
                                       (0,1,2), (2,3,0),
                                       (2,3,0), (2,0,1),
                                       (0,2,3), (0,1,2),
                                       (3,1,2), (3,0,1)]
        texture_coordinates_data = self.combine_points(texture_coordinates, texture_coordinates_indices)

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

    def get_circle_model(self):
        n = 30
        r = 1
        vertices:list[tuple[float,float,float]]= []
        for angle in [360/n*i for i in range(0, n)]:
            x = math.cos(math.radians(angle))
            y = math.sin(math.radians(angle))
            z = 0
            vertices.append((x,y,z))
        indices = [(0, i, i+1) for i in range(0, n-1, 1)]
        vertex_data = self.combine_points(vertices, indices)
        
        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        texture_coordinates_indices = [(0,1,2) * int((n-1))]
        texture_coordinates_data = self.combine_points(texture_coordinates, texture_coordinates_indices)

        normals = [(0,0,1)*int((n-1)*3)]
        normals = np.array(normals, dtype = np.float32).reshape(int((n-1)*3), 3)

        vertex_data = np.hstack([normals, vertex_data])
        model_data = np.hstack([texture_coordinates_data, vertex_data])
        return model_data

    def get_cone_model(self):
        n = 30
        r = 1
        h = 1
        vertices:list[tuple[float,float,float]]= []
        vertices.append((0, h,0))
        for angle in [360/n*i for i in range(0, n)]:
            x = math.cos(math.radians(angle))
            y = 0
            z = math.sin(math.radians(angle))
            vertices.append((x,y,z))

        indices_1 = [(1, i, i+1) for i in range(1, n, 1)]
        indices_2 = [(0, i+1, i) for i in range(1, n, 1)]
        indices_2.append((0, 1, n-1))
        indices = indices_2 + indices_1
        vertex_data = self.combine_points(vertices, indices)
        
        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        texture_coordinates_indices = [(0,1,2) * int(n*2-1)]
        texture_coordinates_data = self.combine_points(texture_coordinates, texture_coordinates_indices)

        normals = [(0,0,0)*int((n*2-1)*3)]
        normals = np.array(normals, dtype = np.float32).reshape(int((n*2-1)*3), 3)

        vertex_data = np.hstack([normals, vertex_data])
        model_data = np.hstack([texture_coordinates_data, vertex_data])
        return model_data

    def get_fan_model(self):
        u_count = 30
        v_count = 70

        u = [x/10 for x in range(0, u_count+1, 1)]
        v = [x/10 for x in range(0, v_count+1, 1)]
        points_array = np.zeros(shape=(u_count, v_count, 3), dtype=np.float32)
        for u_index in range(u_count):
            for v_index in range(v_count):
                points_array[u_index, v_index, 0] = u[u_index] * math.cos(v[v_index])
                points_array[u_index, v_index, 1] = u[u_index]
                points_array[u_index, v_index, 2] = u[u_index] * math.sin(v[v_index])

        vertices = []
        for u_index in range(1, u_count, 1):
            for v_index in range(0, v_count-1, 1):
                vertices.append(points_array[u_index,v_index,:])
                vertices.append(points_array[u_index-1,v_index+1,:])
                vertices.append(points_array[u_index-1,v_index,:])

                vertices.append(points_array[u_index,v_index,:])
                vertices.append(points_array[u_index,v_index+1,:])
                vertices.append(points_array[u_index-1,v_index+1,:])

                vertices.append(points_array[u_index,v_index,:])
                vertices.append(points_array[u_index-1,v_index,:])
                vertices.append(points_array[u_index-1,v_index+1,:])

                vertices.append(points_array[u_index,v_index,:])
                vertices.append(points_array[u_index-1,v_index+1,:])
                vertices.append(points_array[u_index,v_index+1,:])
        vertex_data = np.array(vertices, dtype=np.float32)

        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        texture_coordinates_indices = []
        for x in range((u_count-1)*(v_count-1)*2*2):
            if x%4==0:
                texture_coordinates_indices.append((3,1,0))
            elif x%4==1:
                texture_coordinates_indices.append((3,2,1))
            elif x%4==2:
                texture_coordinates_indices.append((3,0,1))
            else:
                texture_coordinates_indices.append((3,1,2))
        texture_coordinates_data = self.combine_points(texture_coordinates, texture_coordinates_indices)

        normals = [(0,0,0) * (u_count-1)*(v_count-1)*2*2*3]
        normals = np.array(normals, dtype = np.float32).reshape((u_count-1)*(v_count-1)*2*2*3, 3)

        vertex_data = np.hstack([normals, vertex_data])
        model_data = np.hstack([texture_coordinates_data, vertex_data])
        return model_data