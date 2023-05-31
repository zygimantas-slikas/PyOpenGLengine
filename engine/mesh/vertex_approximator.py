import numpy as np
import moderngl as mgl
import glm
import math
import struct


class MeshConverter:
    def __init__(self):
        pass

    def get_triangles_list(self, points:np.ndarray) -> list[np.ndarray]:
        triangles_count = int(points.shape[0]/3)
        triangles = np.array_split(points, triangles_count)
        return triangles

    def convert_to_spherical(self, points:np.ndarray) ->np.ndarray:
        """Grid of points"""
        # old_shape = points.shape
        # points = points.reshape((-1, 6))
        radiuses = np.sqrt(np.sum(points[:, :, [0,2]]**2, axis=-1))
        arccos:np.ndarray = np.arccos(points[:, :, 0]/radiuses)
        y_negative = points[:, :, 2] < 0
        angles = arccos.copy()
        angles[y_negative] = np.pi*2 - arccos[y_negative]
        points[:, :, 0] = radiuses
        points[:, :, 2] = angles
        # points = points.reshape(old_shape)
        points = np.nan_to_num(points)
        return points

    def convert_to_euclidean(self, points:np.ndarray)->np.ndarray:
        """Row of points"""
        x_positions = points[:, :, -3] * np.cos(points[:, :, -1])
        z_positions = points[:, :, -3] * np.sin(points[:, :, -1])
        points[:, :, -3] = x_positions
        points[:, :, -1] = z_positions

        # x_positions = points[:, :, -6] * np.cos(points[:, :, -4])
        # z_positions = points[:, :, -6] * np.sin(points[:, :, -4])
        # points[:, :, -6] = x_positions
        # points[:, :, -4] = z_positions
        return points

    def compute_edge_normals(self, triangles:list[np.ndarray]) -> list[np.ndarray]:
        result = []
        for triangle in triangles:
            edge_vectors = triangle[[1, 2, 0], :] - triangle
            normals = np.cross(edge_vectors[[1, 2, 0], :], edge_vectors,)
            normal = normals.mean(axis=0)
            normal = normal / np.linalg.norm(normal)
            edge_normals = np.cross(edge_vectors, normal)
            result.append(edge_normals)
        return result

    def did_it_hit(self, ray:np.ndarray, points:np.ndarray, edge_normals:np.ndarray) -> bool:
        vertexes_directions = ray - points
        inside = True
        inside &= np.dot(vertexes_directions[0, :], edge_normals[0, :]) > 0
        inside &= np.dot(vertexes_directions[0, :], edge_normals[2, :]) > 0
        inside &= np.dot(vertexes_directions[1, :], edge_normals[1, :]) > 0
        inside &= np.dot(vertexes_directions[1, :], edge_normals[0, :]) > 0
        inside &= np.dot(vertexes_directions[2, :], edge_normals[1, :]) > 0
        inside &= np.dot(vertexes_directions[2, :], edge_normals[2, :]) > 0
        return inside

    def find_intersection(self, ray:np.ndarray, triangles:list[np.ndarray],
                           edge_normals:list[np.ndarray]) -> np.ndarray:
        for triangle, normals in zip(triangles, edge_normals):
            if self.did_it_hit(ray, triangle, normals):
                break
        midle_point = triangle.mean(axis=0)
        return midle_point

    def triangles_with_hight(self, hight:float, triangles:list[np.ndarray]) -> list[np.ndarray]:
        result_triangles = []
        for triangle in triangles:
            higher = np.any(triangle[:, 1] > hight)
            lower = np.any(triangle[:, 1] < hight)
            if higher and lower:
                result_triangles.append(triangle)
        return result_triangles

    def triangle_to_square(self, points:np.ndarray) -> np.ndarray:
        positions = points[:, -3:]
        triangles = self.get_triangles_list(positions)
        u, v = 6, 6
        mesh = np.empty(shape=(u, v, 6), dtype=np.float32)
        for i, hight in enumerate([x/(u-1) for x in range(0, u)]):
            triangles_in_hight = self.triangles_with_hight(hight, triangles)
            if len (triangles_in_hight) > 0:
                triangles_in_hight = np.stack(triangles_in_hight[0])
                triangles_in_hight = self.convert_to_spherical(triangles_in_hight.reshape((-1, 1, 3)))
                triangles_in_hight = triangles_in_hight.reshape((-1, 3))
                mean_radius = triangles_in_hight[:, 0].mean()
            else :
                mean_radius = 0
            for j, angle in enumerate([(x/100) for x in range(0, 628, int(math.ceil(628/(v))))]):
                ray = np.array([mean_radius, hight, angle], dtype=np.float32)
                ray_x = ray[-3] * np.cos(ray[-1])
                ray_z = ray[-3] * np.sin(ray[-1])
                ray[-3] = ray_x
                ray[-1] = ray_z
                mesh[i, j, -3:] = ray
                if np.linalg.norm(ray) != 0:
                    ray /= np.linalg.norm(ray)
                mesh[i, j, :3] = ray
        return mesh

    def square_to_triangle(self, surface:np.ndarray) -> np.ndarray:
        u_count = surface.shape[0]
        v_count = surface.shape[1]
        texture_coordinates = [(0,0), (1,0), (1,1), (0,1)]
        vertices = []
        for u_index in range(1, u_count, 1):
            for v_index in range(0, v_count-1, 1):
                point_0 = np.concatenate([np.array([*texture_coordinates[0]]), surface[u_index-1, v_index, : ]])
                point_1 = np.concatenate([np.array([*texture_coordinates[1]]), surface[u_index-1, v_index+1, :]])
                point_2 = np.concatenate([np.array([*texture_coordinates[2]]), surface[u_index,v_index+ 1, :]])
                point_3 = np.concatenate([np.array([*texture_coordinates[3]]), surface[u_index, v_index, :]])

                vertices.append(point_3)
                vertices.append(point_1)
                vertices.append(point_0)

                vertices.append(point_3)
                vertices.append(point_2)
                vertices.append(point_1)

                vertices.append(point_3)
                vertices.append(point_0)
                vertices.append(point_1)

                vertices.append(point_3)
                vertices.append(point_1)
                vertices.append(point_2)
        vertex_data = np.array(vertices, dtype=np.float32)
        return vertex_data


class Lagrange:
    def __init__(self):
        self.mesh_converter = MeshConverter()

    def generate_base_functions(self, control_points:np.ndarray,
                                interpolation_points:np.ndarray) -> np.ndarray:
        new_shape = (control_points.shape[0], interpolation_points.shape[0])
        functions_values = np.empty(shape = new_shape, dtype=np.float32)
        for index, x_value in enumerate(control_points):
            reduced_control_points = np.delete(control_points, index)
            denominator:np.ndarray = x_value - reduced_control_points
            denominator = denominator.prod()
            nominators = interpolation_points.reshape((-1, 1)) - reduced_control_points
            nominators:np.ndarray = nominators.prod(axis=1)
            interpolated_values = nominators / denominator
            functions_values[index, :] = interpolated_values
        return functions_values

    def interpolate_coordinates(self, control_points:np.ndarray, multiplyer:int) -> np.ndarray:
        new_lenght = (control_points.shape[0] -1)* multiplyer + 1
        new_points = np.empty(shape=(new_lenght), dtype=np.float32)
        for index, position in enumerate(range(0, new_lenght-1, multiplyer)):
            interpolated_coordinates = np.linspace(control_points[index], 
                                    control_points[index+1], multiplyer+1)
            new_points[position:position+multiplyer] = interpolated_coordinates[:-1]
        new_points[-1] = control_points[-1]
        return new_points

    def interpolate(self, points_grid:np.ndarray, interpolation_multiplyer:int) -> np.ndarray:
        points_grid = points_grid[:, :, -3:]
        spherical_points = self.mesh_converter.convert_to_spherical(points_grid)
        u:np.ndarray = spherical_points[:, 1, 1]
        v:np.ndarray = spherical_points[1, :, 2]
        control_coefficients = spherical_points[:, :, 0]
        interpolated_u = self.interpolate_coordinates(u, interpolation_multiplyer)
        interpolated_v = self.interpolate_coordinates(v, interpolation_multiplyer)
        base_u_functions = self.generate_base_functions(u, interpolated_u)
        base_v_functions = self.generate_base_functions(v, interpolated_v)

        new_size = (base_u_functions.shape[1], base_v_functions.shape[1])
        interpolated_surface = np.empty(shape=new_size, dtype=np.float32)
        for u_index in range(u.shape[0]):
            for v_index in range(v.shape[0]):
                surface_component = (base_u_functions[u_index, :].reshape(-1, 1)
                                      * base_v_functions[v_index, :])
                surface_component *= control_coefficients[u_index, v_index]
                interpolated_surface += surface_component
        u, v = np.meshgrid(interpolated_v, interpolated_u)
        interpolated_surface = np.stack([interpolated_surface, v, u], axis=2)
        normals = np.zeros(shape=interpolated_surface.shape, dtype=np.float32)
        interpolated_surface = np.concatenate([normals, interpolated_surface], axis=2)
        # old_shape = interpolated_surface.shape
        # interpolated_surface = interpolated_surface.reshape((-1, 6))
        interpolated_surface = self.mesh_converter.convert_to_euclidean(interpolated_surface)
        # interpolated_surface = interpolated_surface.reshape(old_shape)
        return interpolated_surface

class Bezier:
    def __init__(self):
        self.mesh_converter = MeshConverter()

    def generate_base_functions(self, interpolation_points:np.ndarray) -> np.ndarray:
        b0 = 1/6*(1-interpolation_points)**3
        b1 = 1/6*(4 - 6*interpolation_points**2 + 3*interpolation_points**3)
        b2 = 1/6*(1+3*interpolation_points+3*interpolation_points**2-3*interpolation_points**3)
        b3 = 1/6*interpolation_points**3
        base_coefficients = np.vstack([b0, b1, b2, b3])
        return base_coefficients

    def bernstein_polynomial(self, n, x):
        # n - number of control points
        # x - points at which to evaluate the polynomial in the parameter interval [0,1]
        T = []
        for i in range(n):
            T.append(x.T ** (n-i+1))
        T = np.array(T)
        
        if n == 3:
            CC = np.array([[ 1, -2,  1],
                        [-2,  2,  0],
                        [ 1,  0,  0]])
        elif n == 4:
            CC = np.array([[-1,  3, -3, 1],
                        [ 3, -6,  3, 0],
                        [-3,  3,  0, 0],
                        [ 1,  0,  0, 0]])
        elif n == 5:
            CC = np.array([[1, -4,   6,-4,1],
                        [-4,12,-12,4 ,0],
                        [6,-12 ,6 ,0 ,0],
                        [-4 ,4 ,0 ,0 ,0],
                        [1 ,0 ,0 ,0 ,0]])
        elif n ==6:
            CC = np.array([[-1 ,5 ,-10 ,10 ,-5 ,1],
                        [5 ,-20 ,30 ,-20 ,5 ,0],
                        [-10 ,30 ,-30 ,10 ,0 ,0],
                        [10 ,-20 ,10 ,0 ,0 ,0],
                        [-5 ,5 ,0 ,0 ,0 ,0],
                        [1 ,0 ,0 ,0 ,0 ,0]])
            
        return np.dot(T, CC).T

    def interpolate(self, points_grid:np.ndarray, interpolation_multiplyer:int) -> np.ndarray:
        points_grid = np.nan_to_num(points_grid)
        points_grid = points_grid[:, :, -3:]
        spherical_points = self.mesh_converter.convert_to_spherical(points_grid)
        u:np.ndarray = spherical_points[:, 1, 1]
        v:np.ndarray = spherical_points[1, :, 2]
        v /= np.pi*2
        control_coefficients = spherical_points[:, :, 0]
        # interpolated_u = self.interpolate_coordinates(u, interpolation_multiplyer)
        # interpolated_v = self.interpolate_coordinates(v, interpolation_multiplyer)
        bernstein_poly_u = self.bernstein_polynomial(6, u)
        bernstein_poly_v = self.bernstein_polynomial(6, v)
        
        interpolated_surface = np.empty(shape=points_grid.shape, dtype=np.float32)
        for i in range(bernstein_poly_u.shape[0]):
            for j in range(bernstein_poly_v.shape[0]):
                point_value = bernstein_poly_u[:, i] @ bernstein_poly_v[j, :] 
                interpolated_surface[i, j, 0] = point_value * control_coefficients[i, j]
                # interpolated_surface[i, j, 1] = point_value * spherical_points[i, j, 1]
                # interpolated_surface[i, j, 2] = point_value * spherical_points[i, j, 2]
        interpolated_surface[:, :, 1] = spherical_points[:, :, 1]
        interpolated_surface[:, :, 2] = spherical_points[:, :, 2]
        
        normals = np.zeros(shape=spherical_points.shape, dtype=np.float32)
        interpolated_surface = np.concatenate([normals, interpolated_surface], axis=2)
        interpolated_surface = self.mesh_converter.convert_to_euclidean(interpolated_surface)
        return interpolated_surface

class BSpline:
    def __init__(self):
        self.mesh_converter = MeshConverter()

    def generate_base_functions(self, interpolation_points:np.ndarray) -> np.ndarray:
        b0 = 1/6*(1-interpolation_points)**3
        b1 = 1/6*(4 - 6*interpolation_points**2 + 3*interpolation_points**3)
        b2 = 1/6*(1+3*interpolation_points+3*interpolation_points**2-3*interpolation_points**3)
        b3 = 1/6*interpolation_points**3
        base_coefficients = np.vstack([b0, b1, b2, b3])
        return base_coefficients

    def interpolate_coordinates(self, control_points:np.ndarray, multiplyer:int) -> np.ndarray:
        new_lenght = (control_points.shape[0] -1)* multiplyer + 1
        new_points = np.empty(shape=(new_lenght), dtype=np.float32)
        for index, position in enumerate(range(0, new_lenght-1, multiplyer)):
            interpolated_coordinates = np.linspace(control_points[index], 
                                    control_points[index+1], multiplyer+1)
            new_points[position:position+multiplyer] = interpolated_coordinates[:-1]
        new_points[-1] = control_points[-1]
        return new_points

    def interpolate(self, points_grid:np.ndarray, interpolation_multiplyer:int) -> np.ndarray:
        points_grid = np.nan_to_num(points_grid)
        points_grid = points_grid[:, :, -3:]
        spherical_points = self.mesh_converter.convert_to_spherical(points_grid)
        u:np.ndarray = spherical_points[:, 1, 1]
        v:np.ndarray = spherical_points[1, :, 2]
        control_coefficients = spherical_points[:, :, 0]
        interpolated_u = self.interpolate_coordinates(u, interpolation_multiplyer)
        interpolated_v = self.interpolate_coordinates(v, interpolation_multiplyer)

        old_size = (u.shape[0], v.shape[0])
        new_size = (interpolated_u.shape[0], interpolated_v.shape[0])
        interpolated_surface = np.empty(shape=new_size, dtype=np.float32)
        for i, new_i in enumerate(range(0, new_size[0]-interpolation_multiplyer, interpolation_multiplyer)):
            for j, new_j in enumerate(range(0, new_size[1]-interpolation_multiplyer, interpolation_multiplyer)):
                inter_points = [x/interpolation_multiplyer for x in range(0, interpolation_multiplyer+1)]
                inter_points = np.array(inter_points, dtype=np.float32)
                u_inter_points = interpolated_u[new_i:new_i+interpolation_multiplyer+1]
                u_base = self.generate_base_functions(inter_points)
                v_inter_points = interpolated_v[new_j:new_j+interpolation_multiplyer+1]
                v_base = self.generate_base_functions(inter_points)

                u_control_points_index = [i-1, i, (i+1)%old_size[0], (i+2)%old_size[0]]
                v_control_points_index = [j-1, j, (j+1)%old_size[0], (j+2)%old_size[0]]
                control_grid = spherical_points[u_control_points_index,:, :]
                control_grid = control_grid[:,  v_control_points_index, 0]
                inter_patch = np.zeros(shape=(interpolation_multiplyer+1,
                                interpolation_multiplyer+1), dtype=np.float32)
                for k in range(u_base.shape[0]):
                    for l in range(v_base.shape[0]):
                        inter_patch += (u_base[k, :].reshape((-1, 1)) *
                                    v_base[l, :] * control_grid[k, l])
                interpolated_surface[new_i:new_i+interpolation_multiplyer+1,
                            new_j:new_j+interpolation_multiplyer+1] = inter_patch
        u, v = np.meshgrid(interpolated_v, interpolated_u)
        interpolated_surface = np.stack([interpolated_surface, v, u], axis=2)
        normals = np.zeros(shape=interpolated_surface.shape, dtype=np.float32)
        interpolated_surface = np.concatenate([normals, interpolated_surface], axis=2)
        interpolated_surface = self.mesh_converter.convert_to_euclidean(interpolated_surface)
        return interpolated_surface