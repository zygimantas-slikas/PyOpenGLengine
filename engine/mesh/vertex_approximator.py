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

    def triangles_with_hight(self, hight:float, triangles:list[np.ndarray],
                              edge_normals:list[np.ndarray]) -> tuple[list, list]:
        result_triangles = []
        result_normals = []
        for triangle, normals in zip(triangles, edge_normals):
            higher = np.any(triangle[:, 1] > hight)
            lower = np.any(triangle[:, 1] < hight)
            if higher and lower:
                result_triangles.append(triangle)
                result_normals.append(normals)
        return (result_triangles, result_normals)

    def triangle_to_square(self, points:np.ndarray) -> np.ndarray:
        positions = points[:, -3:]
        triangles = self.get_triangles_list(positions)
        edge_normals = self.compute_edge_normals(triangles)

        u, v = 30, 10
        mesh = np.empty(shape=(u, v, 6), dtype=np.float32)
        for i, hight in enumerate([x/(u-1) for x in range(0, u)]):
            for j, angle in enumerate([x/int(math.ceil(628/v)) for x in range(0, 628, int(math.ceil(628/v)))]):
                ray = np.array([1, hight, angle], dtype=np.float32)
                ray_x = ray[-3] * np.cos(ray[-1])
                ray_z = ray[-3] * np.sin(ray[-1])
                ray[-3] = ray_x
                ray[-1] = ray_z
        
                mesh[i, j, :3] = ray / np.linalg.norm(ray)
                triangles_in_hight = self.triangles_with_hight(ray[1], triangles, edge_normals)
                if len (triangles_in_hight[0]) > 0:
                    triangles_in_hight = np.stack(triangles_in_hight[0])
                    triangles_in_hight = self.convert_to_spherical(triangles_in_hight.reshape((-1, 1, 3)))
                    triangles_in_hight = triangles_in_hight.reshape((-1, 3))
                    triangles_in_hight = triangles_in_hight[:, 0].mean()
                else :
                    triangles_in_hight = 0

                print(triangles_in_hight)
                # if len(triangles_in_hight[0]) > 0:
                    # intersection = self.find_intersection(ray, *triangles_in_hight)
                    # ray[0] = intersection[0]
                ray = np.array([triangles_in_hight, hight, angle], dtype=np.float32)
                ray_x = ray[-3] * np.cos(ray[-1])
                ray_z = ray[-3] * np.sin(ray[-1])
                ray[-3] = ray_x
                ray[-1] = ray_z
                
                mesh[i, j, -3:] = ray
        # mesh = self.convert_to_spherical(mesh)
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
        # x_positions = vertex_data[:, -3] * np.cos(vertex_data[:, -1])
        # z_positions = vertex_data[:, -3] * np.sin(vertex_data[:, -1])
        # vertex_data[:, -3] = x_positions
        # vertex_data[:, -1] = z_positions

        # x_positions = vertex_data[:, -6] * np.cos(vertex_data[:, -4])
        # z_positions = vertex_data[:, -6] * np.sin(vertex_data[:, -4])
        # vertex_data[:, -6] = x_positions
        # vertex_data[:, -4] = z_positions
        return vertex_data


class Lagrange:
    pass


class Bezier:
    pass


class BSpline:
    pass