import numpy as np
import moderngl as mgl
import glm
import pygame as pg


class Camera:
    def __init__(self, window_size:tuple[int,int], FOV=np.pi*7/18, NEAR=0.1, FAR=100,
        position=(0,1,5), yaw=-90.0, pitch=0.0):
        self.FOV = FOV
        # self.focal_distance = 8
        # self.FOV = 2*np.arctan(2/self.focal_distance)
        self.NEAR = NEAR
        self.FAR = FAR
        self.window_size = window_size
        self.aspect_ratio = window_size[0] / window_size[1]
        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.movement_speed = 0.01
        self.yaw = yaw
        self.pitch = pitch
        self.mouse_sensitivity = 0.1
        self.view_matrix = self.get_view_matrix(self.position)
        self.projection_matrix = self.get_projection_matrix()
        self.locked:bool = False

    def update(self, delta_time)->glm.mat4x4:
        if not self.locked:
            self.move(delta_time)
            self.get_mouse_rotation()
            self.update_camera_vectors()
        self.view_matrix = self.get_view_matrix(self.position)
        return self.view_matrix

    def move(self, delta_time):
        distance = self.movement_speed * delta_time
        forward = self.forward
        up = self.up
        right = self.right
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.position += forward * distance
        if keys[pg.K_s]:
            self.position -= forward * distance
        if keys[pg.K_d]:
            self.position += right * distance
        if keys[pg.K_a]:
            self.position -= right * distance
        if keys[pg.K_LSHIFT]:
            self.position -= up * distance
        if keys[pg.K_SPACE]:
            self.position += up * distance

    def get_mouse_rotation(self):
        relative_x, relative_y = pg.mouse.get_rel()
        self.yaw += relative_x * self.mouse_sensitivity
        self.pitch -= relative_y * self.mouse_sensitivity
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        yaw = glm.radians(self.yaw)
        pitch = glm.radians(self.pitch)

        self.forward.x = glm.cos(pitch) * glm.cos(yaw)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.cos(pitch) * glm.sin(yaw)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0,1,0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def get_projection_matrix(self) -> np.ndarray:
        perspective_projection_to_camera_matrix = np.array(
            [1/(1/self.aspect_ratio)*np.tan(self.FOV/2),0,0,0,
             0,1/np.tan(self.FOV/2),0,0,
             0,0,-1*(self.FAR+self.NEAR)/(self.FAR-self.NEAR), -1,
             0,0,-2*self.FAR*self.NEAR/(self.FAR-self.NEAR),0
             ], dtype=np.float32).reshape((4,4))
        return perspective_projection_to_camera_matrix

    def get_view_matrix(self, position:np.ndarray)->np.ndarray:
        result = np.identity(4, dtype=np.float32)
        result[:3, :3] = np.vstack([self.right, self.up, -1*self.forward]).T
        result[3,:3] = (-1*result[:3, :3].T)@position
        return result