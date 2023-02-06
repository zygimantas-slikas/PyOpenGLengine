import numpy as np
import moderngl as mgl
import glm
import pygame as pg

class Camera:
    def __init__(self, window_size:tuple[int,int], FOV=70, NEAR=0.1, FAR=100,
        position=(0,1,5), yaw=-90, pitch=0):
        self.FOV = FOV
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
        self.mouse_sensitivity = 0.005
        self.view_matrix = self.get_view_matrix(self.position, self.up)
        self.projection_matrix = self.get_projection_matrix()

    def update(self, delta_time):
        self.move(delta_time)
        self.get_mouse_rotation()
        self.update_camera_vectors()
        self.view_matrix = self.get_view_matrix(self.position, self.up)

    def move(self, delta_time):
        distance = self.movement_speed * delta_time
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.position += self.forward * distance
        if keys[pg.K_s]:
            self.position -= self.forward * distance
        if keys[pg.K_d]:
            self.position += self.right * distance
        if keys[pg.K_a]:
            self.position -= self.right * distance
        if keys[pg.K_LSHIFT]:
            self.position -= self.up * distance
        if keys[pg.K_SPACE]:
            self.position += self.up * distance

    def get_mouse_rotation(self):
        relative_x, relative_y = pg.mouse.get_rel()
        self.yaw += relative_x * self.mouse_sensitivity
        self.pitch -= relative_y * self.mouse_sensitivity
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        yaw = glm.radians(self.yaw)
        pitch = glm.radians(self.pitch)

        self.forward.x = glm.cos(self.pitch) * glm.cos(self.yaw)
        self.forward.y = glm.sin(self.pitch)
        self.forward.z = glm.cos(self.pitch) * glm.sin(self.yaw)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0,1,0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def get_projection_matrix(self):
        perspective_projection_to_camera_matrix = glm.perspective(
            glm.radians(self.FOV), self.aspect_ratio, self.NEAR, self.FAR)
        return perspective_projection_to_camera_matrix

    def get_view_matrix(self, position, up_vector)->glm.mat4:
        return glm.lookAt(position, self.position + self.forward, up_vector)