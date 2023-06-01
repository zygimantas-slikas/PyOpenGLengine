import math
import numpy as np
from .scene_object import SceneObject
import moderngl as mgl
import glm

class Bone:
    def __init__(self, scene_object, parent, position, scale):
        self.parent:Bone = parent
        self.children:list[Bone] = []

        self.start:np.ndarray = np.array(position, dtype=np.float32)
        self.length = glm.vec3(0, scale[1], 0)
        self.end:np.ndarray = None

        self.rotation = np.array([0, 0, 0], dtype=np.float32)
        self.transformation_matrix:np.ndarray = self.create_matrix(self.rotation)
        self.scene_object:SceneObject = scene_object
        self.scene_object.position = self.start

    def create_matrix(self, rotation:np.ndarray) -> np.ndarray:
        rotation_axis = glm.vec3(1,0,0)
        rotation_matrix = glm.rotate(rotation[0], rotation_axis)

        rotation_axis = glm.vec3(0,1,0)
        rotation_matrix @= glm.rotate(rotation[1], rotation_axis)

        rotation_axis = glm.vec3(0,0,1)
        rotation_matrix @= glm.rotate(rotation[2], rotation_axis)

        return rotation_matrix

    def get_position(self) -> np.ndarray:
        if self.parent is None:
            position = self.start
        else :
            parent = self.parent
            position = glm.mat4x4((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))
            while parent is not None:
                position = position @ self.parent.transformation_matrix
                if parent.parent is not None:
                    parent = parent.parent
                else :
                    break
            position = position @ glm.vec3(parent.start)
        return position

    def set_rotation(self, rotation:np.ndarray)->None:
        self.rotation = rotation
        self.transformation_matrix = self.create_matrix(self.rotation)
        self.update_tale()

    def set_position(self, new_start:np.ndarray)->None:
        self.start = new_start
        self.update_tale()

    def update_tale(self) -> None:
        cumulative_rotation = self.rotation.copy()
        parent = self.parent
        while parent is not None:
            cumulative_rotation += parent.rotation
            parent = parent.parent
        self.end = self.start + self.create_matrix(cumulative_rotation) @ self.length
        self.scene_object.rotation = cumulative_rotation
        self.scene_object.position = self.start
        for child in self.children:
            child.set_position(self.end)


class Skeleton:
    def __init__(self, mesh, light, position:tuple[float, float, float]):
        self.position = position
        self.scene_objects = []

        self.bones:list[Bone] = []
        self.rotations = []
        self.create_bones(mesh, light)

    def create_bones(self, mesh, light) -> None:
        for i in range(5):
            if i != 4:
                sc_1 = SceneObject(mesh, light, position=(0,0,0), scale=(0.2, 1, 0.2))
            else :
                sc_1 = SceneObject(mesh, light, position=(0,0,0), scale=(0.1, 1, 0.1))
            self.scene_objects.append(sc_1)
            if len(self.bones) == 0:
                bone_parent = None
                bone_1 = Bone(sc_1, bone_parent, position=(0,i,0), scale=(0.3, 1, 0.3))
            else :
                bone_parent = self.bones[-1]
                bone_1 = Bone(sc_1, bone_parent, position=(0,i,0), scale=(0.3, 1, 0.3))
                bone_parent.children.append(bone_1)
            self.bones.append(bone_1)

        for i in range(2):
            sc_1 = SceneObject(mesh, light, position=(0,0,0), scale=(0.3, 1, 0.3))
            self.scene_objects.append(sc_1)
            bone_parent = self.bones[1]
            bone_1 = Bone(sc_1, bone_parent, position=(0,i,0), scale=(0.3, 1, 0.3))
            bone_parent.children.append(bone_1)
            self.bones.append(bone_1)

        for i in range(2):
            sc_1 = SceneObject(mesh, light, position=(0,0,0), scale=(0.3, 1, 0.3))
            self.scene_objects.append(sc_1)
            bone_parent = self.bones[3]
            bone_1 = Bone(sc_1, bone_parent, position=(0,i,0), scale=(0.3, 1, 0.3))
            bone_parent.children.append(bone_1)
            self.bones.append(bone_1)


    def update(self, time:int) -> None:
        self.bones[0].set_position(np.array([0, 2, 0], dtype=np.float32))

        #head
        self.bones[0].set_rotation(np.array([0, 0, 1.5], dtype=np.float32))
        self.bones[1].set_rotation(np.array([0, 0, 1.6], dtype=np.float32))

        #spine
        self.bones[2].set_rotation(np.array([ 0, 0, -1.5], dtype=np.float32))
        self.bones[3].set_rotation(np.array([ 0, 0, 0], dtype=np.float32))

        #legs
        self.bones[-4].set_rotation(np.array([ 0.6, 0, 0], dtype=np.float32))
        self.bones[-3].set_rotation(np.array([-0.6, 0, 0], dtype=np.float32))

        self.bones[-2].set_rotation(np.array([ 0.6, 0, 1.5], dtype=np.float32))
        self.bones[-1].set_rotation(np.array([-0.6, 0, 1.5], dtype=np.float32))

        rotation_angle = 0.4*math.sin(time*8)
        new_rotation =  np.array([0, rotation_angle, 0], dtype=np.float32)
        self.bones[4].set_rotation(new_rotation)
        
        for id, bone in enumerate(self.bones[-4:]):
            rotation_angle = 0.2*math.sin(time*3)
            new_rotation = bone.rotation + np.array([0, 0, rotation_angle], dtype=np.float32)
            bone.set_rotation(new_rotation)


