import math
import numpy as np
from .scene_object import SceneObject
import moderngl as mgl
import glm

class Bone:
    def __init__(self, scene_object, parent, position, scale):
        self.parent:Bone = parent
        self.start:np.ndarray = np.array(position, dtype=np.float32)
        self.length = glm.vec3(0, scale[1], 0)
        self.end:np.ndarray = None
        self.rotation = np.array([0, 0, 0], dtype=np.float32)
        self.transformation_matrix:np.ndarray
        self.scene_object:SceneObject = scene_object
        self.create_matrix()

    def create_matrix(self) -> np.ndarray:
        rotation_axis = glm.vec3(1,0,0)
        rotation = glm.rotate(self.rotation[0], rotation_axis)

        rotation_axis = glm.vec3(0,1,0)
        rotation @= glm.rotate(self.rotation[1], rotation_axis)

        rotation_axis = glm.vec3(0,0,1)
        rotation @= glm.rotate(self.rotation[2], rotation_axis)

        # shift = glm.vec3(self.start)
        transformation = glm.translate(rotation, self.length)
        self.transformation_matrix = transformation
        self.end = transformation @ self.start
        parent = self.parent
        # if parent is not None:
        #     self.transformation_matrix = parent.transformation_matrix @ self.transformation_matrix

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


class Skeleton:
    def __init__(self, mesh, light, position:tuple[float, float, float]):
        self.position = position
        self.scene_objects = []

        self.bones = []
        self.rotations = []
        self.create_bones(mesh, light)

    def create_bones(self, mesh, light) -> None:
        for i in range(5):
            sc_1 = SceneObject(mesh, light, position=(0,0,0), scale=(0.3, 1, 0.3))
            self.scene_objects.append(sc_1)
            if len(self.bones) == 0:
                bone_parent = None
            else :
                bone_parent = self.bones[-1]
            bone_1 = Bone(sc_1, bone_parent, position=(0,i,0), scale=(0.3, 1, 0.3))
            self.bones.append(bone_1)
 
    def update(self, time:int) -> None:
        
        # self.bones[0].start[2] = math.cos(time*2)
        for id, bone in enumerate(self.bones[:]):
            bone:Bone
            bone.scene_object.position = bone.get_position()
            
            if id == 0:
                bone.rotation = np.array([0.1*math.sin(time*2), 0, 0], dtype=np.float32)
            else :
                bone.rotation = np.array([0, 0, 0], dtype=np.float32)
            bone.create_matrix()
            
            rotation = bone.rotation.copy()
            parent = bone.parent
            while parent is not None:
                rotation += parent.rotation
                parent = parent.parent
            # rotation = np.array([rotation, 0, 0], dtype=np.float32)

            bone.scene_object.rotation = rotation


            # bone.start[1] += math.sin(time*0.01)
            # bone.scene_object.position[1] += 0.01*math.sin(time*10)
        pass

