import glm


class Light:
    def __init__(self, position:tuple[float,float,float]=(3, 3, -3),
     color:tuple[float,float,float]=(1,1,1)):
     self.position = glm.vec3(position)
     self.color = glm.vec3(color)
     self.ambient_intensity = 0.1*self.color
     self.diffuse_intensity = 0.8*self.color
     self.specular_intensity = 1.0*self.color