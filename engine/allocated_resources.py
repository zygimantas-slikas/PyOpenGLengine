from . import mesh


class Resources:
    def __init__(self):
        self.vertexes:dict[str, mesh.Vertexes] = {}
        self.textures:dict[str, mesh.Texture] = {}
        self.shaders:dict[str, mesh.Shaders] = {}
        self.meshes:dict[str, mesh.Mesh] = {}

    def delete(self):
        for vertex in self.vertexes.values():
            vertex.delete()
        for shader in self.shaders.values():
            shader.delete()
        for mesh in self.meshes.values():
            mesh.delete()
