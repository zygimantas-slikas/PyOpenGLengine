import numpy as np
import moderngl as mgl
import glm
import pygame as pg
import light
import camera

class Model:
    def __init__(self, gl_context:mgl.Context, camera_projection:glm.mat4x4,
    view_matrix, light:light.Light)->None:
        self.gl_context = gl_context
        self.vertex_buffer = self.get_vertex_buffer_object()
        self.shader_program = self.get_shader_program("default")
        self.vertex_array = self.get_vertex_array_object()
        self.model_matrix = self.get_model_matrix() 
        self.texture = self.get_texture('textures/test.png')
        
        self.shader_program["light.position"].write(light.position)
        self.shader_program["light.ambient_intensity"].write(light.ambient_intensity)
        self.shader_program["light.diffusion_intensity"].write(light.diffuse_intensity)
        self.shader_program["light.specular_intensity"].write(light.specular_intensity)

        self.shader_program['u_texture_0'] = 0
        self.texture.use(location = 0)
        self.shader_program['camera_projection_matrix'].write(camera_projection)
        self.shader_program['view_matrix'].write(view_matrix)
        self.shader_program['model_matrix'].write(self.model_matrix)

    def get_texture(self, path):
        texture = pg.image.load(path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        # texture.fill('red')
        texture = self.gl_context.texture(size = texture.get_size(), components=3,
        data=pg.image.tostring(texture, 'RGB'))
        return texture

    def get_model_matrix(self):
        model_matrix = glm.mat4()
        return model_matrix

    def update(self, time, view_matrix, camera_position):
        model_matrix = glm.rotate(self.model_matrix, time*0.5, glm.vec3(0,1,0))
        self.shader_program['model_matrix'].write(model_matrix)
        self.shader_program['view_matrix'].write(view_matrix)
        self.shader_program["camera_position"].write(camera_position)

    def render(self, time:float, camera:camera.Camera)->None:
        self.update(time, camera.view_matrix, camera.position)
        self.vertex_array.render()

    def destroy(self)->None:
        self.vertex_buffer.release()
        self.shader_program.release()
        self.vertex_array.release()

    def get_vertex_array_object(self)->mgl.VertexArray:
        vertex_array = self.gl_context.vertex_array(self.shader_program, 
        [(self.vertex_buffer, '2f 3f 3f', 'in_texture_coordinates_0', 'in_normal', 'in_position')])
        return vertex_array

    def get_vertex_data(self)->np.ndarray:
        # vertex_data = np.array([(-0.6, -0.8, 0.0), (0.6, -0.8, 0.0), (0.0, 0.8, 0.0)],
        #  dtype=np.float32)
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

    def get_vertex_buffer_object(self)->mgl.Buffer:
        vertex_data = self.get_vertex_data()
        vertex_buffer = self.gl_context.buffer(vertex_data)
        return vertex_buffer

    def get_shader_program(self, shader_file_name:str)->mgl.Program:
        with open(f'shaders/{shader_file_name}.vert') as file:
            vertex_shader = file.read()
        with open(f'shaders/{shader_file_name}.frag') as file:
            fragment_shader = file.read()
        #compiles shders program on cpu for later use on gpu
        program = self.gl_context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        return program


class Shaders:
    def __init__(self, gl_context:mgl.Context, vertex_shader=None, fragment_shader=None):
        self.gl_context:mgl.Context = gl_context
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.vertex_file_ending=".vert"
        self.fragment_file_ending=".frag"
        self.program:mgl.Program = None

    def delete(self):
        """Deallocates resources from OpenGL."""
        self.program.release()

    def read_from_file(self, file_name:str)->mgl.Program:
        with open("shaders/"+file_name+self.vertex_file_ending) as file:
            vertex_shader = file.read()
        with open("shaders/"+file_name+self.fragment_file_ending) as file:
            fragment_shader = file.read()
        #compiles shders program on cpu for later use on gpu
        program = self.gl_context.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.program = program
        self.program['u_texture_0'] = 0
        return program


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


class Texture:
    def __init__(self, gl_context:mgl.Context, file_path:str=None, fill_color:str=None):
        self.gl_context:mgl.Context = gl_context
        self.texture:pg.Surface = None
        self.file_path:str = None
        self.fill_color:str = None
        if file_path is not None:
            self.file_apth = file_path
            self.load_texture(file_path)
        elif fill_color is not None:
            self.fill_color = fill_color
            self.texture.fill(self.fill_color) 

    def load_texture(self, file_path:str)->pg.Surface:
        self.file_path = file_path
        texture = pg.image.load(file_path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.gl_context.texture(size = texture.get_size(), components=3,
        data=pg.image.tostring(texture, 'RGB'))
        self.texture = texture
        return texture
    

class Mesh:
    def __init__(self, gl_context:mgl.Context, vertex:Vertexes=None,
     texture:Texture=None, shaders:Shaders=None)->None:
        """Creates object for holding mesh objects data, and reference to allocated space on GPU 
        buffer."""
        self.gl_context:mgl.Context = gl_context
        self.vertex:Vertexes = vertex
        self.texture:Texture = texture
        self.shaders:Shaders = shaders
        self.mesh_in_shader:mgl.VertexArray = None
        self.texture.texture.use(location = 0)

    def create_objects_shader(self)->mgl.VertexArray:
        if (self.vertex.buffer is not None
            and self.texture.texture is not None
            and self.shaders.program is not None):
            self.mesh_in_shader = self.gl_context.vertex_array(self.shaders.program, 
            [(self.vertex.buffer, '2f 3f 3f', 'in_texture_coordinates_0', 'in_normal', 'in_position')])
            return self.mesh_in_shader
        else:
            print("Failed to create a mesh. Some data missing.")
            return None
    
    def delete(self):
        """Deallocates resources from OpenGL."""
        self.mesh_in_shader.release()
        self.vertex.delete()
        self.shaders.delete()

    def read_shaders(self, file_name:str)->mgl.Program:
        ...

    def load_object_from_file(self, file_anme:str)->list:
        ...

    def generate_polygons(self, vertex_locations:np.ndarray, texture_locations:np.ndarray):
        ...

class SceneObject:
    def __init__(self, mesh:Mesh, light:light.Light, position=(0,0,0)):
        self.mesh:Mesh = mesh
        self.model_transformation_matrix = glm.mat4()
        self.update_lighting(light)

    def update_lighting(self, light:light.Light):
        self.mesh.shaders.program["light.position"].write(light.position)
        self.mesh.shaders.program["light.ambient_intensity"].write(light.ambient_intensity)
        self.mesh.shaders.program["light.diffusion_intensity"].write(light.diffuse_intensity)
        self.mesh.shaders.program["light.specular_intensity"].write(light.specular_intensity)

    def update(self, time:int, view_matrix, camera_projection_matrix, camera_position, transform:glm.mat4 = None):
        model_matrix = glm.rotate(self.model_transformation_matrix, time*0.5, glm.vec3(0,1,0))
        self.mesh.shaders.program['model_matrix'].write(model_matrix)
        self.mesh.shaders.program['view_matrix'].write(view_matrix)
        self.mesh.shaders.program["camera_position"].write(camera_position)
        self.mesh.shaders.program['camera_projection_matrix'].write(camera_projection_matrix)

    def render(self, time:int, light:light.Light, camera:camera.Camera):
        self.update(time, camera.view_matrix, camera.projection_matrix,camera.position)
        self.mesh.mesh_in_shader.render()

    