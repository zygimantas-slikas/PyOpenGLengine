import moderngl as mgl

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

    def use_texture(self, texture_index:int):
        self.program['u_texture_0'] = texture_index