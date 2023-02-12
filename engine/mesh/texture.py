import numpy as np
import moderngl as mgl
import pygame as pg

class Texture:
    created_textures_count = 0
    def __init__(self, gl_context:mgl.Context, file_path:str=None, fill_color:str=None):
        self.gl_context:mgl.Context = gl_context
        self.texture_data:mgl.Texture = None
        self.file_path:str = None
        self.fill_color:str = None
        self.index:int = None
        if file_path is not None:
            self.file_apth = file_path
            self.load_texture(file_path)
        elif fill_color is not None:
            texture = pg.Surface(size=(50,50))
            self.fill_color = fill_color
            texture.fill(self.fill_color)
            self.texture_data = self.gl_context.texture(size = texture.get_size(), components=3,
                data=pg.image.tostring(texture, 'RGB')) 
        self.texture_data.use(location = Texture.created_textures_count)
        self.index = Texture.created_textures_count
        Texture.created_textures_count += 1

    def load_texture(self, file_path:str)->pg.Surface:
        self.file_path = file_path
        texture = pg.image.load(file_path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.gl_context.texture(size = texture.get_size(), components=3,
        data=pg.image.tostring(texture, 'RGB'))
        self.texture_data = texture
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        texture.anisotropy = 32.0
        return texture