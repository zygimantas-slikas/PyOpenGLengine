import sys
import numpy as np
import pygame as pg
import pygame.locals
import moderngl as mgl
import engine


class GraphicsEngine:
    def __init__(self, win_size=(1200, 700)):
        pg.init()
        self.WIN_SIZE = win_size
        self.FPS = 144
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.gl_context = mgl.create_context()
        self.gl_context.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

        self.clock:pg.time.Clock = pg.time.Clock()
        self.time:float = 0
        self.delta_time:int = 0
        
        self.camera = engine.Camera(self.WIN_SIZE)
        self.resources = engine.Resources()
        self.scene = engine.Scene(self.resources, self.camera, self.gl_context)
        self.scene.initiate_resources()
        self.scene.create_objects_instances()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.resources.delete()
                pg.quit()
                sys.exit()

    def render(self):
        #resset screen
        self.gl_context.clear(color=(0.08, 0.16, 0.18))
        #swap buffers in bouble buffer rendering
        # self.model_1.render(self.time, self.camera)
        self.scene.render(self.time)
        pg.display.flip()

    def get_time(self)->float:
        self.time = pg.time.get_ticks() * 0.001
        return self.time

    def run(self):
        while True:
            self.get_time()
            self.check_events()
            self.camera.update(self.delta_time)
            self.render()
            self.delta_time:int = self.clock.tick(self.FPS)

if __name__ == "__main__":
    app = GraphicsEngine()
    app.run()
