# pywavefront_test.py


# import pywavefront
# from pywavefront import visualization


# # [create a window and set up your OpenGl context]
# obj = pywavefront.Wavefront('../images/eyebrows/eyeBrowLeft.obj')

# # [inside your drawing loop]
# visualization.draw(obj)



#!/usr/bin/env python
"""This script shows an example of using the PyWavefront module."""
import ctypes
import os
import sys

sys.path.append('..')

import pyglet
from pyglet.gl import *

from pywavefront import visualization
import pywavefront

# Create absolute path from this module
# file_abspath = os.path.join(os.path.dirname(__file__), '../images/eyebrows/Transphormator N121015.obj')
# file_abspath = os.path.join(os.path.dirname(__file__), '../images/sofa/Sofa 4.obj')
file_abspath = os.path.join(os.path.dirname(__file__), '../images/skeleton/skeleton.obj')
# file_abspath = os.path.join(os.path.dirname(__file__), '../images/eyebrows/eyeBrowLeft.obj')

rotation = 0
meshes = pywavefront.Wavefront(file_abspath,create_materials=True)
window = pyglet.window.Window(resizable=False)
lightfv = ctypes.c_float * 4


@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(0.0, 0.0, -3.0)
    glRotatef(rotation, 0.0, 1.0, 0.0)
    glRotatef(-25.0, 1.0, 0.0, 0.0)
    glRotatef(45.0, 0.0, 0.0, 1.0)

    glEnable(GL_LIGHTING)

    visualization.draw(meshes)


def update(dt):
    global rotation
    rotation += 40.0 * dt

    if rotation > 360.0:
        rotation = 0.0
        pass


pyglet.clock.schedule(update)
pyglet.app.run()