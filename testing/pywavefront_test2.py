


# import pywavefront
# scene = pywavefront.Wavefront('../images/sofa/Sofa 4.obj',  parse=False, create_materials=True)   
# # scene = pywavefront.Wavefront('../images/eyebrows/Transphormator N121015.obj',  parse=False, create_materials=True)   
# # print(type(scene))
# scene.parse()  # Explicit call to parse() needed when parse=False
# print(help(scene.parse()))



# # Iterate vertex data collected in each material
# for name, material in scene.materials.items():
#     # Contains the vertex format (string) such as "T2F_N3F_V3F"
#     # T2F, C3F, N3F and V3F may appear in this string
#     # print(material.vertex_format)
#     print()
#     # Contains the vertex list of floats in the format described above
#     # print (material.vertices)
#     print()
#     # Material properties
#     # print(material.diffuse)
#     print()
#     # print(material.ambient)
#     print()
#     # print(material.textur)




import pywavefront
from pywavefront import visualization


import pyglet
from pyglet.gl import *

window = pyglet.window.Window(resizable=True)


# [create a window and set up your OpenGl context]
# meshes = pywavefront.Wavefront('../images/sofa/Sofa 4.obj',  create_materials=True )
# meshes = pywavefront.Wavefront('../images/eyebrows/eyeBrowLeft.obj',  create_materials=True )
meshes = pywavefront.Wavefront('../images/skeleton/skeleton.obj',  create_materials=True )

# [inside your drawing loop]
# visualization.draw(meshes)

print(type(meshes))
print(dir(meshes))


l = ['add_mesh', 'file_name', 'materials', 'mesh_list', 'meshes', 'mtllibs', 'parse', 'parser', 'parser_cls', 'vertices']

print(meshes.vertices)
print(meshes.file_name)
print(meshes.materials)

# for m in l:
#     print(m,'---------------')
#     print(meshes.m)

@window.event
def on_draw():
    # window.clear()
    
    visualization.draw(meshes)


pyglet.app.run()