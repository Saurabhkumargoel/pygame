# import plotly.graph_objects as go
# import numpy as np
# import sys
# import pywavefront

# # meshes = pywavefront.Wavefront('../images/input_obj_zip/image02616.obj',  create_materials=True )
# meshes = pywavefront.Wavefront('../images/skeleton/skeleton.obj',  create_materials=True )
# print(meshes.vertices[0])
# sys.exit()
# # print(meshes.vertices)
# x,y,z, r,g,b = np.array(meshes.vertices).T

# # meshes = pywavefront.Wavefront('../images/eyebrows/eyeBrowLeft.obj',  create_materials=True )
# # x,y,z = np.array(meshes.vertices).T
# print(x,y,z)


# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='green', opacity=0.50)])
# fig.show()





import plotly.graph_objects as go
import numpy as np
import sys
import pywavefront

meshes = pywavefront.Wavefront('../images/eyebrows/eyeBrowLeft.obj',  create_materials=True )
# x,y,z = np.array(meshes.vertices).T

# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='green', opacity=0.50)])
# fig.write_image('hello.png')


import pyrender
# generate mesh
# sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.8)
# meshes.vertices+=1e-2*np.random.randn(*meshes.vertices.shape)
mesh = pyrender.Mesh.from_trimesh(meshes, smooth=False)


# compose scene
scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[0, 0, 0])
camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

scene.add(mesh, pose=  np.eye(4))
scene.add(light, pose=  np.eye(4))

c = 2**-0.5
scene.add(camera, pose=[[ 1,  0,  0,  0],
                        [ 0,  c, -c, -2],
                        [ 0,  c,  c,  2],
                        [ 0,  0,  0,  1]])

# render scene
r = pyrender.OffscreenRenderer(512, 512)
color, _ = r.render(scene)

plt.figure(figsize=(8,8)), plt.imshow(color);