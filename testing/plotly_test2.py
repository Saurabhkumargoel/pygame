# plotly_test2.py

import numpy as np
import plotly.graph_objects as go
import sys
import cv2


def obj_data_to_mesh3d(file, odata=None):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    # lines = odata.splitlines()   
   
    with open(file) as objfile:
        for line in objfile:
            slist = line.split()
            if slist:
                if slist[0] == 'v':
                    vertex = np.array(slist[1:], dtype=float)
                    vertices.append(vertex)
                elif slist[0] == 'f':
                    face = []
                    for k in range(1, len(slist)):
                        face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                    if len(face) > 3: # triangulate the n-polyonal face, n>3
                        faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                    else:    
                        faces.append([face[j][0]-1 for j in range(len(face))])
                else: pass
    
    
    return np.array(vertices), np.array(faces) 



# with open('../images/car/car.obj') as objfile:
#   obj_data = objfile.read()

# vertices, faces = obj_data_to_mesh3d(obj_data)
# vertices, faces = obj_data_to_mesh3d('../images/car/car.obj')
vertices, faces = obj_data_to_mesh3d('../images/eyebrows/eyeBrowLeft.obj')
mycolor =  np.tile(np.array([255,120,0]),(28,1)) # creating sample  color to vertex color x,y,z,r,g,b
# print (mycolor)
vertices = np.append(vertices,mycolor,axis=1)

# print (vertices[:,3:][0])
# print (vertices)

x, y, z = vertices[:,:3].T
I, J, K = faces.T

mesh = go.Mesh3d(
            x=x,
            y=y,
            z=-z,
            vertexcolor=vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!                      
            i=I,
            j=J,
            k=K,
            name='',
            showscale=False)


layout = go.Layout(
                    # title='Mesh3d from a Wavefront obj file',
                   # font=dict(size=14, color='black'),
                   # width=900,
                   # height=800,
                   scene=dict(xaxis=dict(visible=True),
                              yaxis=dict(visible=True),  
                              zaxis=dict(visible=True), 
                              # aspectratio=dict(x=1.5,
                              #                  y=0.9,
                              #                  z=0.5
                              #            ),
                              camera=dict(eye=dict(x=1, y=1, z=1)),
                        ), 
                  paper_bgcolor='rgb(235,235,235)',
                  # margin=dict(t=175)
              ) 


fig = go.Figure(data=[mesh], layout=layout)
# fig = go.Figure(data=[mesh])
fig.show()
print(dir(fig))
img_byte = fig.to_image(format='png')

# image_byte to numpy array
decoded = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)
print(decoded)

cv2.imwrite('decoded.png',decoded)
# cv2.imshow('decoded.png',decoded)
# cv2.waitKey(0)

