
import numpy as np
import plotly.graph_objects as go
import sys
import cv2


class EyeBrows:

    def __init__(self,obj_file):

        self.obj_file = obj_file
        self.get_vertices()
        self.create_mesh()
        self.apply_layout()


    def get_vertices(self):

        vertices = []
        faces = []

        with open(self.obj_file) as objfile:
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
    
    
        mycolor =  np.tile(np.array([255,120,0]),(28,1)) # creating sample  color to vertex color x,y,z,r,g,b
        # return np.array(vertices), np.array(faces) 
        self.faces = np.array(faces) 
        self.vertices = np.array(vertices)
        # print (mycolor)
        self.vertices = np.append(self.vertices,mycolor,axis=1)


    def create_mesh(self):

        x, y, z = self.vertices[:,:3].T
        I, J, K = self.faces.T

        self.mesh = go.Mesh3d(
            x=x,
            y=y,
            z=-z,
            vertexcolor=self.vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!                      
            i=I,
            j=J,
            k=K,
            name='',
            showscale=False)


    def apply_layout(self):

        self.layout = go.Layout(
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
                                  camera=dict(eye=dict(x=2, y=1, z=1)),
                            ), 
                      paper_bgcolor='rgb(235,235,235)',
                      # margin=dict(t=175)
                  ) 


    def update_layout(self,update_dict):

        self.fig.layout.update(dict1=update_dict)
        img_byte = fig.to_image(format='png')

        img_decoded = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)
        return img_decoded


    def get_image(self,angle=0):

        self.fig = go.Figure(data=[self.mesh], layout=self.layout)

        self.fig.show()
        # print(dir(fig))
        img_byte = self.fig.to_image(format='png')

        # image_byte to numpy array
        img_decoded = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)


        img_decoded = img_decoded[:,:,:3]
        # cv2.imshow('img_decoded',img_decoded)
        # cv2.waitKey(0)

        # print(img_decoded)
        # print(type(img_decoded))

        return img_decoded




if __name__ == "__main__":
    eb = EyeBrows('images/eyebrows/eyeBrowLeft.obj')
    eb.get_image()


