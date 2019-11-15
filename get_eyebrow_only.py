
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Surface
import math
import cv2
import time


class EyeBrows:

    def __init__(self, obj_file='images/eyebrows/eyeBrowLeft.obj'):

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
            z=z,
            vertexcolor=self.vertices[:, 3:], #the color codes must be triplets of floats  in [0,1]!!                      
            i=I,
            j=J,
            k=K,
            name='',
            showscale=True)

        # print(self.mesh)

    def apply_layout(self):

        self.layout = go.Layout(
                # title='Mesh3d from a Wavefront obj file',
               # font=dict(size=14, color='black'),
               # width=900,
               # height=800,
               scene=dict(xaxis=dict(visible=True,
                                    backgroundcolor="rgb(200, 200, 230)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white"),
                        yaxis=dict(visible=True, 
                                    backgroundcolor="rgb(230, 200,230)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white"),  
                        zaxis=dict(visible=True,
                                    backgroundcolor="rgb(230, 230,200)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white"), 
                        xaxis_title='X AXIS TITLE',
                        yaxis_title='Y AXIS TITLE',
                        zaxis_title='Z AXIS TITLE',
                        dragmode='orbit',
                          # aspectratio=dict(x=1.5,
                          #                  y=0.9,
                          #                  z=0.5
                          #            ),
                          camera=dict( 
                                        up=dict(x=0, y=1, z=0),
                                        # center=dict(x=0, y=0, z=0),
                                        eye=dict(x=0, y=0, z=-2.5)
                                        # eye=dict(x=0, y=0, z=-2.5),
                                        # center=dict(x=0, y=0, z=0),
                                        # up=dict(x=1, y=0, z=0),
                                    ),
                    ), 
              paper_bgcolor='rgb(255,255,255)',
              # margin=dict(t=175)
          ) 



    def update_layout(self,angle={}):

        # print(angle.get('angle_up_down',0))
        # print(angle.get('angle_head_tilt',0))
        # print(angle.get('angle_left_right',0))

        angle_up_down = math.radians(angle.get('angle_up_down',0))
        angle_head_tilt = math.radians(angle.get('angle_head_tilt',0))
        angle_left_right = math.radians(angle.get('angle_left_right',0))

        camera = dict(
                up=dict(x=math.sin(angle_head_tilt), y=math.cos(angle_head_tilt), z=0) # for head tilt
                # up=dict(x=math.sin(angle_head_tilt), y=math.cos(angle_head_tilt), z=0) # for up down
                # eye=dict(x=-2.5, y=0, z=0)
            )

        self.fig.update_layout(scene_camera=camera) # scene_dragmode='orbit', title=str(camera)

        img_byte = self.fig.to_image(format='png')

        # image_byte to numpy array
        img_decoded = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)
        self.img_decoded = img_decoded[:,:,:3]

        self.img_decoded[np.where((self.img_decoded==[255,255,255]).all(axis=2))] = [0,0,0];

        return self.img_decoded, self.img_decoded



    def get_image(self,angle={}):


        self.fig = go.Figure(data=[self.mesh], layout=self.layout)

        # surface = Surface({
        #     'colorscale':'YlGnBu',
        #     'surfacecolor':eb_png
        # })

        # print(dir(self.fig.layout.template))
        # print(self.fig['layout']['template'])
        # self.fig.layout.template.update(data={'surface':[{'colorbar': {'outlinewidth': 1, 'ticks': ''},
        #          # 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'],
        #          #                [0.2222222222222222, '#7201a8'],
        #          #                [0.3333333333333333, '#9c179e'],
        #          #                [0.4444444444444444, '#bd3786'],
        #          #                [0.5555555555555556, '#d8576b'],
        #          #                [0.6666666666666666, '#ed7953'],
        #          #                [0.7777777777777778, '#fb9f3a'],
        #          #                [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']],
        #         'surfacecolor':eb_png[:,:,:],
        #          'type': 'surface'}]})

        
        # json = self.fig.to_json(self.fig,{'pretty':True, 'validate':False, 'remove_uids':False}) 
        # print(self.fig['layout']['template']['data']['surface'])





        self.fig.show(validate=False)




        # img_byte = self.fig.to_image(format='png') # -----------------------------------------------------

        # # image_byte to numpy array
        # img_decoded = cv2.imdecode(np.frombuffer(img_byte, np.uint8), -1)
        # # remove 4th channel
        # self.img_decoded = img_decoded[:,:,:3]

        # # to convert it into black
        # self.img_decoded[np.where((self.img_decoded==[255,255,255]).all(axis=2))] = [0,0,0];






if __name__ == "__main__":

    eb = EyeBrows()
    eb.get_image()


    # update_layout(scene_camera=camera, scene_dragmode='orbit', title=name)


