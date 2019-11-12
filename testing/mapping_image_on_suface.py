# mapping_image_on_suface.py

# import plotly.plotly as py
from plotly.graph_objs import *

import numpy as np

x=np.linspace(0,5, 200)
y=np.linspace(5, 10, 200)
X,Y=np.meshgrid(x,y)
z=(X+Y)/(2+np.cos(X)*np.sin(Y))


# import matplotlib.pyplot as plt
import matplotlib.cm as cm
# %matplotlib inline
import cv2

# img = cv2.imread('../images/eyebrows/e10.png')
img = cv2.imread('/home/divesh/Pictures/AK/AK.jpeg')
print(type(img), img.shape)


cv2.imshow('img', img )
cv2.waitKey(0)


# def mpl_to_plotly(cmap, pl_entries):
#     h=1.0/(pl_entries-1)
#     pl_colorscale=[]
#     for k in range(pl_entries):
#         C=map(np.uint8, np.array(cmap(k*h)[:3])*255)
#         pl_colorscale.append([round(k*h,2), 'rgb'+str((C[0], C[1], C[2]))])
#     return pl_colorscale


# pl_grey=mpl_to_plotly(cm.Greys_r, 21)
# print("pl_grey--", type(pl_grey), pl_grey)

surf=Surface(x=x, y=y, z=z,
             # colorscale=img,
             surfacecolor=img,
             showscale=False
            )

noaxis=dict( 
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks='',
            title='',
            zeroline=False)
layout = Layout(
         title='Mapping an image onto a surface', 
         font=Font(family='Balto'),
         width=800,
         height=800,
         scene=Scene(xaxis=XAxis(noaxis),
                     yaxis=YAxis(noaxis), 
                     zaxis=ZAxis(noaxis), 
                     aspectratio=dict(x=1,
                                      y=1,
                                      z=0.5
                                     ),
                    )
        )


fig=Figure(data=[surf], layout=layout)
# py.sign_in('empet', 'api_key')
fig.show()
# py.iplot(fig, filename='mappingLenaSurf')



