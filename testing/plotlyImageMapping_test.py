# %matplotlib inline
# import matplotlib.pyplot as plt
import cv2
import numpy as np
img=cv2.imread('../images/eyebrows/e10.png')
cv2.imshow('img', img)
cv2.waitKey(0)
print ('image shape', img.shape)


my_img=img[10:-10, :, :]
my_img.shape


cv2.imshow('my_img', my_img)
cv2.waitKey(0)

# cv2.imshow('my_imgcrop', my_img[:,:, 2])
cv2.imshow('my_imgcrop', my_img[:,:, 2])
cv2.waitKey(0)

pl_img=my_img[:,:, 2] # 

print("pl_img.shape---", pl_img.shape)
# L, C=pl_img.shape
C, L=pl_img.shape
# assert L==C


plotly_blue='rgb(68, 122, 219)'# the  blue color in Plotly logo


# import plotly.plotly as py
from plotly.graph_objs import *


pl_scl=[ [0.0, 'rgb(150,75,0)'], #plotly_blue
         [0.5, 'rgb(150,75,0)'],
         [0.5, 'rgb(255,255,255)' ], #white
         [1.0, 'rgb(255,255,255)' ]]


# Prepare data to represent a cube face as a Plotly Surface:

x=np.linspace(0, L-1,  L)
y=np.linspace(0, L-1, L)
X, Y = np.meshgrid(x, y)



zm=np.zeros(X.shape)
zM=(L-1)*np.ones(X.shape)


def make_cube_face(x,y,z, colorscale=pl_scl, is_scl_reversed=False, 
                   surfacecolor=pl_img, text='Plotly cube'):
    return Surface(x=x, y=y, z=z,
                   colorscale=colorscale,
                   reversescale=is_scl_reversed,
                   showscale=False,
                   surfacecolor=surfacecolor,
                   text=text,
                   hoverinfo='text'
                  )



trace_zm=make_cube_face(x=X, y=Y, z=zm,  is_scl_reversed=True, surfacecolor=pl_img)
trace_zM=make_cube_face(x=X, y=Y, z=zM,  is_scl_reversed=True, surfacecolor=np.flipud(pl_img))
trace_xm=make_cube_face(x=zm, y=Y, z=X, surfacecolor=np.flipud(pl_img))
trace_xM=make_cube_face(x=zM, y=Y, z=X, surfacecolor=pl_img)
trace_ym=make_cube_face(x=Y, y=zm, z=X, surfacecolor=pl_img)
trace_yM=make_cube_face(x=Y, y=zM, z=X, surfacecolor=np.fliplr(pl_img))


noaxis=dict( 
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks='',
            title='',
            zeroline=False)


min_val=-0.01
max_val=L-1+0.01

layout = Layout(
         title="",
         width=500,
         height=500,
         scene=Scene(xaxis=XAxis(noaxis, range=[min_val, max_val]),
                     yaxis=YAxis(noaxis, range=[min_val, max_val]), 
                     zaxis=ZAxis(noaxis, range=[min_val, max_val]), 
                     aspectratio=dict(x=1,
                                      y=1,
                                      z=1
                                     ),
                     camera=dict(eye=dict(x=-1.25, y=-1.25, z=1.25)),
                    ),
         
        paper_bgcolor='rgb(240,240,240)',
        hovermode='closest',
        margin=dict(t=50)
        )


fig=Figure(data=Data([trace_zm, trace_zM, trace_xm, trace_xM, trace_ym, trace_yM]), layout=layout)
fig.show()
# py.sign_in('empet', 'api_key')
# py.iplot(fig, filename='Plotly-cube')























# %matplotlib inline
# import matplotlib.pyplot as plt
# import numpy as np
# img=plt.imread('Data/Plotly-logo3.png')
# plt.imshow(img)
# print 'image shape', img.shape


# my_img=img[10:-10, :, :]
# my_img.shape


# plt.imshow(my_img)

# pl_img=my_img[:,:, 2] # 
# L, C=pl_img.shape
# assert L==C


# plotly_blue='rgb(68, 122, 219)'# the  blue color in Plotly logo


# import plotly.plotly as py
# from plotly.graph_objs import 


# pl_scl=[ [0.0, 'rgb(68, 122, 219)'], #plotly_blue
#          [0.5, 'rgb(68, 122, 219)'],
#          [0.5, 'rgb(255,255,255)' ], #white
#          [1.0, 'rgb(255,255,255)' ]]


# # Prepare data to represent a cube face as a Plotly Surface:

# x=np.linspace(0, L-1,  L)
# y=np.linspace(0, L-1, L)
# X, Y = np.meshgrid(x, y)



# zm=np.zeros(X.shape)
# zM=(L-1)*np.ones(X.shape)


# def make_cube_face(x,y,z, colorscale=pl_scl, is_scl_reversed=False, 
#                    surfacecolor=pl_img, text='Plotly cube'):
#     return Surface(x=x, y=y, z=z,
#                    colorscale=colorscale,
#                    reversescale=is_scl_reversed,
#                    showscale=False,
#                    surfacecolor=surfacecolor,
#                    text=text,
#                    hoverinfo='text'
#                   )



# trace_zm=make_cube_face(x=X, y=Y, z=zm,  is_scl_reversed=True, surfacecolor=pl_img)
# trace_zM=make_cube_face(x=X, y=Y, z=zM,  is_scl_reversed=True, surfacecolor=np.flipud(pl_img))
# trace_xm=make_cube_face(x=zm, y=Y, z=X, surfacecolor=np.flipud(pl_img))
# trace_xM=make_cube_face(x=zM, y=Y, z=X, surfacecolor=pl_img)
# trace_ym=make_cube_face(x=Y, y=zm, z=X, surfacecolor=pl_img)
# trace_yM=make_cube_face(x=Y, y=zM, z=X, surfacecolor=np.fliplr(pl_img))


# noaxis=dict( 
#             showbackground=False,
#             showgrid=False,
#             showline=False,
#             showticklabels=False,
#             ticks='',
#             title='',
#             zeroline=False)


# min_val=-0.01
# max_val=L-1+0.01

# layout = Layout(
#          title="",
#          width=500,
#          height=500,
#          scene=Scene(xaxis=XAxis(noaxis, range=[min_val, max_val]),
#                      yaxis=YAxis(noaxis, range=[min_val, max_val]), 
#                      zaxis=ZAxis(noaxis, range=[min_val, max_val]), 
#                      aspectratio=dict(x=1,
#                                       y=1,
#                                       z=1
#                                      ),
#                      camera=dict(eye=dict(x=-1.25, y=-1.25, z=1.25)),
#                     ),
         
#         paper_bgcolor='rgb(240,240,240)',
#         hovermode='closest',
#         margin=dict(t=50)
#         )


# fig=Figure(data=Data([trace_zm, trace_zM, trace_xm, trace_xM, trace_ym, trace_yM]), layout=layout)
# py.sign_in('empet', 'api_key')
# py.iplot(fig, filename='Plotly-cube')