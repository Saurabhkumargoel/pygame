import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils_fswap

print ("Press T to draw the keypoints and the 3D model")
print ("Press R to start recording to a video file")

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
predictor_path = "fswapfiles/shape_predictor_68_face_landmarks.dat"
image_name = "fswapfiles/AK-2.jpeg"
# image_name = "../data/AK-2.jpeg"
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils_fswap.load3DFaceModel("fswapfiles/candide.npz")

print(type(predictor))


# print("mean3DShape", mean3DShape)
# print("blendshapes", blendshapes)
# print("mesh",mesh, len(mesh))
# print("idxs3D", idxs3D)
# print("idxs2D", idxs2D)





projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

textureImg = cv2.imread(image_name)
# print(textureImg.shape)
textureCoords = utils_fswap.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)

print("textureCoords--", textureCoords)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

while True:
    cameraImg = cap.read()[1]
    # print(cameraImg)
    shapes2D = utils_fswap.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    # print('shapes2D--', type(shapes2D), zip(shapes2D[0][0],shapes2D[0][1]))
    # for x,y in zip(shapes2D[0][0],shapes2D[0][1]):
    #     print(x,y)

    if shapes2D is not None:
        for shape2D in shapes2D:

            # print(shape2D[0], len(shape2D[0]))   # list of [[x1,x2,x3.....xn],[y1,y2,y3.....yn]]
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            #rendering the model to an image
            shape3D = utils_fswap.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            #blending of the rendered face with the image
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)

            # apply rendered image on cameraImg
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
       

            #drawing of the mesh and keypoints
            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

    if writer is not None:
        writer.write(cameraImg)

    cv2.imshow('image', cameraImg)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            print ("Starting video writer")
            writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print ("Writer succesfully opened")
            else:
                writer = None
                print ("Writer opening failed")
        else:
            print ("Stopping video writer")
            writer.release()
            writer = None
