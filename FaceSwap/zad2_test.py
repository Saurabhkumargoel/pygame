import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

from landmarks_detector import LandmarksDetector


def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(model_description_path), \
            "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), \
            "Model weights are not found at '%s'" % (model_weights_path)
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model


used_devices = set([args.d_fd, args.d_lm, args.d_hp])
self.context = InferenceContext()
context = self.context
context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
for d in used_devices:
    context.get_plugin(d).set_config({
        "PERF_COUNT": "YES" if args.perf_stats else "NO"})

log.info("Loading models")
face_detector_net = self.load_model(args.m_fd)
landmarks_net = self.load_model(args.m_lm)
head_pose_net = self.load_model(args.m_hp)
# face_reid_net = self.load_model(args.m_reid)

self.face_detector = FaceDetector(face_detector_net,
                                  confidence_threshold=args.t_fd,
                                  roi_scale_factor=args.exp_r_fd)

self.landmarks_detector = LandmarksDetector(landmarks_net)
self.head_pose_detector = HeadPoseDetector(head_pose_net)
self.face_detector.deploy(args.d_fd, context)
self.landmarks_detector.deploy(args.d_lm, context,
                               queue_size=self.QUEUE_SIZE)
self.head_pose_detector.deploy(args.d_hp, context,
                               queue_size=self.QUEUE_SIZE)

print ("Press T to draw the keypoints and the 3D model")
print ("Press R to start recording to a video file")

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"

image_name = "../data/jolie.jpg"
# image_name = "../data/AK-2.jpeg"
# image_name = "/home/divesh/OPENVINO/facedetection/images/eyebrows/e10.png"
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")

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

# textureImg = cv2.imread(image_name)
textureImg = cameraImg #  tetsing
# print(textureImg.shape)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)

def get_vertices(obj_file='/home/divesh/OPENVINO/facedetection/images/eyebrows/eyeBrowLeft.obj'):

    vertices = []
    faces = []

    with open(obj_file) as objfile:
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
    faces = np.array(faces) 
    vertices = np.array(vertices)
    # print (mycolor)
    # vertices = np.append(vertices,mycolor,axis=1)

    return vertices


# textureCoords = get_vertices()
print("textureCoords--", textureCoords)

renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

while True:
    cameraImg = cap.read()[1]
    # print(cameraImg)
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    print('shapes2D--', type(shapes2D), zip(shapes2D[0][0],shapes2D[0][1]))
    print(shapes2D)
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
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
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
