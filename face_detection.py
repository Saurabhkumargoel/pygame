import logging as log
import os.path as osp
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IEPlugin
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector

DEVICE_KINDS = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', metavar="PATH", default='0',
                         help="(optional) Path to the input video " \
                         "('0' for the camera, default)")
    general.add_argument('-o', '--output', metavar="PATH", default="",
                         help="(optional) Path to save the output video to")
    general.add_argument('--no_show', action='store_true',
                         help="(optional) Do not display output")


    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', metavar="PATH", default="", required=False,
                        help="Path to the Face Detection model XML file")
    models.add_argument('-m_lm', metavar="PATH", default="", required=False,
                        help="Path to the Facial Landmarks Regression model XML file")
    models.add_argument('-m_hp', metavar="PATH", default="", required=False,
                        help="Path to the Facial Head Position Regression model XML file")


    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Detection model (default: %(default)s)")
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Facial Landmarks Regression model (default: %(default)s)")
    infer.add_argument('-d_hp', default='CPU', choices=DEVICE_KINDS,
                       help="(optional) Target device for the " \
                       "Face Facial Head Position model (default: %(default)s)")
    infer.add_argument('-v', '--verbose', action='store_true',
                       help="(optional) Be more verbose")
    
    return parser




def main():
    args = build_argparser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug(str(args))

    # visualizer = Visualizer(args)
    # visualizer.run(args)


    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()


"""

# Set target device and Load HW Plugin
from openvino.inference_engine import IENetwork, IEPlugin

plugin = IEPlugin(device=args.device, plugin_dir=plugin_dir)
...

# Read an IR Model
model_xml = args.model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

# net holds the network information like input and output; we can access
net = IENetwork(model=model_xml, weights=model_bin)

# Load model to plugin
Exec_net = plugin.load(network=net)


# Read data into Input Blob
# Read and preprocess input images 
n,c,h,w = net.inputs[input_blob].shape
images = np.ndarray(shape=(n,c,h,w))
for i in range(n):
	image = cv2.imread(args.input[i])
	if image.shape[::-1] != (h,w):
		log.warning("Image {} is resized from {} to {}")
		image = cv2.resize(image, (w,h))


	image = image.transpose((2,0,1))
	# Change data layout from HWC to CHW
	images[i] image


# inference
res = Exec_net.infer(input={input_blob:images}) # work async


# Process Output from Output Blob

res = res[out_blob]

for i,probs in enumerate(res):
	probs=np.squeeze(probs)
	top_ind = np.argsort(probs)[-args.number_top:][::-1]

	print(f"Image {args.input[i]}")

	for id in top_ind:
		det_label = labels_map[id] if labels_map else f"#{id}"
		print(f"{probs[id]} label det_label")
	print("---" * 50)

"""













