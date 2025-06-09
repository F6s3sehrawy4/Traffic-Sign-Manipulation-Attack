import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
from tensorflow import compat as ttf
tf = ttf.v1
tf.disable_v2_behavior()

import sys

# Define the directory to save results
RESULTS_DIR = 'results_test'
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.append('C:\\Users\\farah\\PycharmProjects\\traffic_sign-detection\\models-master\\research')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous'

# Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
MODEL_PATH = os.path.join(MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'inference_graph/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = ('gtsdb3_label_map.pbtxt')

NUM_CLASSES = 3

# Load the frozen model graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'fake(old)'
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# Size, in inches, of the output images.
IMAGE_SIZE = (1600, 900)

# TensorFlow session for inference
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        for idx, image_path in enumerate(TEST_IMAGE_PATHS):
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            print(f"Original image shape: {image_np.shape}")
            image_np_expanded = np.expand_dims(image_np, axis=0)
            print(f"Expanded image shape: {image_np_expanded.shape}")

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            print(f"Detections: {num_detections}, Boxes: {boxes}, Scores: {scores}, Classes: {classes}")

            # Visualization
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=6)

            # Save the image to the results folder
            result_image_path = os.path.join(RESULTS_DIR, f'result_{idx + 1}.jpg')
            Image.fromarray(image_np).save(result_image_path)
            print(f"Image saved to: {result_image_path}")
