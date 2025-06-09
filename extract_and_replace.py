import os
import cv2
import random
import numpy as np
from PIL import Image
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

REPLACEMENT_DIR = 'replacements'  # Folder containing replacement images
RESULTS_DIR = 'dataset/test/fake'  # Folder to save results

os.makedirs(RESULTS_DIR, exist_ok=True)
import sys

sys.path.append('C:\\Users\\farah\\PycharmProjects\\traffic_sign-detection\\models-master\\research')

from utils import label_map_util

MODEL_NAME = 'faster_rcnn_resnet_101'

# Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
MODEL_PATH = os.path.join(MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'inference_graph/frozen_inference_graph.pb')

# Load all replacement image paths
replacement_images = [os.path.join(REPLACEMENT_DIR, f) for f in os.listdir(REPLACEMENT_DIR) if
                      f.endswith(('jpg', 'png', 'jpeg'))]

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

PATH_TO_TEST_IMAGES_DIR = 'real_scenes'
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# Size of the output images.
IMAGE_SIZE = (1600, 900)

REAL_DIR = 'dataset/test/real'  # Directory to save original unmodified images
os.makedirs(REAL_DIR, exist_ok=True)

# TensorFlow session for inference
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        for idx, image_path in enumerate(TEST_IMAGE_PATHS):
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            h, w, _ = image_np.shape  # Get image dimensions

            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Track if the image was modified
            is_modified = False

            # Process detections
            for i in range(int(num_detections)):
                if scores[0][i] > 0.5:  # Confidence threshold
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    (startX, startY, endX, endY) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

                    # Randomly choose a replacement image
                    replacement_image_path = random.choice(replacement_images)
                    replacement_image = cv2.imread(replacement_image_path)

                    # Resize replacement image to match bounding box size
                    replacement_resized = cv2.resize(replacement_image, (endX - startX, endY - startY))

                    # Replace detected region with resized image
                    image_np[startY:endY, startX:endX] = replacement_resized

                    # Mark as modified
                    is_modified = True

            # Save the original image if it was modified
            if is_modified:
                original_image_path = os.path.join(REAL_DIR, f'original_{idx + 1}.jpg')
                Image.open(image_path).save(original_image_path)  # Save the unmodified original image
                print(f"Original image saved to: {original_image_path}")

                # Save the modified image
                result_image_path = os.path.join(RESULTS_DIR, f'replaced_result_{idx + 1}.jpg')
                Image.fromarray(image_np).save(result_image_path)
                print(f"Modified image saved to: {result_image_path}")
            else:
                print(f"No modifications made to image: {image_path}")
