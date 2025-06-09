import os
import numpy as np
from tensorflow import compat as ttf
tf = ttf.v1
tf.disable_v2_behavior()
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

tf.disable_v2_behavior()

# Paths
MODEL_NAME = 'faster_rcnn_resnet_101'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'inference_graph', 'frozen_inference_graph.pb')
PATH_TO_LABELS = 'gtsdb3_label_map.pbtxt'  # Replace with your label map
TEST_IMAGES_DIR = 'dataset/train/fake'  # Folder containing test images
OUTPUT_DIR = 'fakes_detected'  # Folder to save output images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of classes in your label map
NUM_CLASSES = 3

# Load the model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

# Helper function to load an image as a NumPy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Get list of test images
test_image_paths = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

# Run detection on each image
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Input and output tensors
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in test_image_paths:
            # Load image
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Run the model
            (boxes_out, scores_out, classes_out, num_detections_out) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )

            # Visualize the results
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes_out),
                np.squeeze(classes_out).astype(np.int32),
                np.squeeze(scores_out),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.5,
            )

            # Save the output image
            output_image_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
            output_image = Image.fromarray(image_np)
            output_image.save(output_image_path)
            print(f"Processed and saved: {output_image_path}")
