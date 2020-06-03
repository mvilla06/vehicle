#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from utils import visualization_utils as vis_util
from utils import label_map_util
import tensorflow as tf
import numpy as np
import os
import cv2

# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
MIN_SCORE_THRESH = 0.5

def cocoDetectionCallback(cameraImageMsg, args):
    bridge, detectionGraph, sess, classIndexMapping, cocoDetectionPub = args

    cameraImage = bridge.imgmsg_to_cv2(
        cameraImageMsg, desired_encoding='passthrough').copy()

    cameraImageExpanded = np.expand_dims(cameraImage, axis=0)
    imageTensor = detectionGraph.get_tensor_by_name('image_tensor:0')
    boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
    scores = detectionGraph.get_tensor_by_name('detection_scores:0')
    classes = detectionGraph.get_tensor_by_name('detection_classes:0')
    numDetections = detectionGraph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, numDetections],
        feed_dict={imageTensor: cameraImageExpanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        cameraImage,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        classIndexMapping,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=MIN_SCORE_THRESH)
    
    cocoDetectionPub.publish(bridge.cv2_to_imgmsg(cameraImage, encoding='bgr8'))


def listener():
    rospy.init_node('cocoDetection', anonymous=False)
    bridge = CvBridge()

    # Load detection graph (into default graph)
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        odGraphDef = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

    # Load label mappings
    labelMap = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        labelMap, max_num_classes=NUM_CLASSES, use_display_name=True)
    classIndexMapping = label_map_util.create_category_index(categories)

    # Initialize Session
    sess = tf.compat.v1.Session(graph=detectionGraph)

    cocoDetectionPub = rospy.Publisher(
        '/camera/cocoDetection', Image, queue_size=1)
    rospy.Subscriber(
        '/multisense/left/image_rect_color', Image, cocoDetectionCallback,
        (bridge, detectionGraph, sess, classIndexMapping, cocoDetectionPub))

    rospy.spin()


if __name__ == '__main__':
    listener()
