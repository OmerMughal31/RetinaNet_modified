# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:39:54 2021

@author: engro
"""

import os

import keras
from keras_retinanet import models
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version

import numpy as np
import cv2
import json

class Evaluation:
    def __init__(self):
        # Reading the classes and respective index from classes.json file
        self.classes = {
                    value["id"] - 1: value["name"]
                    for value in json.load(
                        open("classes.json", "r")
                    ).values()
                }
        self.num_classes = 2
        self.colors_classes = [np.random.randint(0, 256, 3).tolist() for _ in range(self.num_classes)]
        #Threshold on score to filter detections with (defaults to 0.05).
        self.score_threshold = 0.5
        # IoU Threshold to count for a positive detection (defaults to 0.5).
        self.iou_threshold = 0.05
        # Max Detections per image (defaults to 100).
        self.max_detections = 100
        # Setup GPU device
        self.gpu = 0
        setup_gpu(self.gpu)
        # Rescale the image so the smallest side is min_side.
        self.image_min_side = 800
        # Rescale the image if the largest side is larger than max_side.
        self.image_max_side = 1333
        # make save path if it doesn't exist
        self.save_path = "/eveluation"
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # optionally load anchor parameters when the inference model has been generated along with training
        # Provide the path of config of file such as (self.config = "path_to_config_file")
        self.config = None
        self.anchor_params = None
        if self.config and "anchor_parameters" in self.config:
            self.anchor_params = parse_anchor_parameters(self.config)
        
        # Backbone Network
        self.backbone_network = "resnet50"
        self.weight_dir = "snapshots"
        # Model to be evaluated
        self.model_to_load = os.path.join(self.weight_dir, "resnet50_csv_17.h5")
        # Convert the trained model to ind=ference model
        self.convert_model = True
        
        # load the model
        print("Loading model, this may take a second...")
        self.model = models.load_model(self.model_to_load, backbone_name=self.backbone_network)
        self.model = models.convert_model(self.model, anchor_params=self.anchor_params)
    
    def preprocess_image(self, x, mode='caffe'):
        """ 
        Preprocess an image by subtracting the Image mean.
    
        Inputs:
            1- x: np.array of shape (None, None, 3) or (3, None, None).
            2- mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.
    
        Outputs:
            The input with the Image mean subtracted.
        """
        # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already
        # covert always to float32 to keep compatibility with opencv
        x = x.astype(np.float32)
    
        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x -= [103.939, 116.779, 123.68]
    
        return x
        
    def compute_resize_scale(self, image_shape, min_side=800, max_side=1333):
        """ 
        Compute an image scale such that the image size is constrained to min_side and max_side.
    
        Inputs:
            1- min_side: The image's min side will be equal to min_side after resizing.
            2- max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    
        Outputs:
            1- A resizing scale.
        """
        (rows, cols, _) = image_shape
    
        smallest_side = min(rows, cols)
    
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
    
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
    
        return scale
    
    def resize_image(self, img, min_side=800, max_side=1333):
        """ 
        Resize an image such that the size is constrained to min_side and max_side such that image aspect ratio is valid to trained RetinaNet Model.
    
        Inputs:
            1- min_side: The image's min side will be equal to min_side after resizing.
            2- max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    
        Outputs:
            1- A resized image.
        """
        # compute scale to resize the image
        scale = self.compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
        return img, scale
    
    def draw_boxes(self, image, boxes, scores, labels, colors, classes):
        """
        Drawing the results from the predictions (boxes, labels and confidence scores) of specified RetinaNet model on test set (images/video frames).
        Inputs:
            1- image: Image of test set or frame of video in test set
            2- boxes: Predicted bboxes by the RetinaNet model for either a particular images or video frame
            3- scores: Confidence scores for the prection sof each object in images/ video frame
            4- labels: Predicted labels by the RetinaNet model for either a particular images or video frame
            5- colors: Assigned color to the class to quickly recognize the correctness if predicted class
            6- classes: dataset classes
        
        Outputs:
            None
        
        """
        for b, l, s in zip(boxes, labels, scores):
            class_id = int(l)
            class_name = classes[class_id]
            
            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = colors[class_id]
            label = '-'.join([class_name, score])
            
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    def get_detections(self, frame, model, score_threshold, max_detections):
        """ 
        Get the detections from the model.
        Inputs:
            1- frame: Image of test set or frame of video in test set
            2- model: The model to run on the images.
            3- score_threshold: The score confidence threshold to use.
            4- max_detections: The maximum number of detections to use per image.
        Outputs:
            1- image_boxes: Predicted bboxes for object in image/frame
            2- image_scores: Confidence scores for the objects detected in images/frames
            3- image_labels: class labels of objects in images/frames
        """
        raw_image = frame
        image = self.preprocess_image(raw_image.copy())
        image, scale = self.resize_image(image)
    
        if keras.backend.image_data_format() == "channels_first":
            image = image.transpose((2, 0, 1))
    
        # run network
        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))[:3]
        # correct boxes for image scale
        boxes /= scale
        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]
        # select those scores
        scores = scores[0][indices]
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        
        return image_boxes, image_scores, image_labels

if __name__ == "__main__":
    
    evaluate = Evaluation()
    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()
    
    video_detected_boxes = []
    video_detected_labels_ids = []
    video_detected_scores = []
    
    count = 0
    cap = cv2.VideoCapture('datasets/roiId.0_2020-06-30T09_03_21Z_2020-07-08T04_03_26Z_7_0_999.mkv')
    #cap = cv2.VideoCapture(0)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_length)
    success,image = cap.read()
    success = True
    while success and image is not None:
        success,frame = cap.read()
        # Make sure that the number of extracted frames are equal to number of labelled frames
        if (count < video_length-1):
            count += 1
            print(count)
            # Predict the images/ frames of videos
            detected_box, confidence_scores, object_labels = evaluate.get_detections(
                frame,
                evaluate.model, 
                evaluate.score_threshold, 
                evaluate.max_detections
                )
            # Draw the predictions
            evaluate.draw_boxes(
                frame, 
                detected_box, 
                confidence_scores, 
                object_labels, 
                evaluate.colors_classes, 
                evaluate.classes)
            video_detected_boxes.append(detected_box)
            video_detected_labels_ids.append(object_labels)
            video_detected_scores.append(confidence_scores)
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    count = 0
    cap.release()
    cv2.destroyAllWindows()
    
    # Writing the annotations in json file
    annotations_json = {}
    for i, (box, label, score) in enumerate(zip(video_detected_boxes, video_detected_labels_ids, video_detected_scores)):
        annotations_rois = []
        for b,l,s in zip(box, label, score):
            class_id = int(l)
            class_name = evaluate.classes[class_id]
            
            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            label = class_name
            dummy_rois_data = {
                "coords":[
                    xmin, 
                    ymin, 
                    xmax, 
                    ymax],
                "label":label,
                "cofidence_factor":score
                }
            annotations_rois.append(dummy_rois_data)
        annotations_json[str(i)] = {
            "event": None,
            "label": None,
            "rois": annotations_rois
            }
    with open('data.json', 'w') as outfile:
        json.dump(annotations_json, outfile)
