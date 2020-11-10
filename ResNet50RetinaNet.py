import keras

import sys

sys.path.insert(0, "../")


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)

model_path = os.path.join("snapshots/ResNet50_Weights", "resnet50_csv_149.h5")

# load retinanet model
model = models.load_model(model_path, backbone_name="resnet50")

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {
    0: "cargo_ship",
    1: "truck",
    2: "helicopter",
    3: "sailboat",
    4: "soldier",
    5: "cruise_ship",
    6: "person",
    7: "controller",
    8: "bicycle",
    9: "car",
    10: "bird",
    11: "sairplane",
    12: "drone",
}

# load image
counter = 1
print(counter)
for image_path in glob.glob("datasets/config_folder/dataset/images_test_set/*.jpg"):
    image = read_image_bgr(image_path)
    print(image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

        cv2.imwrite("evaluated_images/eveluated_image_" + str(counter) + ".jpg", draw)

        counter += 1
    # plt.figure(figsize=(15, 15))
    # plt.axis("off")
    # plt.imshow(draw)
    # plt.show()

