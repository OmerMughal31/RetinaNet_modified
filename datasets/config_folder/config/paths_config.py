# import the necessary packages
import os

# initialize the base path for the logos dataset


#############################################################

# Base pathe when working in docker container

# BASE_PATH = "datasets/config_folder/dataset"

#############################################################

#############################################################

# Base pathe when working in system with spyder or jupyter notebook
BASE_PATH = "dataset"

#############################################################


# build the path to the annotations and input images
ANNOT_PATH = os.path.sep.join([BASE_PATH, "annotations"])
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])

# degine the training/testing split
TRAIN_TEST_SPLIT = 1.00
TEST_VAL_SPLIT = 1.00

#  build the path to the output training and test .csv files
TRAIN_CSV = os.path.sep.join([BASE_PATH, "train.csv"])
VAL_CSV = os.path.sep.join([BASE_PATH, "val.csv"])
TEST_CSV = os.path.sep.join([BASE_PATH, "test.csv"])

# build the path to the output classes CSV files
CLASSES_CSV = os.path.sep.join([BASE_PATH, "classes.csv"])

# build the path to the output predictions dir
OUTPUT_DIR = os.path.sep.join([BASE_PATH, "predictions"])
