# -*- coding: utf-8 -*-
"""
Created on Fri April  3 13:30:43 2020

@author: Omer
"""

"""
This file has been written to generate the training, testing and validation datasets for EffcienctDet.


1- Dataset has been labelled with LabelImg.
2- Annotation files are in XML format
3- The outputs are train, test, val and classes files which are CSV files generated from XML format annotations.

Usage:
    1- Please follow the same director structure to optimally use the code.
        parent:
            -datasets
                --config_folder
                    --config
                        --paths_config.py
                    --annotations
                    --images
                    --build_dataset.py
    
    2- Configuration of base paths and general directory settings
        i- There is a director named "config" has been created where a PY file names as "paths_config.py" is given.
        ii- If you are working in docker container, then
            a- In path-config file change the "BASE_PATH" variable with the aboslute path of parent directory where this file is created. (actually: "datasets/config_folder/dataset")
        
        iii- If you are working with spyder of jupyter notebook in python enviornmrnt, then,
            a- In path-config file change the "BASE_PATH" variable with the annotation director path. (actually: "dataset")
            
            #####
                However this path management is commented in the config file. If you will be using the same directory arrangements, then, you just need to uncomment the path according to working enviornment. Otherwise, please define the base paths carefully.
                #####
        
    3- Place XML annotation files in folder annoptations
    4- Place images into images folder
    5- Output CSV files are present in "dataset" directory. NOT IN "datasets" DIRECTORY
    
    6- IMPORTANT
        i- First generate the train.csv and val.csv by setting the --split argument 0.5/0.8/0.9 whatever the user want. It will generate the train and validation set CSV files. Move the train.csv, val.csv and class.csv to 'protected_folder' directory.
        ***Please put the annotations of for train and test set only in annotation folder at this time.
        ***Please put the images of for train and test set only in images folder at this time.
        
        ii- To generate the test set, please put a none zero (1.0 always) value for --splittest argument vale while keeping the --split argument 0.0. It will generate the permant test set which is not going to change and independent from the concept that howm many times you gnerate your training and validation annotations.
        ####
            Replace the empty generated train.csv, val.csv and classes.csv files with CSV files from 'protected_folder'.
            ####
            
        ***Please put the annotations of for only test set in annotation folder at this time.
        ***Please put the images of for test set only in images folder at this time.
        
        ####
            This is a kind a limitation of program but this strategy has been followed to keep the test set always similar for every model or version of training, while, changing the train or validation set. So that the evaluation of every model is done on the same test set.
            
"""


# import the necessary packages
from config import paths_config as config
from bs4 import BeautifulSoup
from imutils import paths
import argparse
import random
import os


class DataPreparationXml:

    """
    Functions to define the arguments whcih are always initialted whenever the object of the class is created.

    """

    def __init__(self):
        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-a", "--annotations", default=config.ANNOT_PATH, help="path to annotations"
        )
        ap.add_argument(
            "-i", "--images", default=config.IMAGES_PATH, help="path to images"
        )
        ap.add_argument(
            "-t",
            "--train",
            default=config.TRAIN_CSV,
            help="path to output training CSV file",
        )
        ap.add_argument(
            "-v",
            "--val",
            default=config.VAL_CSV,
            help="path to output validate CSV file",
        )
        ap.add_argument(
            "-e", "--test", default=config.TEST_CSV, help="path to output test CSV file"
        )
        ap.add_argument(
            "-c",
            "--classes",
            default=config.CLASSES_CSV,
            help="path to output classes CSV file",
        )
        ap.add_argument(
            "-s", "--split", type=float, default=0.0, help="train and val split"
        )
        ap.add_argument(
            "-b", "--splittest", type=float, default=1.0, help="val and test split"
        )
        args = vars(ap.parse_args())

        # Create easy variable names for all the arguments
        # What ever the user will define here,the directory structure should be according to that
        # Otherwise leave these as these have been defined

        self.annot_path = args["annotations"]
        self.images_path = args["images"]
        self.train_csv = args["train"]
        self.val_csv = args["val"]
        self.test_csv = args["test"]
        self.classes_csv = args["classes"]
        self.train_test_split = args["split"]
        self.test_val_split = args["splittest"]
        # initialize the set of classes we have
        self.CLASSES = set()

    """
    This Function manages the test train split by shuffeling the paths and applying split afterwords.
    
    Inputs:
        1- Image paths
        2- Train Val split ratio (already given through arguments. Please change there.)
        3- Test val split ratio (already given through arguments. Please change there.)
    
    Outputs:
        Managed dataset
    
    """

    def test_train_val_splits(
        self,
        paths_to_images: str,
        test_train_split_ratio: float,
        test_val_split_ratio: str,
    ) -> list:
        # grab all image paths then construct the training, validating and testing splits
        self.image_paths = list(paths.list_files(paths_to_images))
        random.shuffle(self.image_paths)

        self.division_i = int(len(self.image_paths) * test_train_split_ratio)
        self.trainimage_paths = self.image_paths[: self.division_i]
        self.test_val_split_paths = self.image_paths[self.division_i :]

        self.division_j = int(len(self.test_val_split_paths) * test_val_split_ratio)
        self.testimage_paths = self.test_val_split_paths[: self.division_j]
        self.valimage_paths = self.test_val_split_paths[self.division_j :]

        # create the list of datasets to build
        self.dataset = [
            ("train", self.trainimage_paths, self.train_csv),
            ("test", self.testimage_paths, self.test_csv),
            ("val", self.valimage_paths, self.val_csv),
        ]
        return self.dataset

    """
    This function actually writes the train, test and val annotation CSV files
    
    Input:
         Managned data fron the test_train_val_split() function
    Outputs:
        None
    """

    def get_annotations(self, data_to_read: list) -> None:
        # loop over the datasets
        for (d_type, image_paths, output_csv) in data_to_read:
            # load the contents
            print("[INFO] creating '{}' set...".format(d_type))
            print("[INFO] {} total images in '{}' set".format(len(image_paths), d_type))

            # open the output CSV file
            csv = open(output_csv, "w")

            # loop over the image paths
            for image_path in image_paths:
                # build the corresponding annotation path
                fname = image_path.split(os.path.sep)[-1]
                fname = "{}.xml".format(fname[: fname.rfind(".")])
                annotPath = os.path.sep.join([self.annot_path, fname])

                # load the contents of the annotation file and buid the soup
                contents = open(annotPath).read()
                soup = BeautifulSoup(contents, "html.parser")

                # extract the image dimensions
                w = int(soup.find("width").string)
                h = int(soup.find("height").string)
                # loop over all object elements
                for o in soup.find_all("object"):
                    # extract the label and bounding box coordinates
                    label = o.find("name").string
                    xmin = int(float(o.find("xmin").string))
                    ymin = int(float(o.find("ymin").string))
                    xmax = int(float(o.find("xmax").string))
                    ymax = int(float(o.find("ymax").string))

                    # truncate any bounding box coordinates that fall outside
                    # the boundaries of the image
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmax)
                    ymax = min(h, ymax)

                    # ignore the bounding boxes where the minimum values are larger
                    # than the maximum values and vice-versa due to annotation errors
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    elif xmax <= xmin or ymax <= ymin:
                        continue

                    # write the image path, bb coordinates, label to the output CSV
                    row = [
                        os.path.abspath(image_path),
                        str(xmin),
                        str(ymin),
                        str(xmax),
                        str(ymax),
                        str(label),
                    ]
                    csv.write("{}\n".format(",".join(row)))

                    # update the set of unique class labels
                    self.CLASSES.add(label)

            # close the CSV file
            csv.close()

        # write the classes to file
        print("[INFO] writing classes...")
        csv = open(self.classes_csv, "w")
        rows = [",".join([c, str(i)]) for (i, c) in enumerate(self.CLASSES)]
        csv.write("\n".join(rows))
        csv.close()

        return None


if __name__ == "__main__":
    dpxml = DataPreparationXml()
    test_train_val_data = dpxml.test_train_val_splits(
        dpxml.images_path, dpxml.train_test_split, dpxml.test_val_split
    )
    annotations = dpxml.get_annotations(test_train_val_data)
