# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, sys, random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

def load_paths(directory):
    paths = []
    for files in os.listdir(directory):
        if (files != ".DS_Store"):
            paths.append(directory+'/'+files)
    return paths

# def explore_dataset():
# Reading the CSV file of training information
directory = os.getcwd()
file_path = directory + '/dataset/'
print("Head of training dataset labelled image file")
train = pd.read_csv(file_path + 'val.csv')
train.columns = ['filename','xmin','ymin','xmax','ymax','classname']
train.head()

print("unique files")
# Number of unique training images

print(train['filename'].nunique())

print("unique classes")
# Number of classes
print(train['classname'].value_counts())


