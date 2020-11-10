# -*- coding: utf-8 -*-
"""
Created on Fri April  3 13:12:06 2020

@author: Omer
"""

"""
    User Guide
        
    1- Create a directory named as "data_augmentation" in and put this dataaugmentation.py file in it.
    2- Collect the images for your dataset and carete a subdirector inside parent such as "images_to_augment". Then,
        i- Either put all the images in the directory "images_to_augment" or
        ii- Put those images in "images_to_augment" which you need to augment
    3- Please do not change the imported liberaries even some are not used
    4- After rotation and affine transformation, image sizes would increase than your required size, for instance, if the input image is (640 x 640 x 3); then, after rotation its size will increase with respect to your angle of rotation. 
    **** please place all augmented images into a new directory "images_to_resize" and comment all the the menthods in augment_images() methods in main() except resize images. Then, change the path of images folders from "images_to_aument" to "images_to_resize". Run the files again to get the final augmented images of your required size such as (640 x 640 x 3).
    
    5- All images are not passed through the operation such as change in brightness, contrast and transformation. The image path's have been split by using sciket-learn. So, the ratio of images for operation scan be set in 'augment_images()' in main().
    
    "Sample Director Structure"
    
    -data_augmentation
        -- images_to_augment
        --images_resize
        --rotated_images
        --resized_images
        --changed_brightness
        --changed_contrast
        --transformesd_images
        --augmented_images

"""
    
"""
        Import the required libraries
"""

import numpy as np
import matplotlib
import os
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageFilter, ImageStat, ImageEnhance 
import imutils
import ntpath
import sys
import matplotlib.pyplot as plt
import glob
import math
#from blend_modes import blend_modes
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
#import grabcut
global counter
counter = 0

class DataAugmentation:
    def __init__(self):
        self.current_path = os.getcwd()


    '''
            Importing the paths of the image files inside the directory
    '''
    
    def load_paths(self, directory: str) -> list:
        self.paths = []
        for files in sorted(os.listdir(directory)):
            if (files != ".DS_Store"):
                self.paths.append(directory+'/'+files)
        return self.paths
    
    """
        To make the system general for every typr of dataset, directories for every operation will be created atonomously. The images after with specific operation will be placed in specific folders.
        
        1- resized_images -> contains the images after resizing
        2- rotated_images -> contains images after rotations
        3- changed_brightness -> contains the images after brightness manipulation
        4- changed_contrast -> contains the images after contrast stretching
        5- transformed_images -> contains the images after apllication of affine transfprmation for shearing of images
        6- augmented_images -> contains the actual outputs of augmneted images after complete augmentation process. These are our actual output images whcih will be our dataset and have to be labeled.
        """
    def make_directories(self):
        if (not os.path.exists("resized_images")):
            os.mkdir("resized_images")
        
        if (not os.path.exists("rotated_images")):
            os.mkdir("rotated_images")
            
        if (not os.path.exists("changed_brightness")):
            os.mkdir("changed_brightness")
        
        if (not os.path.exists("changed_contrast")):
            os.mkdir("changed_contrast")
        
        if (not os.path.exists("augmented_dataset")):
            os.mkdir("augmented_dataset")
        
        if (not os.path.exists("Transformed_Images")):
            os.makedirs("Transformed_Images")

    """
    ------------------------------------------------------------------------------
        Resize the size of the images
        width = maximum width of the image (Which you want to have in output image)
        height = Maximum height of the image (Which you want to have in output images)
        
        Usually we want to have square images where Width = Height
        output_dim = (width, height) (Output dimentions of the image)
    ------------------------------------------------------------------------------
    """
    def resize_image(self,width: int, height: int) -> int:
        
        global counter
        
        self.paths = self.load_paths("images_to_augment")
        
        
        # Spacified the size of the output image
        self.output_dim = (width, height)
        
        # Going to loop over all the images in the folder to resize them
        for img_path in self.paths:
            #print(img_path)
            # Reading image as numpy array
            self.img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            
            # resize image
            self.resized_img = cv2.resize(self.img, self.output_dim, interpolation = cv2.INTER_AREA)
            
            # Writing image into output directory
            cv2.imwrite("resized_images/"+str(counter)+".jpg",self.resized_img)
            counter += 1
        counter = 2505
        return self.paths, counter


    ###############################################################################
    #if __name__ == "__main__":
        #image_paths = load_paths("Drones")
        #resize_image(640,640)
    ###############################################################################

    """
    To Rotate the images either on
    90,180,270 or 360/0 degrees
    360/0 degrees will give us the original positioned images
    Angles have to spacified by the user
    """

    def rotate_images(self) -> int:
        global counter
        
        self.paths = self.load_paths("resized_images")
        
        for image_path in self.paths:
            
            self.image = cv2.imread(image_path)
            self.head, self.tail = ntpath.split(image_path)
            
            self.title,self.extension = self.tail.split('.')
            self.angles_to_rotate = [-18,18]
            #for angle in np.arange(135, 360, 110):
            for angle in self.angles_to_rotate:
                self.rotated = imutils.rotate_bound(self.image, angle)
                cv2.imwrite("rotated_images/"+str(counter)+".jpg",self.rotated)
                counter += 1
                #cv2.imwrite("Transformed_Images/"+title+"/"+str(t[19])+".png",dst19)
        counter = 2505
        return counter


    """
    ------------------------------------------------------------------------------
    Gamma correction usually changing the apearance if colours in the image are in a same way
    as the human eye will percieve the colours.
    
    Reference
        https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    
    Gamma values < 1 will make the image brighter just by brightening the dark pixels in the images
    Gamma values > 1 will darken the images to make bright pixels more visible na dark pixels to go
    darker
        
        Inputes:
            Image (array) file
            Basic start value of Gamma (1.0)
        Outputs:
            Gamma stretched images with different rannge of gamma
    """
    def adjust_gamma(self,image, gamma= 1.0):
        global counter
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        self.invGamma = 1.0 / gamma
        #print('processed_1')
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        #print('processed_2')
    
        # apply gamma correction using the lookup table
        return cv2.LUT(image, self.table)

    """
        Brightness is an attribute of visual perception in which a source appears to be radiating 
        or reflecting light. In other words, brightness is the perception elicited 
        by the luminance of a visual target. It is not necessarily proportional to luminance.
        
        Inputs: Paths to images
        Outputs: image files with 3 different brightness values
        '** however these values can be defined by the user by changing the gamma values'
    """
    def change_brightness(self, path: str) -> int:
        global counter
              
        self.image = cv2.imread(path)
        self.head, self.tail = ntpath.split(path)
            
        self.title,self.extension = self.tail.split('.')
    
        for gamma in np.arange(0.75, 1.0, 0.5):
            # ignore when gamma is 1 (there will be no change to the image)
            if gamma == 1:
                continue
        
            # apply gamma correction and show the images
            gamma = gamma if gamma > 0 else 0.1
            self.adjusted = self.adjust_gamma(self.image, gamma=gamma)
            cv2.imwrite("changed_brightness/"+self.title+".jpg",self.adjusted)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.adjusted)
            
            counter += 1
        
        return counter


    def change_contrast(self, img, level):
        global counter
        self.factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            self.value = 128 + self.factor * (c - 128)
            return max(0, min(255, self.value))
        return img.point(contrast)
    """
        Contrast is the difference in luminance or colour that makes an object 
        (or its representation in an image or display) distinguishable. 
        In visual perception of the real world, contrast is determined by the difference 
        in the color and brightness of the object and other objects within the same field of view.
        
        Inputs:
            path of image
            
        Outputs:
            image with changed contrast
            counter value = 1
            
        '** however the contanst values can be defined by user by changing the values for step variable'
    """
    
    def change_contrast_multi(self, path: str) -> int:
        global counter
           
        self.steps = []
        for step in range(-50, 0, 100):
            self.steps.append(step)
        
        self.img = Image.open(path)
        self.head, self.tail = ntpath.split(path)
        
        self.title, self.extension = self.tail.split('.')
        self.width, self.height = self.img.size
        self.contrast_image = Image.new('RGB', (self.width * len(self.steps), self.height))
        for n, level in enumerate(self.steps):
            self.img_filtered = self.change_contrast(self.img, level)
            self.contrast_image.paste(self.img_filtered, (self.width * n, 0))
            self.img_filtered.save("changed_contrast/"+self.title+".jpg")
            self.img_filtered.save("augmented_dataset/"+self.title+".jpg")
            counter += 1
                
        #counter = 1
        return counter
    
    """
        Affine transformation is used for sheating the images at certain level
        Therefore, I am applying affine transformations to shear the images which will change the shape of objects
        in th images and their position at certain level
        Inputs:
            path of image
            
        Outputs:
            image with shearing
            counter value = 1
    """
    
    def img_transform(self, paths: str) -> int:
        global counter
        
        self.img = cv2.imread(paths, cv2.IMREAD_UNCHANGED)
        self.rows,self.cols,self.ch = self.img.shape
        self.head, self.tail = ntpath.split(paths)
    
        self.title, self.extension = self.tail.split('.')
    
        self.t = []
        for i in range(0,100):
            self.t.append(i)
        self.matrix_to_apply = random.randint(1,9)
                
        #EAST FACING
        if (self.matrix_to_apply == 1):
            
            self.pts1 = np.float32([[self.cols/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols/10,self.rows/2]])
            self.pts2 = np.float32([[self.cols/5,self.rows/5],[self.cols/2,self.rows/8],[self.cols/5,self.rows/1.8]])
            self.M = cv2.getAffineTransform(self.pts1,self.pts2)
            self.dst1 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst1)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst1)
            counter += 1
        
        #NORTH-WEST FACING
        elif (self.matrix_to_apply == 2):

            self.pts3 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts4 = np.float32([[self.cols*4.5/5,self.rows/5],[self.cols/2,self.rows/8],[self.cols*4.5/5,self.rows/1.8]])
            self.M = cv2.getAffineTransform(self.pts3,self.pts4)
            self.dst2 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst2)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst2)
            counter += 1
            
        #RIGHT TILTED FORWARD FACING
        if (self.matrix_to_apply == 3):
            self.pts7 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts8 = np.float32([[self.cols*10/12,self.rows/6],[self.cols/2.2,self.rows/8],[self.cols*8.4/10,self.rows/1.8]])
            self.M = cv2.getAffineTransform(self.pts7,self.pts8)
            self.dst3 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst3)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst3)
            counter += 1
                
        #FORWARD FACING W/ DISTORTION 2
        if (self.matrix_to_apply == 4):
            
            self.pts15 = np.float32([[self.cols/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts16 = np.float32([[self.cols/11,self.rows/10],[self.cols/2.1,self.rows/10],[self.cols*8.5/10,self.rows/1.95]])
            self.M = cv2.getAffineTransform(self.pts15,self.pts16)
            self.dst4 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst4)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst4)
            counter += 1
        
        #SHRINK 1
        if (self.matrix_to_apply == 5):
            
            self.pts25 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts26 = np.float32([[self.cols*8/10,self.rows/10],[self.cols*1.34/3,self.rows/10.5],[self.cols*8.24/10,self.rows/2.5]])
            self.M = cv2.getAffineTransform(self.pts25,self.pts26)
            self.dst5 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst5)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst5)
            counter += 1
        
        #SHRINK 2
        if (self.matrix_to_apply == 6):
            
            self.pts27 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts28 = np.float32([[self.cols*8.5/10,self.rows*3.1/10],[self.cols/2,self.rows*3/10],[self.cols*8.44/10,self.rows*1.55/2.5]])
            self.M = cv2.getAffineTransform(self.pts27,self.pts28)
            self.dst6 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst6)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst6)
            counter += 1
        
        #FORWARD FACING W/ DISTORTION 8
        if (self.matrix_to_apply == 7):
            
            self.pts31 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts32 = np.float32([[self.cols*8.75/10,self.rows/9.1],[self.cols/1.95,self.rows/8],[self.cols*8.5/10,self.rows/2.05]])
            self.M = cv2.getAffineTransform(self.pts31,self.pts32)
            self.dst7 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst7)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst7)
            counter += 1
        
        
        #FORWARD FACING W/ DISTORTION 9
        if (self.matrix_to_apply == 8):
            
            self.pts33 = np.float32([[self.cols*9/10,self.rows/10],[self.cols/2,self.rows/10],[self.cols*9/10,self.rows/2]])
            self.pts34 = np.float32([[self.cols*8.75/10,self.rows/9.1],[self.cols/1.95,self.rows/9],[self.cols*8.5/10,self.rows/2.2]])
            self.M = cv2.getAffineTransform(self.pts33,self.pts34)
            self.dst8 = cv2.warpAffine(self.img,self.M,(self.cols,self.rows))
            cv2.imwrite("Transformed_Images/"+self.title+".jpg",self.dst8)
            cv2.imwrite("augmented_dataset/"+self.title+".jpg",self.dst8)
            counter += 1
        
        counter = 1
        return counter

##############################################################################
if __name__ == "__main__":
#    image_paths = load_paths("Drones")
#    resize_image(640,640)
##############################################################################

    def augment_images():
        data_aug = DataAugmentation()
        #global counter
        counter = 1
        data_aug.make_directories()
        
        data_aug.resize_image(640, 640)
        
        data_aug.rotate_images()
        
        directory = "rotated_images"
        paths = data_aug.load_paths(directory)
        random.shuffle(paths)
        
        """
        # This configuration of fuction call is used when the user wants the every image shopuld be changed in contrast, brightness and then sheared
                
        directory = "changed_brightness"
        paths = data_aug.load_paths(directory)
        data_aug.change_brightness(paths)

        directory = "changed_contrast"
        paths = data_aug.load_paths(directory)
        data_aug.change_contrast_multi(paths)

        """

        paths = np.array(paths)
        paths_to_change_brightness ,paths_to_nochange = train_test_split(paths,test_size=0.66)
        paths_to_transform, paths_to_change_contrast = train_test_split(paths_to_nochange, test_size = 0.5)
        
        for image_path in paths_to_change_brightness:
            data_aug.change_brightness(image_path)
        for image_path in paths_to_change_contrast:
            data_aug.change_contrast_multi(image_path)
        for image_path in paths_to_transform:
            data_aug.img_transform(image_path)
            
        counter = 1
        return counter
    
    augment_images()

