# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:59:09 2020

@author: engro
"""

"""
This programe file will help me to examin my dataset quickly such as
    
    1- How does my dataset look like?
        As, head of dataset will display some rows of dataset to let me know
    2- How many unique images are there is my dataset?
        During the labeling I have converted every object present in the image
        Dataset distribution int train, test and val dataset has been done with                         build_dataset.py.
        thetefore, to see the length of each file I need this program
    3- How many unique classes I have in my dataset?
        As, I have mannualy labelled my dataset but there could be mistake in labelling
        therefore, I am checking again to confirm that is everything have been labelled correctly. Furthermore, this programe file will tell me that how many instances for each class have been labelled.
        for instance,
        drone = 125
        person = 580
    4- Are the extracted labels for training of neural network correct?
        This programe will load randomly some images and corresponding labels
        then, it will draw the annotations on loaded images and save it to my directory.
        so that I could see them
"""
import numpy as np
import pandas as pd
import cv2
import os
import csv
import glob
import matplotlib.pyplot as plt

class VarifyLabels:
    
    # Took the path of base directory and file in the directory
    # ** I have to give the file name for test tarin and val dataset annotations mannualy
    # Return None
    def __init__(self, file_path: str = '/val.csv') -> None:
        self.base_path = os.getcwd()
        self.file_name = self.base_path + file_path
    
    def make_directories(self):
        if (not os.path.exists("verified_images")):
            os.mkdir("verified_images")
            
    # Inputs:
        # Take the dataframe of labelled data "CSV" file
    # Outputs:
        # Head of dataset
        # Length of particular file
        # Unique classes in this particular file of labelled dataset
        # Labelled instances for each class
    def examin_dataset(self, data_frame:list)->None:
        
        print("Head of training dataset labelled image file")
        print(data_frame.head())
        
        print("unique image files")
        # Number of unique training images
        
        print(data_frame['filename'].nunique())
        
        print("unique classes")
        # Number of classes
        print(data_frame['classname'].value_counts())
    
    # Inputs:
        # Name of file
        # Uses the examin_dataset function inside
    # outputs:
        # Dataframe for provided CSV file
    def read_files(self, name_of_file:str)->None:
        self.df = pd.read_csv(name_of_file, index_col=None, header=None)
        self.df.columns = ['filename','xmin','ymin','xmax','ymax','classname']
        
        self.examin_dataset(self.df)
        return self.df
    
    def plot_box_statistics(self,bbox_area_details:tuple, bbox_location_details:tuple)->None:
        plt.style.use('ggplot')
        n = 5
        
        bad = bbox_area_details
        bld = bbox_location_details
        
        
        fig, (ax1, ax2) = plt.subplots(2)
        index = np.arange(n)
        bar_width = 0.5
        opacity = 0.9
        # plt.ylim(0, 5000)
        
        ax1.bar(index, bad, bar_width, alpha=opacity, color='green',
                        label='BBox Chractristics w.r.t Area')
        
        for i, val in enumerate(bad):
            ax1.text(i, val, int(val), horizontalalignment='center', 
                     verticalalignment='bottom', fontdict={'fontweight':1000, 'size':12})
                
        ax1.set_xlabel('Dataset Details-based on Object Size', family='Times New Roman')
        ax1.set_ylabel('Number of Objects', family='Times New Roman')
        ax1.set_title('Analysis of Dataset Based on Object Sizes and Locations Inside Images', family='Times New Roman')
        ax1.set_xticks(index + bar_width / 2)
        
        ax1.set_xticklabels(('Tiny Objects','Small Objects',
                            'Medium Objects','Large Objects', 'XLarge Objects'), family='Times New Roman')
        # ax1.legend(loc="upper right")
        
        
        ax2.bar(index, bld, bar_width, alpha=opacity, color='blue',
                        label='BBox Chractristics w.r.t Location')
        for i, val in enumerate(bld):
            ax2.text(i, val, int(val), horizontalalignment='center', 
                     verticalalignment='bottom', fontdict={'fontweight':1000, 'size':12})
        
        ax2.set_xlabel('Dataset Details-based on Object Location', family='Times New Roman')
        ax2.set_ylabel('Number of Objects', family='Times New Roman')
        ax2.set_xticks(index + bar_width / 2)
        
        ax2.set_xticklabels(('Objects in 1st Quadrent','Objects in 2nd Quadrent',
                            'Objects in 3rd Quadrent','Objects in 4th Quadrent', 'Objects on Image Center'), family='Times New Roman')
        # ax2.legend(loc="upper right")
        
        plt.show()
        
        return None
        
        
        # Inputs:
        # Dataframe of either train, test or val file whichever I want to examin
    # Outputs:
        # Images with drawn boxes seved in my directory
    def varify_annotations(self, dataset_array:list)->list:
        
        tinybox_counter = 0 # (32 x 32) sized objects
        smallbox_counter = 0 # (64 x 64) sized objects
        mediumbox_counter = 0 # (128 x 128) sized objects
        largebox_counter = 0 # (256 x 256) sized objects
        xlbox_counter = 0 # (512 x 512) sized objects
        # sample_bbox_area =[1024, 4096, 16384, 65536, 262144] # (32x32)(64x64)(128x128)(256x256)(512x512) sized box areas
        
        first_quadr_box = 0
        second_quadr_box = 0
        third_quadr_box = 0
        fourth_quadr_box = 0
        centered_box = 0
        
        # First column
        self.filenames = np.array(dataset_array['filename'])
        self.total_instances = len(self.filenames)
        
        self.bbox_data = np.array(dataset_array[['xmin', 'ymin', 'xmax', 'ymax']])
        self.classes_data = np.array(dataset_array['classname'])
        
        self.dataset = np.array(dataset_array[['filename', 'xmin', 'ymin', 'xmax', 'ymax','classname']])
        self.converted_dataset = []
        self.counter = 1
        for n,line in enumerate(self.dataset):
            #print(data)
            file_name = str(line[0])
            x_min = int (line[1])
            y_min = int(line[2])
            x_max = int(line[3])
            y_max = int(line[4])
            object_class = str(line[5])
            width = int((x_max - x_min))
            height = int((y_max - y_min))
            
            # Compute the area of bbox
            bbox_area = width * height
            
            # Count the number of boxes for each size category based on their area size
            if(bbox_area <= 1024):
                # its a tiny object
                tinybox_counter += 1
            elif(bbox_area > 1024 and bbox_area <= 4096):
                # its a small object
                smallbox_counter += 1
            elif(bbox_area > 4096 and bbox_area <= 16384):
                # Its a medium sized object
                mediumbox_counter += 1
            elif(bbox_area > 16384 and bbox_area <= 65536):
                # its a large sized object
                largebox_counter += 1
            else:
                # its an xlsized object
                xlbox_counter += 1
            
            # print('area of bbox = ',bbox_area)
            
            # Compute the center point of bbox
            center_x = (x_min + x_max)/2
            center_y = (y_min + y_max)/2
            
            # print('center of box is ', (center_x, center_y))
            
            if(center_x > 320 and center_y < 320):
                # Object in first quadrent region of image
                first_quadr_box += 1
            elif(center_x < 320 and center_y < 320):
                # Object in second quadrent region of image
                second_quadr_box += 1
            elif(center_x < 320 and center_y > 320):
                # Object in third quadrent region of image
                third_quadr_box += 1
            elif(center_x > 320 and center_y > 320):
                # Object in Fourth quadrent region of image
                fourth_quadr_box += 1
            else:
                # Object in Fourth quadrent region of image
                centered_box += 1
                
            #row = [file_name, x_min, y_min, width, height, object_class]
            #converted_dataset.append(row)
            
            # img = cv2.imread(file_name)
            # print(img.shape)
            # cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,255,0),2)
            # cv2.putText(img,object_class,(x_min,y_min),0,1,(0,0,255))
            # # cv2.imshow('image',img)
            # cv2.imwrite("verified_images/verified_image_" + str(self.counter) + ".jpg", img)
            # self.counter += 1
            # # cv2.waitKey(0)
            
            # if (n == 50):
            #     break
            
        cv2.destroyAllWindows()
        bbox_area_info = (tinybox_counter, smallbox_counter, mediumbox_counter, largebox_counter, xlbox_counter)
        bbox_location_info = (first_quadr_box, second_quadr_box, third_quadr_box, fourth_quadr_box, centered_box)
        
        self.plot_box_statistics(bbox_area_info, bbox_location_info)
        print('bbox area details are ', ('no. of tiny objects =', tinybox_counter, 'no. of small objects =', smallbox_counter, 'no. of medium objects =', mediumbox_counter, 'no. of large objects =', largebox_counter, 'no. of xl objects =', xlbox_counter))
        print('inside the image location details of objects ', ('in first quard = ',first_quadr_box, 'in second quard = ',second_quadr_box,'in third quard = ', third_quadr_box, 'in fourth quard = ',fourth_quadr_box, 'in center of image = ',centered_box))
        return self.dataset

if __name__ == "__main__":
    
    # Instance of class has been created
    vl = VarifyLabels()
    
    # Generated the targeted directories if not present
    vl.make_directories()
    
    # generated an examined the dataframe
    required_dataframe = vl.read_files(vl.file_name)
    
    # Varified the images
    output_dataset = vl.varify_annotations(required_dataframe)
    
    
    
    