# This file has been written to read the tsv format files
# and convert them into a csv file so that retinaNet could be trained.

# importing the required liberaries
import os
import csv

class DataPreparation:
    def __init__(self, relative_path: str = '/Drones/') -> None:
        self.current_path = os.getcwd()
        self.base_path = self.current_path + relative_path
    
    # Inputs:
            # Name of folder in which files are present
            # Extensions for the files such as bboxes.tsv  for bbox file and jpg for image files
    
    # Read the files in the directory
    # Check for the name of class label and bbox files are similar or not
    # because if the names different class and bbox files will be encoded for one file
    # Then data generation for object detector would be wrong
    def get_filenames(self, directory:str, extension:str)->list:
        
        self.files_paths = []
        for filename in os.listdir(self.base_path + directory):
            self.files_details = self.base_path + directory + filename 
            if (self.files_details.split(".")[1:] == extension.split(".")):
                self.files_paths.append(self.files_details)
            
        return self.files_paths
    
    # Inputs:
            # Folder name, where the training files are present
            # Extension for training image files e.g. 'jpg'/ 'png' etc
            # Extension for bbox files which is 'bboxes.tsv' because its has names separted by '.' as '1.bboxes.tsv'
            
    # Returns:
            # List of training image paths
            # List of training images bbox paths
            # list of training images bbox class label file paths
    def get_positive_data(self,
                          directory: str = "positive/",
                          image_extension: str = "jpg",
                          box_extension: str = "bboxes.tsv",
                          classes_extension: str = "bboxes.labels.tsv") -> tuple:
        image_filenames = self.get_filenames(directory, image_extension)
        box_filenames = self.get_filenames(directory, box_extension)
        classes_filenames = self.get_filenames(directory, classes_extension)
        return image_filenames, box_filenames, classes_filenames
    
    # Inputs:
        # Folder name, where the validation image files are present
        # Extension for validation image files e.g. 'jpg'/ 'png' etc
        # Extension for validation image bbox files which is 'bboxes.tsv' because its has names separted by '.' as '1.bboxes.tsv'
        
    # Returns:
            # List of validation image paths
            # List of validation image bbox paths
            # list of validation image bbox class label file paths
    def get_val_data(self,
                          directory: str = "valImages/",
                          image_extension: str = "jpg",
                          box_extension: str = "bboxes.tsv",
                          classes_extension: str = "bboxes.labels.tsv") -> tuple:
        image_filenames = self.get_filenames(directory, image_extension)
        box_filenames = self.get_filenames(directory, box_extension)
        classes_filenames = self.get_filenames(directory, classes_extension)
        return image_filenames, box_filenames, classes_filenames
    
    # Inputs:
            # Folder name, where the files are present
            # Extension for image files e.g. 'jpg'/ 'png' etc
            # Extension for bbox files which is 'bboxes.tsv' because its has names separted by '.' as '1.bboxes.tsv'
            
    # Returns:
            # List of image paths
            # List of bbox paths
            # list of bbox class label file paths
            
    def get_test_data(self,
                          directory: str = "testImages/",
                          image_extension: str = "jpg",
                          box_extension: str = "bboxes.tsv",
                          classes_extension: str = "bboxes.labels.tsv") -> tuple:
        image_filenames = self.get_filenames(directory, image_extension)
        box_filenames = self.get_filenames(directory, box_extension)
        classes_filenames = self.get_filenames(directory, classes_extension)
        return image_filenames, box_filenames, classes_filenames
    
    # Inputs:
            # Folder name, where the files are present
            # Extension for image files e.g. 'jpg'/ 'png' etc
            # Extension for bbox files which is 'bboxes.tsv' because its has names separted by '.' as '1.bboxes.tsv'
            
    # Returns:
            # List of image paths
            # List of bbox paths
            # list of bbox class label file paths
            
    def get_negative_data(self,
                          directory: str = "negative/",
                          image_extension: str = "jpg") -> tuple:
        image_filenames = self.get_filenames(directory, image_extension)
        return image_filenames
    
    # Inputs:
            # Image file names
            # Bounding boxes file names
    # Outputs:
            # annotated dats
    def get_annotations(self, image_filenames: list, box_filenames: list) -> list:
        self.annotations_data = []
        for index, box_filename in enumerate(box_filenames):
            with open(box_filename, "r") as box_file:
                box_data = box_file.read()
                rows = box_data.split("\n")
                for i in range(0, len(rows) - 1):
                    self.annotations_data.append(
                        [image_filenames[index]] + [int(x) for x in rows[i].split("\t")] + ['drone']
                    )
        return self.annotations_data
    
    # Inputs:
            # Image file names (Becuase negative images has no bbox and class labels therefore, there would be no
            # bbox file name provided)
    # Outputs:
            # Negative annotated data
    
    def get_negative_annotations(self, image_filenames: list) -> list:
        self.anntation_data = []
        for i in range(0,len(image_filenames)):
            self.anntation_data.append([image_filenames[i], "", "", "", "", ""])
        return self.anntation_data
    
    # Inputs:
            # Name of CSV file
            # Data which is supposed to be written into csv file
    # Outputs:
            # CSV file for given data
    def write_csv_file(self, filename: str, file_to_write: list) -> None:
        with open(self.current_path + "/" + filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            wr.writerows(file_to_write)
        
        
if __name__ == "__main__":
    dp = DataPreparation()
    positive_image_filenames, positive_box_filenames, positive_class_filenames = dp.get_positive_data()
    val_image_filenames, val_box_filenames, val_class_filenames = dp.get_val_data()
    test_image_filenames, test_box_filenames, test_class_filenames = dp.get_test_data()
    negative_image_filenames = dp.get_negative_data()
    
    positive_annotated_data = dp.get_annotations(positive_image_filenames, positive_box_filenames)
    val_annotated_data = dp.get_annotations(val_image_filenames, val_box_filenames)
    test_annotated_data = dp.get_annotations(test_image_filenames, test_box_filenames)
    negative_annotated_data = dp.get_negative_annotations(negative_image_filenames)
    
    train_annotated_data = positive_annotated_data + negative_annotated_data
    
    dp.write_csv_file("train.csv",train_annotated_data)
    dp.write_csv_file("val.csv",val_annotated_data)
    dp.write_csv_file("test.csv", test_annotated_data)
    
    classes_train = [["drone" , 0], ["dummy", 1]]
    dp.write_csv_file("classes.csv", classes_train)
    
    