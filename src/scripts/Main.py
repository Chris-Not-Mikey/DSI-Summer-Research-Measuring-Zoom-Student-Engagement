import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import csv
import glob
import os
import subprocess
import copy
from BlinkDetection import BlinkDetector
from GazeTracking import GazeTracker

# This script will run Gaze Detection from OpenFace's provided Feature Extraction code
# From this, it will generate multiavariate kernel density estimate

# Pipeline Variables

files = []
data_template = {
    "timestamp": 0,
    "confidence": 0,
    "gaze_angle_x": 0,
    "gaze_angle_y": 0,
    "x_36": 0,
    "x_37": 0,
    "x_38": 0,
    "x_39": 0,
    "x_40": 0,
    "x_41": 0,
    "y_36": 0,
    "y_37": 0,
    "y_38": 0,
    "y_39": 0,
    "y_40": 0,
    "y_41": 0,
    "x_42": 0,
    "x_43": 0,
    "x_44": 0,
    "x_45": 0,
    "x_46": 0,
    "x_47": 0,
    "y_43": 0,
    "y_43": 0,
    "y_44": 0,
    "y_45": 0,
    "y_46": 0,
    "y_47": 0,
}

# Gaze Tracking Variables
gaze_angle_x = []
gaze_angle_y = []
confidence_threshold = 0.98

# Blink detection variables
eye_features_2D_list = []
ear_independent_time = []
ear_left_list = []

# Read the CSV File generated by the OpenFace precomplied binaries
def read_csv_file(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        counter = 0

        indices = copy.deepcopy(data_template)

        for row in reader:

            eye_features_2D_element = copy.deepcopy(data_template)

            if counter != 0:

                row_element = 0
                for i in row:

                    # Fill in the relevant data from reading the CSV file
                    check_gaze_features(row_element, i, indices)
                    check_eye_features(row_element, i, eye_features_2D_element, indices)            
                    row_element = row_element + 1

            # If counter = 0, match the name of the .csv element with its respective index
            else:
                column_index = 0
                for j in row:
                    check_indices(j, column_index, indices)
                    column_index = column_index + 1


            # This looks strange, but the way the data structure works,
            # The first element in the counter is all 0's
            if counter >= 1:
                eye_features_2D_list.append(eye_features_2D_element)

            counter = counter + 1


def check_indices(name, column_index, indices):
    index_names = ["timestamp", "confidence", "gaze_angle_x", "gaze_angle_y", "x_36", "x_37", "x_38", "x_39", "x_40", "x_41",
     "y_36", "y_37", "y_38", "y_39", "y_40", "y_41", "x_42", "x_43", "x_44", "x_45", "x_46",
      "x_47", "y_42", "y_43", "y_44", "y_45", "y_46", "y_47"]

    for i in index_names:
        if name.strip() == i:
            indices[i] = column_index
            return True


def check_eye_features(row_element, data, eye_features_2D_element, indices):

    eye_features = ["timestamp", "x_36", "x_37", "x_38", "x_39", "x_40", "x_41", "y_36", "y_37", "y_38", "y_39", "y_40", "y_41", 
    "x_42", "x_43", "x_44", "x_45", "x_46", "x_47", "y_42", "y_43", "y_44", "y_45", "y_46", "y_47"]
    

    for i in eye_features:
        if row_element == indices[i]:
            eye_features_2D_element[i] = data
            break
    

def check_gaze_features(row_element, data, indices):
     # Check Confidence. If less than 0.98, we will skip that value
    if row_element == indices["confidence"]:
        
        if float(data) < confidence_threshold:
            return False

    # Gaze Angle X is in column  11
    if row_element == indices["gaze_angle_x"]:
        gaze_angle_x.append(data)
        return True
        
    # Gaze Angle Y is in column 12
    if row_element == indices["gaze_angle_y"]:
        gaze_angle_y.append(data)
        return True
    
    return True

# Loads all the media files to use 
def load_media_to_parse():
    for file in glob.glob("../../data/Media/*.mov"):
        video = os.path.splitext(os.path.basename(file))
        print(video[0])
        files.append(video[0])

# TODO: Remove. This is for speeding up computation while debuggin
files = ["blink_test", "blink_test_2"]

# The heart of the algorithm
if __name__ == "__main__":

    files = ["blink_test", "blink_test_2"]

    # For each file (video of a person's/people's face(s)) we do eye tracking, blink detection, and pupilometry
    for name in files:

        # Use the OpenFace precompiled Binaries to start gaze tracking
        gaze_tracker = GazeTracker(gaze_angle_x,gaze_angle_y)
        gaze_tracker.track_gaze(name)

        # Read the CSV file
        read_csv_file('../../data/gaze_outputs/' + name + '.csv')

        # Plot outputs of gaze
        gaze_tracker.plot_kernels(name)

        # Now detect blinks in the footage
        detector = BlinkDetector(eye_features_2D_list, ear_independent_time, ear_left_list)
        detector.calculate_left_EAR()
        # detector.plot_EAR_vs_time(file_name)
        detector.calculate_number_blinks()

        print("There were " + str(detector.get_blinks()) + " blinks recorded in  " + name)

        # Clear data. If this is not done, there will be some leftover data that WILL affect computation
        eye_features_2D_list.clear()
        ear_independent_time.clear()
        ear_left_list.clear()  


    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")