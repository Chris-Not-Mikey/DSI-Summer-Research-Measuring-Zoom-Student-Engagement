
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import seaborn as sns
import csv
import glob
import os
import subprocess
import copy

# This script will run Gaze Detection from OpenFace's provided Feature Extraction code
# From this, it will generate multiavariate kernel density estimate

files = []
gaze_angle_x = []
gaze_angle_y = []
confidence_threshold = 0.98

indices = {
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


eye_features_2D = {
    "timestamp": 0,
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

eye_features_2D_list = []
ear_independent_time = []
ear_left_list = []

def read_csv_file(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:

            eye_features_2D_element = copy.deepcopy(eye_features_2D)

            if counter != 0:

                row_element = 0
                for i in row:

                    # Check Confidence. If less than 0.98, we will skip that value
                    if row_element == indices["confidence"]:
                        #print(row_element)
                        if float(i) < confidence_threshold:
                            break

                    # Gaze Angle X is in column  11
                    if row_element == indices["gaze_angle_x"]:
                        #print(row_element)
                        gaze_angle_x.append(i)
                        
                    # Gaze Angle Y is in column 12
                    if row_element == indices["gaze_angle_y"]:
                        #print(row_element)
                        gaze_angle_y.append(i)


                    check_eye_features(row_element, i, eye_features_2D_element)
                   

                    
                    row_element = row_element + 1

            else:
                print("we are here!")
                column_index = 0
                for j in row:
                    check_indeces(j, column_index)
                    column_index = column_index + 1


            # This looks strange, but the way the data structure works,
            # The first element in the counter is all 0's
            if counter >= 1:
                eye_features_2D_list.append(eye_features_2D_element)

            counter = counter + 1


def check_eye_features(row_element, data, eye_features_2D_element):

    eye_features = ["timestamp", "x_36", "x_37", "x_38", "x_39", "x_40", "x_41", "y_36", "y_37", "y_38", "y_39", "y_40", "y_41", 
    "x_42", "x_43", "x_44", "x_45", "x_46", "x_47", "y_42", "y_43", "y_44", "y_45", "y_46", "y_47"]
    

    for i in eye_features:
        if row_element == indices[i]:
            eye_features_2D_element[i] = data
            break
    

def check_indeces(name, column_index):
    index_names = ["timestamp", "confidence", "gaze_angle_x", "gaze_angle_y", "x_36", "x_37", "x_38", "x_39", "x_40", "x_41",
     "y_36", "y_37", "y_38", "y_39", "y_40", "y_41", "x_42", "x_43", "x_44", "x_45", "x_46",
      "x_47", "y_42", "y_43", "y_44", "y_45", "y_46", "y_47"]

    for i in index_names:
        if name.strip() == i:
            indices[i] = column_index
            return True


def measure_eye_aspect_ratio():
    for i in eye_features_2D_list:

        x_21_1 = (float(i["x_37"]) - float(i["x_41"])) ** 2
        y_21_1 = (float(i["y_37"]) - float(i["y_41"])) ** 2
        distance_left_1 = np.sqrt((x_21_1 + y_21_1))

        x_21_2 = (float(i["x_38"]) - float(i["x_40"])) ** 2
        y_21_2 = (float(i["y_38"]) - float(i["y_40"])) ** 2
        distance_left_2 = np.sqrt((x_21_2 + y_21_2))

 
        x_21_3 = (float(i["x_36"]) - float(i["x_39"])) ** 2
        y_21_3 = (float(i["y_36"]) - float(i["y_39"])) ** 2
        distance_left_3 = np.sqrt((x_21_3 + y_21_3))

        ear_left = (distance_left_1 + distance_left_2) / (2.0 * (distance_left_3))
        ear_left_list.append(ear_left)
        ear_independent_time.append(float(i["timestamp"]))


def plot_EAR_vs_time(name):
    path = "../../data/kernel_plots/" + name + "_EAR"
    plt.scatter(ear_independent_time, ear_left_list)
    plt.savefig(path)

def calculate_number_blinks():

    number_blinks = 0
    ear_threshold = 0.2

    # Get local minimums from the EAR data recorded
    # Function returns the indices of the local mins
    min_index = argrelextrema(np.array(ear_left_list), np.less, order=20)
    ear_left_array = np.array(ear_left_list)

    # Use the indices to max a list of minumns
    minimums = []
    for i in min_index:
        minimums.append(ear_left_array[i])


    # see if the minimums make the cutoff
    for j in minimums:
        for k in j:
            if k < ear_threshold:
                number_blinks = number_blinks + 1

    return number_blinks


def plot_kernels(name):

    path = "../../data/kernel_plots/" + name 
    data = {
        "x": gaze_angle_x,
        "y": gaze_angle_y
    }

    g = sns.JointGrid(x="x", y="y", data=data, space=0)
    g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
    g = g.plot_marginals(sns.kdeplot, shade=True)
    plt.savefig(path)


for file in glob.glob("../../data/Media/*.mov"):
    video = os.path.splitext(os.path.basename(file))
    print(video[0])
    files.append(video[0])

# TODO: Remove. This is for speeding up computation while debuggin
files = ["blink_test", "blink_test_2"]

for name in files:
    file_name = name
    os.chdir("../../OpenFacePrecompiledBinaries/bin")
    subprocess.call(['./FeatureExtraction', '-f', '../../data/Media/' + file_name + '.mov', '-gaze', '-2Dfp', '-out_dir', '../../data/gaze_outputs'])
    read_csv_file('../../data/gaze_outputs/' + file_name + '.csv')
    #plot_kernels(name)
    measure_eye_aspect_ratio()
    plot_EAR_vs_time(name)
    blinks = calculate_number_blinks()
    print("There were " + str(blinks) + " blinks recorded in " + name)
    eye_features_2D_list.clear()
    ear_independent_time.clear()
    ear_left_list.clear()  



print("hello world")