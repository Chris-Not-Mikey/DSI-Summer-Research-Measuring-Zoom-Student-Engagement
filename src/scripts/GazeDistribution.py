
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import csv
import glob
import os
import subprocess

# This script will run Gaze Detection from OpenFace's provided Feature Extraction code
# From this, it will generate multiavariate kernel density estimate

files = []
gaze_angle_x = []
gaze_angle_y = []
confidence_threshold = 0.98

indices = {
    "confidence": 0,
    "gaze_angle_x": 0,
    "gaze_angle_y": 0,
    "x_37": 0,
    "x_38": 0,
    "x_40": 0,
    "x_41": 0,
    "y_37": 0,
    "y_38": 0,
    "y_40": 0,
    "y_41": 0,
    "x_43": 0,
    "x_44": 0,
    "x_46": 0,
    "x_47": 0,
    "y_43": 0,
    "y_44": 0,
    "y_46": 0,
    "y_47": 0,
}

def read_csv_file(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:
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
                    
                    row_element = row_element + 1

            else:
                print("we are here!")
                column_index = 0
                for j in row:
                    check_indeces(j, column_index)
                    column_index = column_index + 1

            counter = counter + 1


def check_indeces(name, column_index):

    if name.strip() == "confidence":
        print("dow we get here????")
        indices["confidence"] = column_index
        return True

    if name.strip() == "gaze_angle_x":
        indices["gaze_angle_x"] = column_index
        return True

    if name.strip() == "gaze_angle_y":
        indices["gaze_angle_y"] = column_index
        return True

    if name.strip() == "x_37":
        indices["x_37"] = column_index
        return True

    if name.strip() == "x_38":
        indices["x_38"] = column_index
        return True

    if name.strip() == "x_40":
        indices["x_40"] = column_index
        return True

    if name.strip() == "x_41":
        indices["x_41"] = column_index
        return True

    if name.strip() == "y_37":
        indices["y_37"] = column_index
        return True

    if name.strip() == "y_38":
        indices["y_38"] = column_index
        return True

    if name.strip() == "y_40":
        indices["y_40"] = column_index
        return True

    if name.strip() == "y_41":
        indices["y_41"] = column_index
        return True

    if name.strip() == "x_43":
        indices["x_43"] = column_index
        return True

    if name.strip() == "x_44":
        indices["x_44"] = column_index
        return True

    if name.strip() == "x_46":
        indices["x_46"] = column_index
        return True

    if name.strip() == "x_47":
        indices["x_47"] = column_index
        return True

    if name.strip() == "y_43":
        indices["y_43"] = column_index
        return True

    if name.strip() == "y_44":
        indices["y_44"] = column_index
        return True

    if name.strip() == "y_46":
        indices["y_46"] = column_index
        return True

    if name.strip() == "y_47":
        indices["y_47"] = column_index
        return True

  



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
#files = ["up_down_test"]

for name in files:
    file_name = name
    os.chdir("../../OpenFacePrecompiledBinaries/bin")
    subprocess.call(['./FeatureExtraction', '-f', '../../data/Media/' + file_name + '.mov', '-gaze', '-2Dfp', '-out_dir', '../../data/gaze_outputs'])
    read_csv_file('../../data/gaze_outputs/' + file_name + '.csv')
    plot_kernels(name)


print("hello world")