
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import csv
import glob
import os
import subprocess

files = []
gaze_angle_x = []
gaze_angle_y = []
confidence_threshold = 0.98

def read_csv_file(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:
            if counter != 0:

                row_element = 0
                for i in row:
                

                    # Check Confidence. If less than 0.98, we will skip that value
                    if row_element == 3:
            
                        if float(i) < confidence_threshold:
                            break

                    # Gaze Angle X is in column  11
                    if row_element == 11:
                        gaze_angle_x.append(i)
                        
                    # Gaze Angle Y is in column 12
                    if row_element == 12:
                        gaze_angle_y.append(i)
                    
                    row_element = row_element + 1

            counter = counter + 1


def plot_kernels():

    data = {
        "x": gaze_angle_x,
        "y": gaze_angle_y
    }

    g = sns.JointGrid(x="x", y="y", data=data, space=0)
    g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
    g = g.plot_marginals(sns.kdeplot, shade=True)
    plt.show()



for file in glob.glob("../../data/Media/*.mov"):
    video = os.path.splitext(os.path.basename(file))
    print(video[0])
    files.append(video[0])

for name in files:
    file_name = name
    os.chdir("../../OpenFacePrecompiledBinaries/bin")
    subprocess.call(['./FeatureExtraction', '-f', '../../data/Media/' + file_name + '.mov', '-gaze', '-out_dir', '../../data/gaze_outputs'])
    read_csv_file('../../data/gaze_outputs/' + file_name + '.csv')
    plot_kernels()


# sns.set(color_codes=True)
# sns.distplot(gaze_angle_x, axlabel="Gaze Angle Radians", label="x",  kde_kws={"color": "b", "lw": 1, "label": "x-KDE"})
# sns.distplot(gaze_angle_y, label="y",kde_kws={"color": "orange", "lw": 1, "label": "y-KDE"} )
# plt.show()


print("hello world")