import os
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns

class GazeTracker:
    def __init__(self, gaze_angle_x, gaze_angle_y):
        self.confidence_threshold = 0.98
        self.gaze_angle_x = gaze_angle_x
        self.gaze_angle_y = gaze_angle_y


    def change_to_open_face_dir(self):
        os.chdir("../../OpenFacePrecompiledBinaries/bin")

    # Run the program to track gaze angle
    def track_gaze(self, name):
        subprocess.call(['./FeatureExtraction', '-f', '../../data/Media/' + name + '.mov', '-gaze', '-2Dfp', '-out_dir', '../../data/gaze_outputs'])

    # Plot Kernels of gaze angles
    def plot_kernels(self, name):

        path = "../../data/kernel_plots/" + name 
        data = {
            "x": self.gaze_angle_x,
            "y": self.gaze_angle_y
        }

        g = sns.JointGrid(x="x", y="y", data=data, space=0)
        g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
        g = g.plot_marginals(sns.kdeplot, shade=True)
        plt.savefig(path)
        plt.close()
