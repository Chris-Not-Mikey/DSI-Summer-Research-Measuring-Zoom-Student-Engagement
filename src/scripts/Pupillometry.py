import numpy as np
import matplotlib.pyplot as plt


class Pupillometer:

    def __init__(self, pupil_features_2D_list):
        self.pupil_features_2D_list = pupil_features_2D_list
        self.pupil_independent_time = []
        self.pupil_radius_left_list = []

    def calc_left_pupil_radius(self):
        for i in self.pupil_features_2D_list:

            x_21_1 = (float(i["eye_lmk_x_25"]) - float(i["eye_lmk_x_21"])) ** 2
            y_21_1 = (float(i["eye_lmk_y_25"]) - float(i["eye_lmk_y_21"])) ** 2
            radius_1 = (np.sqrt((x_21_1 + y_21_1)))/2.0

            x_21_1 = (float(i["eye_lmk_x_26"]) - float(i["eye_lmk_x_22"])) ** 2
            y_21_1 = (float(i["eye_lmk_y_26"]) - float(i["eye_lmk_y_22"])) ** 2
            radius_2 = (np.sqrt((x_21_1 + y_21_1)))/2.0

            x_21_1 = (float(i["eye_lmk_x_27"]) - float(i["eye_lmk_x_23"])) ** 2
            y_21_1 = (float(i["eye_lmk_y_27"]) - float(i["eye_lmk_y_23"])) ** 2
            radius_3 = (np.sqrt((x_21_1 + y_21_1)))/2.0

            x_21_1 = (float(i["eye_lmk_x_20"]) - float(i["eye_lmk_x_24"])) ** 2
            y_21_1 = (float(i["eye_lmk_y_20"]) - float(i["eye_lmk_y_24"])) ** 2
            radius_4 = (np.sqrt((x_21_1 + y_21_1)))/2.0

            
            radius = np.average([radius_1, radius_2, radius_3, radius_4])

            self.pupil_radius_left_list.append(radius)
            self.pupil_independent_time.append(float(i["timestamp"]))


    def plot_radius_vs_time(self, name):
        path = "../../data/kernel_plots/" + name + "_pupil_radius"
        plt.scatter(self.pupil_independent_time, self.pupil_radius_left_list)
        plt.savefig(path)
