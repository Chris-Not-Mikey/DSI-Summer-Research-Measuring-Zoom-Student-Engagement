import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

class BlinkDetector:
    def __init__(self, eye_features_2D_list, ear_independent_time, ear_left_list ):
        self.ear_threshold = 0.2
        self.number_blinks = 0
        self.eye_features_2D_list = eye_features_2D_list
        self.ear_independent_time = ear_independent_time
        self.ear_left_list = ear_left_list


    def calculate_left_EAR(self):
        for i in self.eye_features_2D_list:

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
            self.ear_left_list.append(ear_left)
            self.ear_independent_time.append(float(i["timestamp"]))

    def plot_EAR_vs_time(self, name):
        path = "../../data/kernel_plots/" + name + "_EAR"
        plt.scatter(self.ear_independent_time, self.ear_left_list)
        plt.savefig(path)


    def calculate_number_blinks(self):

        # Get local minimums from the EAR data recorded
        # Function returns the indices of the local mins
        min_index = argrelextrema(np.array(self.ear_left_list), np.less, order=20)
        ear_left_array = np.array(self.ear_left_list)

        # Use the indices to max a list of minumns
        minimums = []
        for i in min_index:
            minimums.append(ear_left_array[i])


        # see if the minimums make the cutoff
        for j in minimums:
            for k in j:
                if k < self.ear_threshold:
                    self.number_blinks = self.number_blinks + 1

    def get_blinks(self):
        return self.number_blinks

        