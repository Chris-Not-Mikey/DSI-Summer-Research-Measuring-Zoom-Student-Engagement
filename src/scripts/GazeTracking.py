import os
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import math
import numpy as np
import csv
import statistics
import copy

# This class runs the gaze tracking algo and the event detection algo
# The gaze tracking will parse importan ocular data form the videos
# The event detection will be used to classify saccades and gazes. 
# NOTE: For best results, Event Detection will need to be trained

class GazeTracker:
    def __init__(self, gaze_angle_x, gaze_angle_y, gaze_features_2D_list):
        self.confidence_threshold = 0.98
        self.gaze_angle_x = gaze_angle_x
        self.gaze_angle_y = gaze_angle_y
        self.gaze_features_2D_list = gaze_features_2D_list
        self.dtype = np.dtype([('t', np.float64),	#time in seconds
	                ('x', np.float32),	#horizontal gaze direction in degrees
	                ('y', np.float32), 	#vertical gaze direction in degrees
	                ('status', np.bool),	#status flag. False means trackloss 
	                ('evt', np.uint8)	#event label:
                                        #0: Undefined
                                        #1: Fixation
                                        #2: Saccade
                                        #3: Post-saccadic oscillation
                                        #4: Smooth pursuit
                                        #5: Blink
        ])
        self.irf_format = np.empty(0, dtype=self.dtype)
        self.index_adjusted_time = []
        self.saccade_full_velocity = []
        self.saccade_x_angle = []
        self.saccade_y_angle = []

        self.saccade_velocity = []
        self.saccade_frequency = []
        self.saccade_length = []
        self.gaze_length = []


    def change_to_open_face_dir(self):
        os.chdir("../../OpenFacePrecompiledBinaries/bin")

    def change_to_fixation_saccade_dir(self):
        os.chdir("../../FixationSaccadeScripts/")

    # run the event detection script
    def event_detection(self,name):
        subprocess.call(['python2', 'run_irf.py', 'irf_2020-06-30_14-20-40', 'etdata', 'openFaceGeneratedData', '--save_csv'])
        #rename the output
        src = 'etdata/openFaceGeneratedData_irf/i2mc/' + name + '_i2mc_raw.mat'
        dst = 'etdata/openFaceGeneratedData_irf/i2mc/' + name + '_i2mc.mat'

        os.chdir("./util_lib/I2MC-Dev/")
        subprocess.call(['matlab', '-nodesktop', '-nosplash', '-r', 'I2MC_rz; exit;'])
        os.chdir("../../")

        # run again if this is the first run with a given media file
        # this looks strange, but this is the expected behavior for the irf script.
        # read the documenation here for more information: https://github.com/r-zemblys/irf
        subprocess.call(['python2', 'run_irf.py', 'irf_2020-06-30_14-20-40', 'etdata', 'openFaceGeneratedData', '--save_csv'])

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
    

    # to use the irf script (in FixationSaccade scripts)
    def convert_to_irf_data_structure(self, name):
        
        counter = 0
        for i in self.gaze_features_2D_list:

            x_radians = float(i["gaze_angle_x"])
            y_radians = float(i["gaze_angle_y"])
            time = i["timestamp"]

            x_degrees = x_radians * (180.0/math.pi)
            y_degrees = y_radians * (180.0/math.pi)

            conversion = np.array([(np.float64(time), np.float32(x_degrees),np.float32(y_degrees),np.bool(False), np.uint8(0))]
            , dtype=self.dtype)

            self.irf_format = np.append(self.irf_format, conversion)

    
        np.delete(self.irf_format, 0)
        print(self.irf_format)

        path = "../../FixationSaccadeScripts/etdata/openFaceGeneratedData/" + name
        np.save(path, self.irf_format)


    def move_output_csv(self, name):
        src = 'etdata/openFaceGeneratedData_irf/' + name + ".csv"
        dst = '../data/event_detection/' + name + ".csv"
        os.rename(src, dst)


    # Get important data from event detection output like
    # Saccade Velocity, Saccade Length, Saccade Frequency, and gaze length
    def parse_event_data(self, name):
        path = '../../data/gaze_outputs/' + name + "_velocity.csv"

        saccade_path = '../../data/event_detection/' + name + ".csv"

        
        # In theory thee features list and the velocities
        # list are the same. However, the IRF classifier
        # drops data with too much noise.
        # So we will keep track of the gaze index seperatly
        # And proactively account of index offsets as we loop
        counter = 0 

        #saccade velocity rolling mean variables
        saccade_velocity_rolling_mean = 0
        saccade_velocity_rolling_list = []

        # 1 = gaze
        # 2 = saccade
        # 3 = post saccade oscillation
        prev_saccade_value = 1
        current_saccade_length = 0
        saccade_frequency_rolling_mean = 0
        saccade_length_rolling_mean = 0
        saccade_length_rolling_list = []
        gaze_length_rolling_mean = 0
        gaze_length_rolling_list = []
        current_gaze_length = 0

        with open(saccade_path, newline='') as s:
            s_reader = csv.reader(s)


            with open(path, newline='') as f:
                reader = csv.reader(f)

                # make csv file object a list
                # so that we can index it
                vel_list = list(reader)
                max_len = len(vel_list)
    
                row_counter = 0
                frame_counter = 0


                for row in s_reader:
                    
                    skip_row = False
                    first_row = False
                    
                    if row_counter == 0:
                        first_row = True
                        skip_row = True
                        counter = counter + 1

                    # If data is malformed, we will skip this next part
                    if row[4] == "False":
                        skip_row = True


                    # set rolling mean for saccade velocity
                    if skip_row == False:
                        saccade_velocity_rolling_list.append(float(vel_list[counter][1]))
                        saccade_velocity_rolling_mean = statistics.mean(saccade_velocity_rolling_list)


                    # set rolling mean for Gaze Length and Saccade Frequency
                    if skip_row == False:
        

                        current_saccade_val = int(row[5])
                     

                        if current_saccade_val == 1:
                            current_gaze_length = current_gaze_length + 1

                        if (current_saccade_val == 2 or current_saccade_val == 3) and prev_saccade_value == 1:
                            gaze_length_rolling_list.append(current_gaze_length)
                            gaze_length_rolling_mean = statistics.mean(gaze_length_rolling_list)
                            current_gaze_length = 0

                        if current_saccade_val == 2 or current_saccade_val == 3:
                            current_saccade_length = current_saccade_length + 1

                        if (prev_saccade_value == 2 or prev_saccade_value ==3) and current_saccade_val == 1:
                            saccade_frequency_rolling_mean = saccade_frequency_rolling_mean + 1

                            saccade_length_rolling_list.append(current_saccade_length)
                            saccade_length_rolling_mean = statistics.mean(saccade_length_rolling_list)
                            current_saccade_length = 0

                        #update prev
                        prev_saccade_value = current_saccade_val


                    if skip_row == False:

                        # Make sure to avoid overflow
                        if counter < max_len:
                            self.index_adjusted_time.append(float(row[1]))

                            self.saccade_full_velocity.append(float(vel_list[counter][1]))

                            x_radians = float(self.gaze_features_2D_list[row_counter]["gaze_angle_x"])
                            y_radians = float(self.gaze_features_2D_list[row_counter]["gaze_angle_y"])

                            x_degrees = x_radians * (180/math.pi)
                            y_degrees = y_radians * (180/math.pi)

                            self.saccade_x_angle.append(x_degrees)
                            self.saccade_y_angle.append(y_degrees)

                            counter = counter + 1


                    # 900 refers to 900 frames, equivalent to 30 seconds.
                    if frame_counter == 900:

                        # add saccade velocity
                        self.saccade_velocity.append(saccade_velocity_rolling_mean)
                        self.saccade_frequency.append(saccade_frequency_rolling_mean)

                        # Divide by 30 fps to convert to seconds
                        self.saccade_length.append(saccade_length_rolling_mean/30)
                        self.gaze_length.append(gaze_length_rolling_mean/30)

                        # clear for next rolling mean
                        saccade_velocity_rolling_mean = 0
                        saccade_velocity_rolling_list = []
                        saccade_frequency_rolling_mean = 0
                        saccade_length_rolling_mean = 0
                        saccade_length_rolling_list = []
                        gaze_length_rolling_mean = 0
                        gaze_length_rolling_list = []


                        # reset frame counter
                        frame_counter = 0


                    row_counter = row_counter + 1
                    frame_counter = frame_counter + 1
                 

        path = "../../data/kernel_plots/" + name + "_saccade_velocities"

        plt.scatter(self.index_adjusted_time, self.saccade_full_velocity, c='b')
        plt.scatter(self.index_adjusted_time, self.saccade_x_angle, c='r')
        plt.scatter(self.index_adjusted_time, self.saccade_y_angle, c='g')

        plt.savefig(path)
        plt.close()



    # Final Output for Engagement prediction (saccade velocity)
    def get_saccade_velocity(self):
        return self.saccade_velocity

    # Final Output for Engagement prediction (saccade frequency)
    def get_saccade_frequency(self):
        return self.saccade_frequency

    # Final Output for Engagement prediction (saccade length)
    def get_saccade_length(self):
        return self.saccade_length

    # Final Output for Engagement prediction (gaze length)
    def get_gaze_length(self):
        return self.gaze_length

