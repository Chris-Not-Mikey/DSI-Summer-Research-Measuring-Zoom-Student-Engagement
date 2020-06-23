import os
import matplotlib.pyplot as plt
import subprocess
import seaborn as sns
import math
import numpy as np
import csv

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
        self.saccade_velocity = []
        self.saccade_x_angle = []
        self.saccade_y_angle = []


    def change_to_open_face_dir(self):
        os.chdir("../../OpenFacePrecompiledBinaries/bin")

    def change_to_fixation_saccade_dir(self):
        os.chdir("../../FixationSaccadeScripts/")

    # run the event detection script
    def event_detection(self,name):
        subprocess.call(['python2', 'run_irf.py', 'irf_2018-03-26_20-46-41', 'etdata', 'openFaceGeneratedData', '--save_csv'])
        #rename the output
        src = 'etdata/openFaceGeneratedData_irf/i2mc/' + name + '_i2mc_raw.mat'
        dst = 'etdata/openFaceGeneratedData_irf/i2mc/' + name + '_i2mc.mat'

        if os.path.exists(src):
            # concert raw to normal
            os.rename(src, dst)
            # run again if this is the first run with a given media file
            # this looks strange, but this is the expected behavior for the irf script.
            # read the documenation here for more information: https://github.com/r-zemblys/irf
            subprocess.call(['python2', 'run_irf.py', 'irf_2018-03-26_20-46-41', 'etdata', 'openFaceGeneratedData', '--save_csv'])

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
        
        for i in self.gaze_features_2D_list:
            x_radians = float(i["gaze_angle_x"])
            y_radians = float(i["gaze_angle_y"])
            time = i["timestamp"]

            x_degrees = x_radians * (180/math.pi)
            y_degrees = y_radians * (180/math.pi)

            conversion = np.array([(np.float64(time), np.float32(x_degrees),np.float32(y_degrees),np.bool(False), np.uint8(0))]
            , dtype=self.dtype)

          
            self.irf_format = np.append(self.irf_format, conversion)

        np.delete(self.irf_format, 0)
        #print(self.irf_format)

        path = "../../FixationSaccadeScripts/etdata/openFaceGeneratedData/" + name
        np.save(path, self.irf_format)


    def move_output_csv(self, name):
        src = 'etdata/openFaceGeneratedData_irf/' + name + ".csv"
        dst = '../data/event_detection/' + name + ".csv"
        os.rename(src, dst)


    def plot_saccade_profile(self, name):
        path = '../../data/gaze_outputs/' + name + "_velocity.csv"

        saccade_path = '../../data/event_detection/' + name + ".csv"

        
        # In theory thee features list and the velocities
        # list are the same. However, the IRF classifier
        # drops data with too much noise.
        # So we will keep track of the gaze index seperatly
        # And proactively account of index offsets as we loop
        counter = 0 

        with open(saccade_path, newline='') as s:
            s_reader = csv.reader(s)

            with open(path, newline='') as f:
                reader = csv.reader(f)

                # make csv file object a list
                # so that we can index it
                data_list = list(reader)
                max_len = len(data_list)
    
                row_counter = 0
                for row in s_reader:
                    
                    skip_row = False
                    first_row = False
                    
                    if row_counter == 0:
                        first_row = True
                        skip_row = True
                        counter = counter + 1

            
                    if row[4] == "False":
                        
                        skip_row = True
                        print("we are here")
                        

                    if skip_row == False:

                        # Make sure to avoid overflow
                        if counter < max_len:
                            self.index_adjusted_time.append(float(row[1]))

                            self.saccade_velocity.append(float(data_list[counter][1]))

                            x_radians = float(self.gaze_features_2D_list[row_counter]["gaze_angle_x"])
                            y_radians = float(self.gaze_features_2D_list[row_counter]["gaze_angle_y"])

                            x_degrees = x_radians * (180/math.pi)
                            y_degrees = y_radians * (180/math.pi)

                            self.saccade_x_angle.append(x_degrees)
                            self.saccade_y_angle.append(y_degrees)

                            counter = counter + 1

                    row_counter = row_counter + 1

        path = "../../data/kernel_plots/" + name + "_saccade_velocities"

        plt.scatter(self.index_adjusted_time, self.saccade_velocity, c='b')
        plt.scatter(self.index_adjusted_time, self.saccade_x_angle, c='r')
        plt.scatter(self.index_adjusted_time, self.saccade_y_angle, c='g')

        plt.savefig(path)
        plt.close()

