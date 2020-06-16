import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import csv
import math
import cv2
import imutils
from imutils import face_utils
import dlib
import time


class Pupillometer:

    def __init__(self, pupil_features_2D_list):
        self.pupil_features_2D_list = pupil_features_2D_list
        self.pupil_independent_time = []
        self.pupil_radius_left_list = []
        self.time_stamp = []
        self.pupil_diameter = []


    # Use OpenFace face landmark features to determine radius
    # This will be less accurate than the results provided by
    # the pupil locater
    def calc_simple_pupil_radius(self):
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


    # Plot Based on OpenFaces's method
    def plot_simple_radius_vs_time(self, name):
        path = "../../data/kernel_plots/" + name + "_pupil_radius"
        plt.scatter(self.pupil_independent_time, self.pupil_radius_left_list)
        plt.savefig(path)
        plt.close()


    # Plot based on Pupil Locater's method
    def plot_advanced_diameter_vs_time(self, name):
        path = "../data/kernel_plots/" + name + "_pupil_diameter"
        plt.scatter(self.time_stamp, self.pupil_diameter)
        plt.savefig(path)
        plt.close()


    def pupil_locater(self):
        # cropped_output.mov is the name of the file that crop_video_to_roi generates
        subprocess.call(['python3', 'inferno.py', 'cropped_output.mov'])


    def change_to_pupil_locater_dir(self):
        os.chdir("../../PupilLocatorScripts")


    # Generate video footage of just the eye region.
    # Takes the whole video and ouputs a greyscale video of just the left eye
    # This can be changed to right eye
    def crop_video_to_roi(self, name):

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')

        cap = cv2.VideoCapture('../data/Media/' + name + ".mov")

        #output settings
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('cropped_output.mov', fourcc, 30, (250,100))



        pos_frame =  cap.get(cv2.CAP_PROP_POS_FRAMES)
        while(True):
            ret, frame = cap.read()

            write_roi = frame

            if ret==True:

                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  #update frames read

                #resize
                frame = imutils.resize(frame, width=500)

                # convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #detect faces
                rects = detector(gray, 1)

                #for each face detected
                for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the landmark (x, y)-coordinates to a NumPy array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # loop over the face parts individually
                    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

                        if (name == "left_eye"):
                            # clone the original image so we can draw on it, then
                            # display the name of the face part on the image
                            clone = frame.copy()
                            # loop over the subset of facial landmarks, drawing the
                            # specific face part
                            for (x, y) in shape[i:j]:
                                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                            # extract the ROI of the face region as a separate image
                            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                            roi = frame[y:y + h, x:x + w]
                            roi = imutils.resize(roi, width=250, height=100)


                            # TODO Worth invesitagating the tradeoff
                            # roi = imutils.resize(roi, width=250, height=100, inter=cv2.INTER_CUBIC)
                            # with cv2.resize. cv2 is more liberal, but imutils preservers aspect ratio

                            dim = (250, 100) 
                            roi = cv2.resize(roi, dim, interpolation=cv2.INTER_CUBIC)
                          
                            out.write(roi)
                            write_roi = roi
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):                 
                                break


                #out.write(write_roi)
        
            else:

                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)


            if  (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break


        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    # read csv file Pupil Locater script generates
    def read_pupil_csv_file(self, filename):
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            for row in reader:

                counter = 0
                for i in row:
                    
                    # frame
                    if counter == 0:
                        self.time_stamp.append(int(i))

                    # Eye width
                    if counter == 3:
                        self.pupil_diameter.append(float(i))

                    counter = counter + 1





