import numpy as np
import sys
import pickle
import csv

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# This script is a class for predicting student engagement 
# via a CNN model. This class allows for training and pre-trained prediction.

class EngagementPredictor:
    def __init__(self, name ):

        self.name = name
        self.gaze_length = []
        self.saccade_frequency = []
        self.saccade_length = []
        self.saccade_velocity = []
        self.blink_rate = []
        self.blink_duration = []
        self.pupil_size = []
        self.pupil_size_simple = []
        self.master_list = []
        self.complete_list = []
        self.engagement = []
        self.results = []
        self.raw_results = []

    
    # Rads the CSV file that is created with all the ocular measurements
    # Each measurment is taken at a 30 seconds timestamp
    def read_csv_file(self):
        path = '../../data/engagement_features/' + self.name + '_engagement.csv'
        with open(path) as s:
            reader = csv.reader(s)

            counter = 0
            for row in reader:
                if counter != 0:
                    self.gaze_length.append(np.float32(row[1]))
                    self.saccade_frequency.append(np.float32(row[2]))
                    self.saccade_length.append(np.float32(row[3]))
                    self.saccade_velocity.append(np.float32(row[4]))
                    self.blink_rate.append(np.float32(row[5]))
                    self.blink_duration.append(np.float32(row[6]))
                    self.pupil_size.append(np.float32(row[7]))
                    self.pupil_size_simple.append(np.float32(row[8]))

                    engagement_element = np.int64(row[9])
                    self.engagement.append(engagement_element)
       
                    element = []

                    element_1 = float(row[1])
                    element_2 = float(row[2])
                    element_3 = float(row[3])
                    element_4 = float(row[4])
                    element_5 = float(row[5])
                    element_6 = float(row[6])
                    element_7 = float(row[7])


                    element.append(element_1)
                    element.append(element_2)
                    element.append(element_3)
                    element.append(element_4)
                    element.append(element_5)
                    element.append(element_6)
                    element.append(element_7)
                    
                    element_np = np.array(element, dtype=np.float32)
                    self.master_list.append(element_np)
                  


                counter = counter + 1


    # Main CNN algo to predict student engagement
    def predict_engagement(self, train):

        cnn_filename = "../../data/cnn_models/"
        np.random.seed(7)

        X = np.array(self.master_list, dtype=np.float32)
        y = np.array(self.engagement, dtype=np.int64)

        start = 0
        if train == True:
            start = len(X)/2

        else:
            start = 0

        X_train = X[0:start]
        y_train = y[0:start]
    
        X_test = X[start:]
        y_test = y[start:]

        # If train, we train and save a new model. This will overwrite an exsisting model
        # NOTE: If you are going to train you might want to adjust (hardcode) the start variable
        # The default is probably fine, but you can always adjust start
        if train == True:
            self.train_model(cnn_filename, X_train, y_train, X_test, y_test, start)

       
        # Predictions made with pre-trained model
        else:
            # load model
            model = tf.keras.models.load_model(cnn_filename)
            print("Printing Predicions")
            self.raw_results = model.predict(X_test)
            self.results = model.predict_classes(X_test)
            self.determine_results()
            self.write_results_to_csv(start)
            print(self.results)


    # CNN training
    def train_model(self, cnn_filename, X_train, y_train, X_test, y_test, start):

        # Create, train, compile and save model
        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_dim=7))
        opt = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.fit(X_train, y_train, epochs=20, batch_size=1)
    
        # save model
        model.save(cnn_filename)

        self.results = model.predict_classes(X_test)
        self.write_results_to_csv(start)


    # Determine if final output is Engaged or not
    def determine_results(self):

        engaged = 0
        for i in self.results:
            if i[0] == 1:
                engaged = engaged + 1

        num_results = len(self.results)
        result = engaged/num_results

        if result > 0.5:
            print("Student Was engaged")
        else:
            print("Student was not engaged")


    # Write engagement predictions to a csv file
    def write_results_to_csv(self, start):
        path = '../../data/engagement_features/' + self.name + '_engagement_RESULTS.csv'
        with open(path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

            counter = start
            csv_writer.writerow(["Row", "GL", "SF", "SL", "SV", "BF", "BD", "PS", "PSS", "ENG"])
            for i in self.results:
                engagement_features = []

                engagement_features.append(self.gaze_length[counter])
                engagement_features.append(self.saccade_frequency[counter])
                engagement_features.append(self.saccade_length[counter])
                engagement_features.append(self.saccade_velocity[counter])
                engagement_features.append(self.blink_rate[counter])
                engagement_features.append(self.blink_duration[counter])
                engagement_features.append(self.pupil_size[counter])
                engagement_features.append(self.pupil_size_simple[counter])

                # 0 = not engaged
                # 1 = engaged
                csv_writer.writerow([counter, engagement_features[0], engagement_features[1],
                engagement_features[2], engagement_features[3], engagement_features[4], engagement_features[5], engagement_features[6], engagement_features[7], i[0]])

                counter = counter + 1
        csvfile.close()


    # Plot the results of student engagement against time
    # This is just for visulation
    def plot_results(self, name):

        Y = []
        X = []
        for i in self.results:
            Y.append(i[0])

        for j in self.raw_results:
            X.append(j[0])

        X = np.array(X)
        Y = np.array(Y)


        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        xs = np.arange(len(X))+0
        masks = Y == 0
        ax.scatter(xs[masks], X[masks], c='r', label='Not Engaged')
        masks = Y == 1
        ax.scatter(xs[masks], X[masks], c='b', label='Engaged')
        ax.plot(xs, X, c='k')
        
        ax.set_xlabel('time (seconds)')
        ax.set_ylabel('engagement metric')
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)

        filename = "../../data/kernel_plots/" + name + "_engagement.png"
        fig.savefig(filename)
        fig.clf()