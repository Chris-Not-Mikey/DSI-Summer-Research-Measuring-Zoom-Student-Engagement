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
from EngagementPredictor import EngagementPredictor


# This script is a class for predicting student engagement 
# via a CNN model. This class allows for training and pre-trained prediction.

class BinaryPredictor:
    def __init__(self, names, engagement, predictor_list ):

        self.name = names
        self.engagement_known = engagement
        self.predict_list = predictor_list
        self.master_list = []
        self.complete_list = []
        self.engagement = []
        self.results = []
        self.raw_results = []



    # Main CNN algo to predict student engagement
    def predict_engagement(self, predictor_list):

        cnn_filename = "../../data/cnn_better_model/"
        np.random.seed(7)


        core_list = []
        for i in predictor_list:
            core_list.append(i.get_master_list())


        if len(core_list) != len(self.engagement_known):
            print("WARNING! The lengths are NOT the same. Major issue")

        X = np.array(core_list, dtype=np.float32)
        y = np.array(self.engagement_known, dtype=np.int64)

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