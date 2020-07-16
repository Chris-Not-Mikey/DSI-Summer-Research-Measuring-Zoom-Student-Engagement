import numpy as np
import sys
import pickle
import csv

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


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


    def predict_engagement(self, train):

        cnn_filename = "../../data/cnn_models/"
        np.random.seed(7)
   
        X = np.array(self.master_list, dtype=np.float32)
        y = np.array(self.engagement, dtype=np.int64)

        X_train = X[0:3]
        y_train = y[0:3]
    
        X_test = X[:]
        y_test = y[:]

   
        # truncate and pad input sequences
        max_review_length = 500
        X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        if train == True:
            self.train_model(cnn_filename, X_train, y_train, max_review_length)

        # load model
        model = tf.keras.models.load_model(cnn_filename)

        # Final evaluation of the model for training
        if train == True:
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1]*100))

        # Predictions made with non training model
        else:
            print("Printing Predicions")
            self.raw_results = model.predict(X_test)
            self.results = model.predict_classes(X_test)



    def train_model(self, cnn_filename, X_train, y_train, max_review_length):

        # MAKE A NEW MODEL
        # create the model
        top_words = 5000
        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=3, batch_size=64)
    
        # save model
        model.save(cnn_filename)


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