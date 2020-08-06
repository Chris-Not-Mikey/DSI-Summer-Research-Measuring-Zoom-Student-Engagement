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
            temp_list = i.get_master_list()
            temp_array = np.array(temp_list, dtype=np.float32)
            core_list.append(temp_array[0:1]) #Adjust this to 30 mins when the time comes


        if len(core_list) != len(self.engagement_known):
            print("WARNING! The lengths are NOT the same. Major issue")

        X = np.array(core_list, dtype=np.float32)
        y = np.array(self.engagement_known, dtype=np.int64)

        start = 0
        train = True
        if train == True:
            start = int(len(X)/2)

        else:
            start = 0


        print(start)
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
        model.add(Dense(1, activation='sigmoid', input_shape= [1, 1, 7]))
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
        print(self.results)
        # self.write_results_to_csv(start)

