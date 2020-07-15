import numpy as np
import sys
import pickle
import csv

import tensorflow as tf

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
        self.master_list = []
        self.complete_list = []
        self.engagement = []

        


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

                    engagement_element = np.int64(row[8])
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


    def predict_engagement(self):

        cnn_filename = "../../data/cnn_models/"

        np.random.seed(7)
        # load the dataset but only keep the top n words, zero the rest
        top_words = 5000
        #(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

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

        # MAKE A NEW MODEL
        # # create the model
        # embedding_vecor_length = 32
        # model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        # model.add(LSTM(100))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        # model.fit(X_train, y_train, epochs=3, batch_size=64)
    
        # # save model
    
        # model.save(cnn_filename)


        # load model
        model = tf.keras.models.load_model(cnn_filename)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(X_train)
        print("Accuracy: %.2f%%" % (scores[1]*100))





# if __name__ == "__main__":

#     # get name from command string
#     name = sys.argv[1]
#     predictor = EngagementPredictor(name)
#     predictor.read_csv_file()
#     predictor.predict_engagement()



       



    

      