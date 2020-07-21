import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
import pickle
import statistics
import csv

import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

class BlinkDetector:
    def __init__(self, eye_features_2D_list, ear_independent_time, ear_left_list, ear_right_list ):
        self.ear_threshold = 0.2
        self.number_blinks = 0
        self.number_hmm_blinks = 0
        self.eye_features_2D_list = eye_features_2D_list
        self.ear_independent_time = ear_independent_time
        self.ear_left_list = ear_left_list
        self.ear_right_list = ear_right_list
        self.ear_avg_list = []
        self.blink_duration = []
        self.blink_frequency = []
        self.blink_truth = []
        self.blink_train_test = []


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

    def calculate_right_EAR(self):
        for i in self.eye_features_2D_list:

            x_21_1 = (float(i["x_43"]) - float(i["x_47"])) ** 2
            y_21_1 = (float(i["y_43"]) - float(i["y_47"])) ** 2
            distance_right_1 = np.sqrt((x_21_1 + y_21_1))

            x_21_2 = (float(i["x_44"]) - float(i["x_46"])) ** 2
            y_21_2 = (float(i["y_44"]) - float(i["y_46"])) ** 2
            distance_right_2 = np.sqrt((x_21_2 + y_21_2))


            x_21_3 = (float(i["x_42"]) - float(i["x_45"])) ** 2
            y_21_3 = (float(i["y_45"]) - float(i["y_45"])) ** 2
            distance_right_3 = np.sqrt((x_21_3 + y_21_3))

            ear_right = (distance_right_1 + distance_right_2) / (2.0 * (distance_right_3))
            self.ear_right_list.append(ear_right)


    # Get the average of the EAR in both the left and the right eye
    def calculate_avg_EAR(self):

        temp_avg = zip(self.ear_left_list, self.ear_right_list)
        for i in temp_avg:
            EAR = (i[0] + i[1])/2
            self.ear_avg_list.append(EAR)

    def write_EAR_to_CSV(self, name):
        path = '../../data/blink_outputs/' + name + '_EAR.csv'
        with open(path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

            counter = 0
            csv_writer.writerow(["time", "EAR", "Blink"])
            for i in self.ear_avg_list:
                row = []
                row.append(self.ear_independent_time[counter])
                row.append(self.ear_avg_list[counter])
                row.append(1)

                csv_writer.writerow([row[0], row[1], row[2]])
                counter = counter + 1

        csvfile.close()

    def write_results_to_CSV(self, name):
        path = '../../data/blink_outputs/' + name + '_blink_results.csv'
        with open(path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

            counter = 0
            csv_writer.writerow(["time", "EAR", "Blink"])
            for i in self.results:
                row = []
                row.append(self.ear_independent_time[counter])
                row.append(self.ear_avg_list[counter])
                row.append(i)

                csv_writer.writerow([row[0], row[1], row[2]])
                counter = counter + 1

        csvfile.close()

         
    #plot ear against time
    def plot_EAR_vs_time(self, name):
        path = "../../data/kernel_plots/" + name + "_EAR"
        plt.scatter(self.ear_independent_time, self.ear_avg_list)
        plt.savefig(path)


    # Thresholding is the simplest blink detection method
    # We will not use this for our feature data
    # The HMM based apporach will be used instead
    def threshold_predict_number_blinks(self):

        # Get local minimums from the EAR data recorded
        # Function returns the indices of the local mins
        min_index = argrelextrema(np.array(self.ear_avg_list), np.less, order=20)
        ear_array = np.array(self.ear_avg_list)

        # Use the indices to max a list of minumns
        minimums = []
        for i in min_index:
            minimums.append(ear_array[i])

        # see if the minimums make the cutoff
        for j in minimums:
            for k in j:
                if k < self.ear_threshold:
                    self.number_blinks = self.number_blinks + 1


    # experimental: A possible better version for calculating blink
    # rate that uses CNN instead of HMM. 
    def cnn_predict_number_blinks(self, name, train):

        cnn_filename = "../../data/cnn_blink_models/"
        np.random.seed(7)
        self.read_csv_file(name)
   
        X = np.array(self.blink_train_test, dtype=np.float32)
        y = np.array(self.blink_truth, dtype=np.float32)

        X_train = X[0:500]
        y_train = y[0:500]
    
        X_val = X[500:1000]
        y_val = y[500:1000]

        X_test = X[1000:]
        y_test = y[1000:]

        # truncate and pad input sequences
        # max_review_length = 500
        # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
        # X_val = sequence.pad_sequences(X_val, maxlen=max_review_length)
        # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

        # y_train = sequence.pad_sequences(y_train, maxlen=max_review_length)
        # y_val = sequence.pad_sequences(y_val, maxlen=max_review_length)
        # y_test = sequence.pad_sequences(y_test, maxlen=max_review_length)


        if train == True:
            #self.train_model(cnn_filename, X_train, y_train, X_val, y_val, X_test, y_test, max_review_length)
            model = Sequential()
            model.add(Dense(1, activation='sigmoid', input_dim=2))
            opt = keras.optimizers.Adam(learning_rate=0.02)
            model.compile(
                optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            model.fit(X_train, y_train, epochs=10, batch_size=3)

            self.results = model.predict_classes(X_test, batch_size=3, verbose=1)
            print(self.results)
            print(len(self.results))
            print(len(self.results[0]))

            # for i in self.results:
            #     print(i)



        # Predictions made with non training model
        else:
            # load model
            model = tf.keras.models.load_model(cnn_filename, compile=True)
            print("Printing Predicions")
            self.raw_results = model.predict(X_test)
            print(self.raw_results)
          
            #self.results = np.argmax(self.raw_results, axis=1) # model.predict_classes(X_train)
            self.results = (model.predict(X_test) > 0.5).astype("int32")
            print(self.results)
            self.write_results_to_CSV(name)



    # A more advanced version for predicting the blink frequency and the blink
    # duration. Fare more accurate than simple thresholding.
    # Also, takes into account the duration of the blink, unlike thresholding.
    def hmm_predict_number_blinks(self, nSamples, name, train):
     
        # load EAR data per frame
        AnnualQ = self.ear_avg_list

        hmm_filename = "../../data/hmm_models/" + "try_this" + ".pkl"


        Q = np.array(AnnualQ)
        if train == True:
            #  log transform the data and fit the HMM
            #Q = np.log(AnnualQ)
            #Q = AnnualQ
           
            hidden_states, mus, sigmas, P, logProb, samples = fitHMM(Q, 100)
    
            #fit Gaussian HMM to Q
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, init_params="stm", params="stc")

            model.startprob_ = np.array([0.05, 0.95])
            model.transmat_ = np.array([[0.3, 0.7],[0.3, 0.7]])
            model.means_ = np.array([[500, 500], [500, 500]])

        
            model.fit(np.reshape(Q,[len(Q),1]))
            with open(hmm_filename, "wb") as file: pickle.dump(model, file)


        with open(hmm_filename, "rb") as file:
            model = pickle.load(file)

        
        # classify each observation as state 0 or 1
        hidden_states = model.predict(np.reshape(Q,[len(Q),1]))
    
        # find parameters of Gaussian HMM
        mus = np.array(model.means_)
        sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
        P = np.array(model.transmat_)
    
        # find log-likelihood of Gaussian HMM
        logProb = model.score(np.reshape(Q,[len(Q),1]))
    
        # generate nSamples from Gaussian HMM
        samples = model.sample(nSamples)
    
        # re-organize mus, sigmas and P so that first row is lower mean (if not already)
        if mus[0] > mus[1]:
            mus = np.flipud(mus)
            sigmas = np.flipud(sigmas)
            P = np.fliplr(np.flipud(P))
            hidden_states = 1 - hidden_states


        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        xs = np.arange(len(Q))+0
        masks = hidden_states == 0
        ax.scatter(xs[masks], Q[masks], c='r', label='Blink')
        masks = hidden_states == 1
        ax.scatter(xs[masks], Q[masks], c='b', label='Open')
        ax.plot(xs, Q, c='k')
        
        ax.set_xlabel('frame')
        ax.set_ylabel('EAR')
        fig.subplots_adjust(bottom=0.2)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)

        filename = "../../data/kernel_plots/" + name + "_HMM_EAR.png"
        fig.savefig(filename)
        fig.clf()


        # temp variables for tracking blink duration and frequency
        blink_length = 0
        blink = False
        blink_duration_rolling_list = []
        blink_duration_rolling_mean = 0
        blink_frequency_rolling_mean = 0
        counter = 0

        for i in hidden_states:
            if i == 0:
                blink = True
                blink_length = blink_length + 1
            

            if i == 1 and blink == True:
                
                # 2 and 21 correspond to 60ms and 700ms respectively (see 4.2.2 of Soukupova-TR-2016-05)
                if blink_length >= 2 and blink_length <= 21:
                    self.number_hmm_blinks = self.number_hmm_blinks + 1
                    blink_frequency_rolling_mean = blink_frequency_rolling_mean + 1

                    # divide by 30 fps to convert to seconds
                    blink_duration_rolling_list.append(blink_length/30)
                    blink_duration_rolling_mean = statistics.mean(blink_duration_rolling_list)
                    
                blink = False
                blink_length = 0

            # 900 refers to 900 frames, equivalent to 30 seconds.
            if counter == 900:

                self.blink_duration.append(blink_duration_rolling_mean)
                self.blink_frequency.append(blink_frequency_rolling_mean)

                # Clear for next rolling calculation
                blink_duration_rolling_mean = 0
                blink_duration_rolling_list = []
                blink_frequency_rolling_mean = 0

                # Reset counter. 
                counter = 0

            counter = counter + 1

    
        return hidden_states, mus, sigmas, P, logProb, samples


    # Final Output for Engagement prediction (duration)
    def get_blink_duration(self):
        return self.blink_duration


    # Final Output for Engagement prediction (frequency)
    def get_blink_frequency(self):
        return self.blink_frequency
    

    def get_blinks(self):
        return self.number_hmm_blinks


    def train_model(self, cnn_filename, X_train, y_train, X_val, y_val, X_test, y_test, max_review_length):

        estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, X_train, y_train, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

     


    def read_csv_file(self, name):
        path = '../../data/blink_outputs/' + name + '_EAR.csv'
        with open(path) as s:
            reader = csv.reader(s)

            counter = 0
            for row in reader:
                if counter != 0:
                    self.blink_train_test.append([(np.float32(row[0]), np.float32(row[1]))])
                    self.blink_truth.append([np.float32(row[2])])
            

                counter = counter + 1


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model