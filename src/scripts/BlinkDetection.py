import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
import pickle
import statistics

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

    def calculate_avg_EAR(self):

        temp_avg = zip(self.ear_left_list, self.ear_right_list)
        for i in temp_avg:
            EAR = (i[0] + i[1])/2
            self.ear_avg_list.append(EAR)
         


    def plot_EAR_vs_time(self, name):
        path = "../../data/kernel_plots/" + name + "_EAR"
        plt.scatter(self.ear_independent_time, self.ear_avg_list)
        plt.savefig(path)


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




    def hmm_predict_number_blinks(self, nSamples, name):
     
        # load EAR data per frame
        AnnualQ = self.ear_avg_list
        
        #  log transform the data and fit the HMM
        #Q = np.log(AnnualQ)
        #Q = AnnualQ
        Q = np.array(AnnualQ)
        # hidden_states, mus, sigmas, P, logProb, samples = fitHMM(logQ, 100)
 
        # fit Gaussian HMM to Q
        # model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, init_params="stm", params="stc")

        # model.startprob_ = np.array([0.05, 0.95])
        # model.transmat_ = np.array([[0.3, 0.7],[0.3, 0.7]])
        # model.means_ = np.array([[500, 500], [500, 500]])

        # # model.startprob_ = np.array([0.99, 0.01])
        # # model.transmat_ = np.array([[0.99, 0.01],[0.99, 0.01]])
        # # model.means_ = np.array([[200,  0.15], [200, 0.35]])
        
        # model.fit(np.reshape(Q,[len(Q),1]))

        hmm_filename = "../../data/hmm_models/" + "try_this" + ".pkl"
        #with open(hmm_filename, "wb") as file: pickle.dump(model, file)


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

                    blink_duration_rolling_list.append(blink_length)
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
    

    def get_blinks(self):

        print(self.blink_duration)
        print(self.blink_frequency)
        return self.number_hmm_blinks

        