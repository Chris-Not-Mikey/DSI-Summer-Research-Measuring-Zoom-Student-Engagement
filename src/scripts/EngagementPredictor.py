import numpy as np
import sys
import pickle
import csv

from sklearn.svm import LinearSVC
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM


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
                    self.gaze_length.append(row[1])
                    self.saccade_frequency.append(row[2])
                    self.saccade_length.append(row[3])
                    self.saccade_velocity.append(row[4])
                    self.blink_rate.append(row[5])
                    self.blink_duration.append(row[6])
                    self.pupil_size.append(row[7])

                    engagement_element = [np.int32(row[8])]
                    self.engagement.append(engagement_element)
       
                    element = []

                    element_1 = [np.float32(row[1])]
                    element_2 = [np.float32(row[2])]
                    element_3 = [np.float32(row[3])]
                    element_4 = [np.float32(row[4])]
                    element_5 = [np.float32(row[5])]
                    element_6 = [np.float32(row[6])]
                    element_7 = [np.float32(row[7])]


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




        model = ChainCRF()
        ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=11)
        
        X = np.array(self.master_list, dtype=np.float32)
        y = np.array(self.engagement, dtype=np.int32)
      
        # X_train = X[0:5]
        # y_train = y[0:5]

        ssvm.fit(X, y)

        # X_test = X[5]
        # y_test = y[5]

        print("Test score with chain: %f" % ssvm.score(X, y))


if __name__ == "__main__":

    # get name from command string
    name = sys.argv[1]
    predictor = EngagementPredictor(name)
    predictor.read_csv_file()
    predictor.predict_engagement()



       



    

      