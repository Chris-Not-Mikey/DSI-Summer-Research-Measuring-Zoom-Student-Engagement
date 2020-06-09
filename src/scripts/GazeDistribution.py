
import matplotlib.pyplot as plt
import numpy as np
import csv


gaze_angle_left = []
gaze_angle_right = []
confidence_threshold = 0.98

def read_csv_file():
    with open('../../data/gaze_outputs/up_down_test.csv', newline='') as f:
        reader = csv.reader(f)
        counter = 0
        for row in reader:
            if counter != 0:

                row_element = 0
                for i in row:
                

                    # Check Confidence. If less than 0.98, we will skip that value
                    if row_element == 3:
            
                        if float(i) < confidence_threshold:
                            break

                    # Gaze Angle X is in column  11
                    if row_element == 11:
                        gaze_angle_left.append(i)
                        
                    # Gaze Angle Y is in column 12
                    if row_element == 12:
                        gaze_angle_right.append(i)
                    
                    row_element = row_element + 1

            counter = counter + 1





read_csv_file()

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

# Plot Histogram on x
x = np.random.normal(size = 1000)
plt.hist(x, bins=50)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()




print("hello world")