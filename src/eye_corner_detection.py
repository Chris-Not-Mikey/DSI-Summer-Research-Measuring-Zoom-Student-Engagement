from imutils import face_utils
import numpy as np
import argparse
import imutils
import math
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")



image = cv2.imread("../data/Columbia_Gaze_Data_Set/0001/0001_2m_15P_10V_0H.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)


# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually

	eye_left = image
	eye_right = image
	nose_width = 100
	
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		print(name)
		

		counter = 0
		if (name == "left_eye" or name == "right_eye"):


			# clone the original image so we can draw on it, then
			# display the name of the face part on the image

			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y - h:math.ceil(y + 1.5*h), x:x + w]
			roi = imutils.resize(roi, width=250, height=250, inter=cv2.INTER_CUBIC)
			# show the particular face part
			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)

			if (name == "left_eye"):
				print("we get here left")
				eye_left = roi
			else:
				print("we get here right")
				eye_right = roi
		
			cv2.waitKey(0)


		if (name == "nose"):
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			nose_width = x
		
			

		else:
			continue
	# visualize all facial landmarks with a transparent overlay


	# concatenate ROI
	nose_width = 100
	eye_right_shape = eye_right.shape
	eye_left_shape = eye_left.shape

	height = max(eye_right_shape[0], eye_left_shape[0])
	width = max(eye_right_shape[1], eye_left_shape[1])
	black = np.zeros((height + 20, width*2 + nose_width +20,3), np.uint8)

	black[10:10 + eye_right_shape[0] , 10:10 + eye_right_shape[1]] = eye_right
	black[10:10 + eye_left_shape[0] , eye_right_shape[1] + nose_width:eye_right_shape[1] + nose_width + eye_left_shape[1]] = eye_left

	cv2.imshow("Black template", black)

	#vis = np.concatenate((eye_left, eye_right), axis=1)
	#cv2.imwrite('out.png', vis)
	#cv2.imshow("merge", vis)
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)


print("hello world")