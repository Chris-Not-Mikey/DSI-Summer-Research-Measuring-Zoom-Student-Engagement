from imutils import face_utils
import numpy as np
import argparse
import imutils
import math
import dlib
import cv2


# dlib face detector
detector = dlib.get_frontal_face_detector()

# dlib "shape" detectector
# this will detect the eyes, eyebrows, mouth, jaw, and nose
# on a face that is detected by the dlib face detector
# For now, we only really care about the eyes
#  (although we will use the nose to get the distance between the eyes)
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# Open and image, resize it, and change the color scheme
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

	# Default values for variables
	# These will all be filled in the loop if
	# there are no errors
	eye_left = image
	eye_right = image
	nose_width = 100 
	
	# loops through the facial features (eg, nose, right eye, mouth, jaw etc)
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		
		if (name == "left_eye" or name == "right_eye"):

			# clone the original image so we can draw on it, then
			# display the name of the face part on the image
			clone = image.copy()
			cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 255), 2)

			# loop over the subset of facial landmarks, drawing the
			# specific face part (puts a dot outline on specified feature)
			for (x, y) in shape[i:j]:
				cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

			# extract the rectangular ROI(aka Region Of Interest) of the face region as a separate image
			# Note: this will not be used in any computation. We use this rectangular ROI for now
			# to debug the elliptical ROI  
			# TODO remove
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]

			# Create a mask for the elliptical ROI
			# center, axes_lenght, color, and thickness are all parameters for the elliptical mask
			mask = np.zeros(image.shape, dtype=np.uint8)
			circle_center = (math.floor(x + w/2),y+2)
			axes_length = (math.floor(w/2), math.floor(h/2))
			color = (255, 255, 255)
			# If thicknes value is too small (say 5), there will be small artifacts in the eye which is unacceptable
			thickness = 10 
			cv2.ellipse(mask, circle_center, axes_length, 0, 0, 360, color, thickness )

			# apply the mask to the original image to extract the elliptical ROI.
			clone_masked = image & mask

			# crop the mask (this removes a lot of unnessary black space)
			clone_masked = clone_masked[circle_center[1] - 2*axes_length[1]:circle_center[1] + 2*axes_length[1],
                           circle_center[0] - 2*axes_length[0]:circle_center[0] + 2*axes_length[0], :]

			roi = imutils.resize(roi, width=250, height=250, inter=cv2.INTER_CUBIC)
			
			
			cv2.imshow("Mask", clone_masked)
			cv2.imshow("ROI", roi)
			cv2.imshow("Image", clone)

			# store the left eye and the right eye in their own variable
			# we will use these two saved images of the individual eyes
			# and merge them together later
			if (name == "left_eye"):
				eye_left = clone_masked
			else:
				eye_right = clone_masked
		
			# For now we control the flow manualy by key inputs.
			# press any key to go from one eye to the next eye
			# This manual input is for intial debugging purposes and will
			# later be removed
			# TODO remove
			cv2.waitKey(0)

		# Get the nose feature
		# We get with width of the nose
		# we use this with the individual eye ROI we got above to find out
		# how much distance we should put in between the eyes
		if (name == "nose"):
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			nose_width = w
		
		else:
			continue

	# concatenate ROI

	# first get shape of the regions, and the total height and width of the merged image
	eye_right_shape = eye_right.shape
	eye_left_shape = eye_left.shape
	height = max(eye_right_shape[0], eye_left_shape[0])
	width = max(eye_right_shape[1], eye_left_shape[1])

	# now concatenate
	black = np.zeros((height + 20, width*2 + nose_width +20,3), np.uint8)
	black[10:10 + eye_right_shape[0] , 10:10 + eye_right_shape[1]] = eye_right
	black[10:10 + eye_left_shape[0] , eye_right_shape[1] + nose_width:eye_right_shape[1] + nose_width + eye_left_shape[1]] = eye_left

	cv2.imshow("Black template", black)
	cv2.imwrite("out.jpg", black)
	cv2.waitKey(0)


print("hello world")