import cv2
from ExtractFeatures import FeatExtract

cap = cv2.VideoCapture('../Videos/DSI_test_video_1.mov')


face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Load the cascades classifiers
if not face_cascade.load(cv2.samples.findFile('../haarcascades/haarcascade_frontalface_alt.xml')):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile('../haarcascades/haarcascade_eye_tree_eyeglasses.xml')):
    print('--(!)Error loading eyes cascade')
    exit(0)


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    feat = FeatExtract(frame, face_cascade, eyes_cascade)
    if ret == True:

        # Display the resulting frame
        # cv2.imshow('Frame',frame)

        feat.detectAndDisplay()

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


print("Hello world!")
