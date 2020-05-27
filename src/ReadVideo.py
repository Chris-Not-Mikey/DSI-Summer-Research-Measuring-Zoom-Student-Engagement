import cv2
from ExtractFeatures import FeatExtract

cap = cv2.VideoCapture('../Videos/DSI_test_video_1.mov')


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    feat = FeatExtract(frame)
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
