# Low-Fidelity Prototype: ORB_FeatureDetector.py
import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
vs = VideoStream(src=1).start()
while True:
    image = vs.read()
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=50)
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    # draw only keypoints location,not size and orientation
    image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow("Original", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
