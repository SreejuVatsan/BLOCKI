# Low-Fidelity Prototype: ShiTomasi_FeatureDetector.py
import numpy as np
import cv2
from collections import deque
from imutils.video import VideoStream
vs = VideoStream(src=1).start()
while True:
    image = vs.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
 corners = np.int0(corners)
 for i in corners:
     x, y = i.ravel()
     cv2.circle(image, (x, y), 3, (0,255,255), -1)
  cv2.imshow("Original", image)
  key = cv2.waitKey(1) & 0xFF
  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
  	break
cv2.destroyAllWindows()
