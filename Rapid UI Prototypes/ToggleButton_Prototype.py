import cv2
import math
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream
from lifxlan import Light
def scaleValue(unscaled, to_min, to_max, from_min, from_max):
    return int((to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min)
light = Light("d0:73:d5:13:f4:17", "10.52.242.14")
brightness_Deque = deque(maxlen=10)
hue_angleDeque = deque(maxlen=10)
saturation_lineDeque = deque(maxlen=10)
vs = VideoStream(src=0).start()
state = 'OFF'
brightness = 0
yellowCenter = (-300, -300)
while True:
    image = vs.read()
    greenLower = (66, 113, 15)
    greenUpper = (89, 255, 255)
    yellowLower = (33, 59, 184)
    yellowUpper = (42, 255, 255)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, None, iterations=2)
    greenMask = cv2.dilate(greenMask, None, iterations=2)
    yelloMask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yelloMask = cv2.erode(yelloMask, None, iterations=2)
    yelloMask = cv2.dilate(yelloMask, None, iterations=2)
    greenCnts = cv2.findContours(greenMask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    yellowCnts = cv2.findContours(yelloMask.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    greenCnts = greenCnts[0] if imutils.is_cv2() else greenCnts[1]
    yellowCnts = yellowCnts[0] if imutils.is_cv2() else yellowCnts[1]
    center = None
    try:
        # only proceed if at least one contour was found
        if len(greenCnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            greenC = max(greenCnts, key=cv2.contourArea)
            greenM = cv2.moments(greenC)
            if greenM["m00"] > 1000:
                state = 'ON'
                greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                brightness = scaleValue(brightness, 0, 100, 256, 768)
                cv2.putText(image, 'Green Object', (greenCenter[0] - 20, greenCenter[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                state = 'OFF'
        # Print ON/OFF near the center of the green block
        cv2.putText(image, state, (int(greenCenter[0] - 40), int(greenCenter[1] + 80)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    except Exception as inst:
        # if inst.__str__().index("greenCenter") < 0 or inst.__str__().index("yellowCenter") < 0: print(inst)
        print(inst)
    cv2.imshow("Window - Main", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
light.set_power("off", rapid=True)
cv2.destroyAllWindows()