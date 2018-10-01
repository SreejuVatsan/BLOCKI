import math
import numpy as np
import cv2
import imutils
from collections import deque
from imutils.video import VideoStream
angleDeque = deque(maxlen=10)
vs = VideoStream(src=1).start()
while True:
    image = vs.read()
    greenLower = (66, 113, 15)
    greenUpper = (89, 255, 255)
    yellowLower = (30, 59, 184)
    yellowUpper = (37, 255, 255)
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
                # state = 'ON'
                greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                cv2.putText(image, 'Center', (greenCenter[0] - 20, greenCenter[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if len(yellowCnts) > 0:
            yellowC = max(yellowCnts, key=cv2.contourArea)
            yellowM = cv2.moments(yellowC)
            # Checking if the area if yellow contour is greater than 1000
            if yellowM["m00"] > 1000:
                yellowCenter = (int(yellowM["m10"] / yellowM["m00"]), int(yellowM["m01"] / yellowM["m00"]))
                cv2.circle(image, yellowCenter, 5, (0, 0, 255), -1)
                cv2.drawContours(image, yellowC, -1, (0, 255, 0), 2)
                cv2.putText(image, 'Movable', (yellowCenter[0] - 20, yellowCenter[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        p1 = np.array(greenCenter)
        p2 = np.array(yellowCenter)
        if cv2.norm(p1 - p2) < 260:
            cv2.line(image, (greenCenter[0], greenCenter[1]), (yellowCenter[0], yellowCenter[1]), (0, 255, 0), 2,
                 lineType=cv2.LINE_AA)
           angle = int(math.atan2(greenCenter[1] - yellowCenter[1], greenCenter[0] - yellowCenter[0]) * -180 / math.pi + 180)
            angleDeque.append(angle)
            cv2.putText(image, str(max(angleDeque)), (int(yellowCenter[0] - 40), int(yellowCenter[1] + 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    except Exception as inst:
        print(inst)
    cv2.imshow("Original", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()