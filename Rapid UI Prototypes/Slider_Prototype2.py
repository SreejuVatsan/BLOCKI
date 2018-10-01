import math
import numpy as np
import cv2
import imutils
from collections import deque
import heapq
from imutils.video import VideoStream


def scaleValue(unscaled, toMin, toMax, fromMin, fromMax):
    return int((toMax-toMin)*(unscaled-fromMin)/(fromMax-fromMin)+toMin)

vs = VideoStream(src=1).start()

# initLocationSet = False
# initLocation = ()

while True:
    image = vs.read()
    # evening
    # greenLower = (79, 50, 20)
    # greenUpper = (92, 255, 255)

    # 9 pm
    greenLower = (66, 59, 61)
    greenUpper = (89, 255, 255)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, None, iterations=2)
    greenMask = cv2.dilate(greenMask, None, iterations=2)

    # finalMask = cv2.add(mask1, mask2)
    greenCnts = cv2.findContours(greenMask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    greenCnts = greenCnts[0] if imutils.is_cv2() else greenCnts[1]

    center = None

    try:
        # only proceed if at least one contour was found
        if len(greenCnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            # greenC = max(greenCnts, key=cv2.contourArea)
            greenC = heapq.nlargest(4, greenCnts, key=cv2.contourArea)
            greenCCount = 0
            for c in greenC:
                greenM = cv2.moments(c)
                if greenM["m00"] > 1000:
                    greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                    # if initLocationSet == False:
                    #     initLocation = greenCenter
                    #     initLocationSet = True
                    # cv2.circle(image, initLocation, 5, (0, 0, 255), -1)
                    cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                    # cv2.line(image, (initLocation[0], initLocation[1]), (greenCenter[0], greenCenter[1]), (0, 255, 0),
                    #          2,
                    #          lineType=cv2.LINE_AA)
                    cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                    # cv2.putText(image, 'Green - Axis', (greenCenter[0] - 20, greenCenter[1] - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # p1 = np.array(initLocation)
                    # p2 = np.array(greenCenter)
                    # sliderValue = int(cv2.norm(p1 - p2))
                    # sliderValue = scaleValue(sliderValue, 0, 100, 0, 300)
                    # sliderValue = sliderValue if sliderValue <= 100 else 100
                    brightness = scaleValue(brightness, 0, 100, 256, 768)
                    # cv2.putText(image, '{}'.format(brightness),
                    #             (int(initLocation[0]) + 15,
                    #              int(initLocation[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (255, 255, 255), 2)
                    cv2.putText(image, '{}'.format(brightness),
                                (greenCenter[0] - 20, greenCenter[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # greenCCount += 1

                if greenCenter[1] < 256:
                    brightness = 256
                elif greenCenter[1] > 768:
                    brightness = 768
                else:
                    brightness = greenCenter[1]
        # else:
        #     initLocationSet = False



    except Exception as inst:
        print(inst)

    cv2.imshow("Original", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()