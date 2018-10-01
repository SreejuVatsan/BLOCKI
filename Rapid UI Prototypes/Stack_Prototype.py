import heapq
from collections import deque
import cv2
import imutils
from imutils.video import VideoStream


def scaleValue(unscaled, to_min, to_max, from_min, from_max):
    return int((to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min)


# areaDeque = deque(maxlen=10)
valueDeque = deque(maxlen=10)
vs = VideoStream(src=1).start()

while True:
    image = vs.read()

    greenLower = (66, 182, 28)
    greenUpper = (89, 255, 255)
    # 9 pm
    # greenLower = (66, 110, 40)
    # greenUpper = (89, 255, 255)

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
            greenCArea = 0

            for c in greenC:
                greenM = cv2.moments(c)
                if greenM["m00"] > 1000:
                    greenCCount += 1
                    greenCArea += greenM["m00"]
                    print("{}".format(greenCArea))
                    valueDeque.append(scaleValue(greenCArea, 0, 100, 0, 21200))
                    # areaDeque.append(greenCArea)
                    greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                    cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                    cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                    # cv2.putText(image, 'Green Block {}'.format(greenCCount), (greenCenter[0] - 20, greenCenter[1] - 20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        value = max(valueDeque)
        # print(value)
        if value > 100:
            value = 100
        elif value < 0:
            value = 0
        cv2.putText(image, '{}'.format(value), (greenCenter[0] - 20, greenCenter[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # average = scaleValue(average, 0, 100, 0, 26000)
        # print("{}".format(int(value)))

    except Exception as inst:
        print(inst)

    cv2.imshow("Original", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
