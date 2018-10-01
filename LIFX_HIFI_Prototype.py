import heapq

import cv2
import math
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream
from lifxlan import Light


def scaleValue(unscaled, to_min, to_max, from_min, from_max):
    return int((to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min)


light = Light("d0:73:d5:13:f4:17", "10.52.242.195")

brightness_Deque = deque(maxlen=10)
hue_angleDeque = deque(maxlen=10)
saturation_lineDeque = deque(maxlen=10)
vs = VideoStream(src=1).start()
state = 'OFF'
brightness = 0
prev_brightness = 0
current_brightness = 0
# greenCenter = (0,0)
yellowCenter = (-300, -300)

while True:
    image = vs.read()

    # Green
    # greenLower = (66, 113, 15)
    # greenUpper = (89, 255, 255)

    # greenLower = (66, 59, 61)
    # greenUpper = (89, 255, 255)

    # Yellow
    # yellowLower = (30, 20, 20)
    # yellowUpper = (37, 255, 255)

    # yellowLowqer = (33, 42, 59)
    # yellowUpper = (42, 255, 255)

    # Demo Settings
    greenLower = (38, 51, 52)
    greenUpper = (58, 255, 255)

    yellowLower = (24, 57, 111)
    yellowUpper = (36, 255, 255)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, None, iterations=2)
    greenMask = cv2.dilate(greenMask, None, iterations=2)

    yelloMask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yelloMask = cv2.erode(yelloMask, None, iterations=2)
    yelloMask = cv2.dilate(yelloMask, None, iterations=2)

    # finalMask = cv2.add(mask1, mask2)

    greenBinary = cv2.moments(greenMask)

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
            greenC = heapq.nlargest(4, greenCnts, key=cv2.contourArea)
            greenCArea = 0
            greenCCount = 0

            for c in greenC:
                greenM = cv2.moments(c)
                if greenM["m00"] > 1000:
                    greenCCount += 1
                    greenCArea += greenM["m00"]
                    if 0 < greenCCount < 1:
                        greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                        cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                        cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                        cv2.putText(image, 'Lifx: Brightness - {}'.format(brightness),
                                    (greenCenter[0] - 20, greenCenter[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        greenCenter = (
                            int(greenBinary["m10"] / greenBinary["m00"]), int(greenBinary["m01"] / greenBinary["m00"]))
                        cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                        cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                        cv2.putText(image, 'Lifx: Brightness - {}'.format(brightness),
                                    (greenCenter[0] - 20, greenCenter[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # print(greenCArea)

            if greenCArea > 0:
                prev_state = state
                state = 'ON'
            else:
                prev_state = state
                state = 'OFF'

            if state == 'ON' and prev_state == 'OFF':
                light.set_power("on")
            if state == 'OFF' and prev_state == 'ON':
                light.set_power("off")

            # Control brightness using Stack
            if greenCArea > 6480 or greenCCount > 1:
                brightness_Deque.append(scaleValue(greenCArea, 0, 100, 0, 25000))
                current_brightness = max(brightness_Deque)
                if current_brightness > 100:
                    current_brightness = 100
                elif current_brightness < 0:
                    current_brightness = 0
                brightness = current_brightness
                if math.fabs(current_brightness - prev_brightness) > 2:
                    prev_brightness = current_brightness
                    current_brightness = scaleValue(current_brightness, 0, 65535, 0, 100)
                    light.set_brightness(current_brightness, rapid=True)

            # Control brightness using Slider
            elif greenCArea < 7100 or (0 < greenCCount < 1):
                brightness_Deque.append(scaleValue(greenCenter[1], 0, 100, 256, 768))
                current_brightness = max(brightness_Deque)
                if current_brightness > 100:
                    current_brightness = 100
                elif current_brightness < 0:
                    current_brightness = 0
                brightness = current_brightness
                if math.fabs(current_brightness - prev_brightness) > 2:
                    prev_brightness = current_brightness
                    current_brightness = scaleValue(current_brightness, 0, 65535, 0, 100)
                    light.set_brightness(current_brightness, rapid=True)

            # Print ON/OFF near the center of the green block
            cv2.putText(image, state, (int(greenCenter[0] - 40), int(greenCenter[1] + 80)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if len(yellowCnts) > 0:
                yellowC = max(yellowCnts, key=cv2.contourArea)
                yellowM = cv2.moments(yellowC)
                # Checking if the area if yellow contour is greater than 1000
                if yellowM["m00"] > 1000:
                    yellowCenter = (int(yellowM["m10"] / yellowM["m00"]), int(yellowM["m01"] / yellowM["m00"]))
                    cv2.circle(image, yellowCenter, 5, (0, 0, 255), -1)
                    cv2.drawContours(image, yellowC, -1, (0, 255, 0), 2)
                    # Print near the center of the yellow block
                    cv2.putText(image, 'Settings', (yellowCenter[0] - 20, yellowCenter[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                yellowCenter = (-300, -300)

            p1 = np.array(greenCenter)
            p2 = np.array(yellowCenter)
            line_length = int(cv2.norm(p1 - p2))

            if line_length < 300:
                cv2.line(image, (greenCenter[0], greenCenter[1]), (yellowCenter[0], yellowCenter[1]), (0, 255, 0), 2,
                         lineType=cv2.LINE_AA)
                # Scaling line_length to 0-255 for display
                saturation = scaleValue(line_length, 0, 255, 300, 0)
                cv2.putText(image, 'Saturation: {}'.format(saturation), (yellowCenter[0] - 60, yellowCenter[1] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # Scaling line_length to 0-65535 for LIFX
                saturation = scaleValue(line_length, 0, 65535, 300, 0)
                prev_saturation = max(saturation_lineDeque) if saturation_lineDeque.__len__() > 0 else 0
                saturation_lineDeque.append(saturation)
                current_saturation = max(saturation_lineDeque)
                current_saturation = current_saturation if current_saturation < 65535 else 65535

                if math.fabs(current_saturation - prev_saturation) > 10:
                    light_color = light.get_color()
                    new_light_color = (light_color[0], current_saturation, light_color[2], light_color[3])
                    light.set_color(new_light_color, rapid=True)

                hue_angle = int(
                    math.atan2(greenCenter[1] - yellowCenter[1],
                               greenCenter[0] - yellowCenter[0]) * -180 / math.pi + 180)
                prev_hue_angle = max(hue_angleDeque) if hue_angleDeque.__len__() > 0 else 0
                hue_angleDeque.append(hue_angle)
                current_hue_angle = max(hue_angleDeque)

                if math.fabs(current_hue_angle - prev_hue_angle) > 1:
                    light_color = light.get_color()
                    hue_value = scaleValue(current_hue_angle, 0, 65535, 0, 359)
                    new_light_color = (hue_value, light_color[1], light_color[2], light_color[3])
                    light.set_hue(hue_value, rapid=True)
                    light.set_color(new_light_color, rapid=True)

                cv2.putText(image, 'Hue: {}'.format(max(hue_angleDeque)),
                            (int(yellowCenter[0] - 40), int(yellowCenter[1] + 80)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
