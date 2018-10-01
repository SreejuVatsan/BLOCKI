import cv2
import math
import numpy as np
import imutils
from collections import deque
from imutils.video import VideoStream
from lifxlan import Light


def scaleValue(unscaled, to_min, to_max, from_min, from_max):
    return int((to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min)


# cv2.namedWindow("Window - Main", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Window - Main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

light = Light("d0:73:d5:13:f4:17", "10.52.242.65")

brightness_Deque = deque(maxlen=10)
hue_angleDeque = deque(maxlen=10)
saturation_lineDeque = deque(maxlen=10)
vs = VideoStream(src=1).start()
state = 'OFF'
brightness = 0
# image_height = 1024
# image_width = 1280
# image_center = (image_height/2, image_width/2)
# greenCenter = (0,0)
yellowCenter = (-300, -300)

while True:
    image = vs.read()

    # greenLower = (66, 113, 15)
    # greenUpper = (89, 255, 255)
    # 9 pm
    greenLower = (66, 59, 61)
    greenUpper = (89, 255, 255)

    yellowLower = (30, 59, 184)
    yellowUpper = (42, 255, 255)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenMask = cv2.inRange(hsv, greenLower, greenUpper)
    greenMask = cv2.erode(greenMask, None, iterations=2)
    greenMask = cv2.dilate(greenMask, None, iterations=2)

    yelloMask = cv2.inRange(hsv, yellowLower, yellowUpper)
    yelloMask = cv2.erode(yelloMask, None, iterations=2)
    yelloMask = cv2.dilate(yelloMask, None, iterations=2)

    # finalMask = cv2.add(mask1, mask2)
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
                prev_state = state
                state = 'ON'
                greenCenter = (int(greenM["m10"] / greenM["m00"]), int(greenM["m01"] / greenM["m00"]))
                cv2.circle(image, greenCenter, 5, (0, 0, 255), -1)
                cv2.drawContours(image, greenC, -1, (0, 255, 0), 2)
                brightness = scaleValue(brightness, 0, 100, 256, 768)
                cv2.putText(image, 'Lifx: Brightness - {}'.format(brightness),
                            (greenCenter[0] - 20, greenCenter[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                prev_state = state
                state = 'OFF'

            if state == 'ON' and prev_state == 'OFF':
                light.set_power("on")
                # print("{} - {}".format(prev_state, state))
            if state == 'OFF' and prev_state == 'ON':
                light.set_power("off")

            # if 255 < greenCenter[1] < 769:
            prev_brightness = max(brightness_Deque) if brightness_Deque.__len__() > 0 else 0
            if greenCenter[1] < 256:
                brightness = 256
            elif greenCenter[1] > 768:
                brightness = 768
            else:
                brightness = greenCenter[1]
            # brightness = scaleValue(greenCenter[1], 0, 65535, 256, 768)
            brightness_Deque.append(brightness)
            current_brightness = max(brightness_Deque)
            if math.fabs(current_brightness - prev_brightness) > 2:
                current_brightness = scaleValue(current_brightness, 0, 65535, 256, 768)
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
            saturation = scaleValue(line_length, 0, 255, 300, 76)
            cv2.putText(image, 'Saturation: {}'.format(saturation), (yellowCenter[0] - 60, yellowCenter[1] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Scaling line_length to 0-65535 for LIFX
            saturation = scaleValue(line_length, 0, 65535, 300, 76)
            prev_saturation = max(saturation_lineDeque) if saturation_lineDeque.__len__() > 0 else 0
            saturation_lineDeque.append(saturation)
            current_saturation = max(saturation_lineDeque)
            current_saturation = current_saturation if current_saturation < 65535 else 65535

            if math.fabs(current_saturation - prev_saturation) > 10:
                light_color = light.get_color()
                new_light_color = (light_color[0], current_saturation, light_color[2], light_color[3])
                light.set_color(new_light_color, rapid=True)
                print(current_saturation)

            # angle = int(math.degrees(math.atan2(greenCenter[1] - yellowCenter[1], greenCenter[0] - yellowCenter[0])))
            hue_angle = int(
                math.atan2(greenCenter[1] - yellowCenter[1], greenCenter[0] - yellowCenter[0]) * -180 / math.pi + 180)
            prev_hue_angle = max(hue_angleDeque) if hue_angleDeque.__len__() > 0 else 0
            hue_angleDeque.append(hue_angle)
            current_hue_angle = max(hue_angleDeque)

            if math.fabs(current_hue_angle - prev_hue_angle) > 1:
                light_color = light.get_color()
                hue_value = scaleValue(current_hue_angle, 0, 65535, 0, 359)
                new_light_color = (hue_value, light_color[1], light_color[2], light_color[3])
                # light.set_hue(hue_value, rapid=True)
                light.set_color(new_light_color, rapid=True)
                # print("{} - {} : {}".format(prev_hue_angle, current_hue_angle, hue_value))

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
