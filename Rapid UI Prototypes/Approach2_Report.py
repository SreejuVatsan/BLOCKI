import numpy as np
import cv2
import imutils
import heapq
from imutils.video import VideoStream


def nothing(x):
    pass


cv2.namedWindow('image')

# easy assigments
hh = 'Hue High'
hl = 'Hue Low'
sh = 'Saturation High'
sl = 'Saturation Low'
vh = 'Value High'
vl = 'Value Low'

cv2.createTrackbar(hl, 'image', 0, 179, nothing)
cv2.createTrackbar(hh, 'image', 0, 179, nothing)
cv2.createTrackbar(sl, 'image', 0, 255, nothing)
cv2.createTrackbar(sh, 'image', 0, 255, nothing)
cv2.createTrackbar(vl, 'image', 0, 255, nothing)
cv2.createTrackbar(vh, 'image', 0, 255, nothing)

# Green Range
cv2.setTrackbarPos(hl, 'image', 66)
cv2.setTrackbarPos(hh, 'image', 89)
cv2.setTrackbarPos(sl, 'image', 113)
cv2.setTrackbarPos(sh, 'image', 255)
cv2.setTrackbarPos(vl, 'image', 15)
cv2.setTrackbarPos(vh, 'image', 255)

# yellowLower = (30, 59, 184)
# yellowUpper = (37, 255, 255)

# Yellow Range
# cv2.setTrackbarPos(hl, 'image', 33)
# cv2.setTrackbarPos(hh, 'image', 42)
# cv2.setTrackbarPos(sl, 'image', 59)
# cv2.setTrackbarPos(sh, 'image', 255)
# cv2.setTrackbarPos(vl, 'image', 184)
# cv2.setTrackbarPos(vh, 'image', 255)

vs = VideoStream(src=1).start()

while True:
    img = vs.read()
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # read trackbar positions for all
    hul = cv2.getTrackbarPos(hl, 'image')
    huh = cv2.getTrackbarPos(hh, 'image')
    sal = cv2.getTrackbarPos(sl, 'image')
    sah = cv2.getTrackbarPos(sh, 'image')
    val = cv2.getTrackbarPos(vl, 'image')
    vah = cv2.getTrackbarPos(vh, 'image')
    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])

    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)
    mask_morph = cv2.erode(mask, None, iterations=4)
    mask_morph = cv2.dilate(mask, None, iterations=4)
    # res = cv2.bitwise_and(frame,frame, mask =mask)

    cv2.imshow('Original',img)
    cv2.imshow('Gaussian Blur',blurred)
    # cv2.imshow('image',mask)
    # cv2.imshow('Mask without Morphological Transformation', mask)
    # cv2.imshow('Mask with Morphological Transformation', mask_morph)
    # cv2.imshow("Original", img)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
