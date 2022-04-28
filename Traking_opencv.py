import cv2
import numpy as np
import time

video_path = "data.mp4"
cap = cv2.VideoCapture(video_path)

backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

while True:
    s = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 50, 150)
    # apply Closing
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    # thresh = cv2.threshold(closing, 254, 255, cv2.THRESH_BINARY)[1]
    print("FPS: ", 1 / (time.time() - s))
    cv2.imshow("frame", closing)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
