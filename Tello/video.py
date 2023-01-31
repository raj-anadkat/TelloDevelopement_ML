from djitellopy import tello
import time as time
import cv2

drone = tello.Tello()
drone.connect()
drone.streamon()

while True:
    img = drone.get_frame_read().frame
    img = cv2.resize(img,(360,240))
    cv2.imshow("Feed",img)
    cv2.waitKey(1)