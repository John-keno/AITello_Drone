from djitellopy import tello
from threading import Thread
import cv2 as cv
import cvzone as cvz
import numpy as np
import time

keepRecording = True

# Initialize opencv DNN
net = cv.dnn.readNet("path to .weight file", "path to .cfg file")
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# load class names using th coco dataset
classFile = 'darknet/coco.names'
with open(classFile, "r") as file_object:
    classNames = file_object.read().split('\n')
    print(classNames)

# initialize drone
uav = tello.Tello()
uav.connect()
uav.streamoff()
uav.streamon()
uavFrame = uav.get_frame_read()


def record():
    # create a VideoWrite object, recording to ./video.avi
    height, width, _ = uavFrame.frame.shape
    video = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(uav.frame)
        time.sleep(1 / 30)

    video.release()


#     outside the functions
# recorder = Thread(target=record)
# recorder.start()
# recorder.join


def capture():
    cv.imwrite("picture.png", uavFrame.frame)


def detect_frames(frame):
    (class_ids, confScores, bboxes) = model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
    for class_ids, confScores, bboxes in zip(class_ids, confScores, bboxes):
        (x, y, w, h) = bboxes
        class_name = classNames[class_ids].upper()
        cvz.cornerRect(frame, bboxes, t=2, colorR=(31, 255, 15), colorC=(255, 255, 255), l=20)
        cv.putText(frame,
                   f'{class_name} {int(confScores * 100)}%',
                   (x, y - 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   (31, 255, 15), 1)


while True:
    try:
        success, img = uavFrame.frame
        if success:
            cv.imshow("UAV FEED", img)
            uav.send_rc_control(0, 0, 0, 0)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as error:
        print(error)
        break

cv.destroyAllWindows()