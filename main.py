import os

from djitellopy import tello
from threading import Thread
import cv2 as cv
import cvzone as cvz
import numpy as np
import sys
import keyPressControl as Kp
import time

# keepRecording = True
global img
# # Initialize opencv DNN
net = cv.dnn.readNet("darknet/yolov4-tiny.weights", "darknet/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)

# load class names using th coco dataset
classFile = 'darknet/coco.names'
with open(classFile, "r") as file_object:
    classNames = file_object.read().split('\n')
    print(classNames)
# initialize keyboard
Kp.init()
# initialize drone
uav = tello.Tello()
uav.connect()
print(uav.get_battery())
uav.streamoff()
uav.streamon()
uavFrame = uav.get_frame_read()
viewFrame = Kp.getFrame((640, 480))

# cap = cv.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)


# def record():
#     # create a VideoWrite object, recording to ./video.avi
#     height, width, _ = uavFrame.frame.shape
#     video = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(*'XVID'), 30, (width, height))
#
#     while keepRecording:
#         video.write(uav.frame)
#         time.sleep(1 / 30)
#
#     video.release()


#     outside the functions
# recorder = Thread(target=record)
# recorder.start()
# recorder.join


# def capture():
#     cv.imwrite("picture.png", uavFrame.frame)

def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if Kp.getKey("LEFT"):
        lr = -speed
    elif Kp.getKey("RIGHT"):
        lr = speed
    if Kp.getKey("UP"):
        fb = speed
    elif Kp.getKey("DOWN"):
        fb = -speed
    if Kp.getKey("w"):
        ud = speed
    elif Kp.getKey("s"):
        ud = -speed
    if Kp.getKey("a"):
        yv = -speed
    elif Kp.getKey("d"):
        yv = speed
    # if Kp.getKey("q"):
    #     # uav.land()
    # if Kp.getKey("e"):
    #     # uav.takeoff()
    if Kp.getKey("z"):
        # save image in date format(yyyy-mm-dd-hh-mm-ss.jpg)
        cv.imwrite(f'images/{time.strftime("%Y%m%d%I%M%S")}.jpg', img)
        time.sleep(0.03)
    return lr, fb, ud, yv


def detect_frames(frame, outputs, conf_thres=None, nms_Thres=None):
    ht, wt, _ = frame.shape
    bbox = []
    class_ids = []
    conf_vals = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > conf_thres:
                # To get the pixel value of detection
                w, h = int(detection[2] * wt), int(detection[3] * ht)
                x, y = int((detection[0] * wt) - w / 2), int((detection[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                conf_vals.append(float(conf))
    indices = cv.dnn.NMSBoxes(bbox, conf_vals, conf_thres, nms_Thres)
    print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        class_name = classNames[class_ids[i]].upper()
        cvz.cornerRect(frame, box, t=2, colorR=(31, 255, 15), colorC=(255, 255, 255), l=20)
        cv.putText(frame, f'{class_name} {round(float(conf_vals[i] * 100), 2)}%', (x, y - 15),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   (31, 255, 15), 1)


while True:
    try:
        # ret, img = cap.read()
        vals = get_keyboard_input()
        print(vals[0], vals[1], vals[2], vals[3])
        print(time.ctime())
        print(time.strftime("%Y%m%d%I%M%S"))
        print(time.thread_time())
        uav.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        img = uavFrame.frame
        img = cv.resize(img, (640, 480))
        # ret, img = cap.read()
        blob = cv.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
        model.setInput(blob)
        layerNames = model.getLayerNames()
        # print(layerNames)

        # To get layers output names
        outputNames = [layerNames[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        # print(outputNames)
        # print(model.getUnconnectedOutLayers())
        outputs = model.forward(outputNames)
        # print(len(outputs))
        # print(outputs[0].shape)
        detect_frames(img, outputs, conf_thres=0.4, nms_Thres=0.2)
        # cv.imshow("UAV FEED", img)
        Kp.getDisplay(img, viewFrame)
        cv.waitKey(1)
        if Kp.win_close():
            uav.land()
            break

    except Exception as error:
        print(error)
        break
uav.streamoff()
uav.land()
cv.destroyAllWindows()
sys.exit(0)
