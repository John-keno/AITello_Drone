from djitellopy import tello
from threading import Thread
import cv2 as cv
import cvzone as cvz
import numpy as np
import time

# keepRecording = True

# # Initialize opencv DNN
net = cv.dnn.readNet("darknet/yolov4-tiny.weights", "darknet/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)

# load class names using th coco dataset
classFile = 'darknet/coco.names'
with open(classFile, "r") as file_object:
    classNames = file_object.read().split('\n')
    print(classNames)

# initialize drone
uav = tello.Tello()
uav.connect()
print(uav.get_battery())
uav.streamoff()
uav.streamon()
uavFrame = uav.get_frame_read()
# cap = cv.VideoCapture(0)


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
        cv.putText(frame, f'{class_name} {round(float(conf_vals[i] * 100),2)}%', (x, y - 15), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   (31, 255, 15), 1)


while True:
    try:
        img = uavFrame.frame
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
        cv.imshow("UAV FEED", img)
        uav.send_rc_control(0, 0, 0, 0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as error:
        print(error)
        break

cv.destroyAllWindows()
