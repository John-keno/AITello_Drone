from djitellopy import tello
from threading import Thread
import cv2 as cv
import cvzone as cvz
import numpy as np
import sys
import KeyControlWindow as Win
import time

# ------------------------initialization-----------------------------------

keepRecording = True
global img
global state
# # Initialize Darknet DNN
net = cv.dnn.readNet("darknet/yolov4-tiny.weights", "darknet/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)

# load class names using th coco dataset
classFile = 'darknet/coco.names'
with open(classFile, "r") as file_object:
    classNames = file_object.read().split('\n')
    print(classNames)

# initialize keyboard
Win.init()

# initialize drone
uav = tello.Tello()
uav.connect()
print(uav.get_battery())
uav.streamoff()
uav.streamon()
uav.get_video_capture()
uavFrame = uav.get_frame_read()
viewFrame = Win.getFrame((640, 480), '  TELLO AI Live Stream')


def uav_state():
    global state
    battery = uav.get_battery()
    temp = uav.get_temperature()
    flight_time = uav.get_flight_time()
    state = (battery, temp, flight_time)


def record():
    # create a VideoWrite object, recording to ./video.avi
    height, width, _ = uavFrame.frame.shape
    video = cv.VideoWriter(f'TELLO AI Videos/{time.strftime("VID%Y%m%d%I%M%S")}.avi',
                           cv.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(uavFrame.frame)
        time.sleep(1 / 30)

    video.release()
    stream.join()


def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if Win.getKey("LEFT"):
        lr = -speed
    elif Win.getKey("RIGHT"):
        lr = speed
    if Win.getKey("UP"):
        fb = speed
    elif Win.getKey("DOWN"):
        fb = -speed
    if Win.getKey("w"):
        ud = speed
    elif Win.getKey("s"):
        ud = -speed
    if Win.getKey("q"):
        yv = -speed
    elif Win.getKey("e"):
        yv = speed
    if Win.getKey("BACKSPACE"):
        uav.land()
        time.sleep(0.03)
    if Win.getKey("RETURN"):
        uav.takeoff()
    if Win.getKey("SPACE"):
        # save image in date format(yyyy-mm-dd-hh-mm-ss.jpg)
        cv.imwrite(f'TELLO AI Captures/{time.strftime("%Y%m%d%I%M%S")}.jpg', img)
        time.sleep(0.03)
    return lr, fb, ud, yv


def detect_frames(frame_img, outputs, conf_thres=None, nms_Thres=None):
    ht, wt, _ = frame_img.shape
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
        cvz.cornerRect(frame_img, box, t=2, colorR=(31, 255, 15), colorC=(255, 255, 255), l=20)
        cv.putText(frame_img, f'{class_name} {round(float(conf_vals[i] * 100), 2)}%', (x, y - 10),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   (31, 255, 15), 2)


def set_input_detect(frame_img=None):
    # convert UAV video to blob format
    blob = cv.dnn.blobFromImage(frame_img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    # get all layers name of Darknet DNN
    layerNames = model.getLayerNames()
    # print(layerNames)
    # To get output layers names
    outputNames = [layerNames[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    # print(outputNames)
    # forward outputs of Darknet DNN
    output = model.forward(outputNames)
    # print(len(outputs))
    # print(outputs[0].shape)
    return output


# ---------------------------main function-----------------------------

# Start recording thread
stream = Thread(target=record)
disp_stat = Thread(target=uav_state)
if keepRecording:
    stream.start()

disp_stat.start()

while True:
    try:
        # get video stream from UAV
        img = uavFrame.frame

        # get keyboard input
        vals = get_keyboard_input()

        # resize to 640px by 480px
        img = cv.resize(img, (640, 480))

        # process video frames for DNN
        outputs = set_input_detect(img)

        # get Detections from Darknet and draw bounding box(s)
        detect_frames(img, outputs, conf_thres=0.4, nms_Thres=0.2)

        # send commands to UAV
        uav.send_rc_control(vals[0], vals[1], vals[2], vals[3])

        # display frames
        Win.getDisplay(img, viewFrame, True, state)
        cv.waitKey(1)
        # check if windows close button is clicked
        if Win.win_close():
            if keepRecording:
                keepRecording = False
                stream.join()
            else:
                pass

            disp_stat.join()
            uav.end()
            cv.destroyAllWindows()
            break
    except KeyboardInterrupt:
        break

if keepRecording:
    keepRecording = False
    stream.join()
else:
    pass

disp_stat.join()
uav.end()
cv.destroyAllWindows()
sys.exit(0)
