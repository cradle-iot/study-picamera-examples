from __future__ import print_function
from imutils.video.webcamvideostream import WebcamVideoStream
from imutils.object_detection import non_max_suppression
import imutils
import time
import numpy as np
import cv2

from datetime import datetime
import os
import sys
import requests
import json

print('model reading')
net = cv2.dnn.readNetFromCaffe('/var/isaax/project/camera/processor/MobileNetSSD_deploy.prototxt',
        '/var/isaax/project/camera/processor/MobileNetSSD_deploy.caffemodel')
print('read ok')

class PersonDetector(object):
    def __init__(self, flip = True):
        self.vs = WebcamVideoStream().start()
        self.flip = flip
        time.sleep(2.0)
        
    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        frame = self.process_image(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def process_image(self, frame):
        frame = imutils.resize(frame, width=300)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.2:
                continue

            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            label = '{}: {:.2f}%'.format(obj[idx], confidence * 100)#('Person', confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count_list[idx] += 1
        
        data = {}
        for i in range(21):
            
            if count_list[i] > 0:
                print('Count_{}: {}'.format(obj[i], count_list[i]))
            
            data[obj[i]] = count_list[i]
            data['date'] = datetime.now().strftime('%Y%m%d%H%M%S')
        
        if sum(count_list) > 0:
            http_post(data)
        
        return frame
    
obj = ["background", "aeroplane", "bicycle", "bird", "boat",
       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
       "sofa", "train", "tvmonitor"]

def http_post(dic):
    post_url = os.environ['POST_URL']
    r = requests.post(post_url, data=json.dumps(dic))
    print(r)
    
