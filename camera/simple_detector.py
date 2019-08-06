from __future__ import print_function
from imutils.video.webcamvideostream import WebcamVideoStream
#from imutils.video.pivideostream import PiVideoStream
from imutils.object_detection import non_max_suppression
import imutils
import time
import numpy as np
import cv2

import os
import datetime
import decimal
import json
import boto3

obj = ["background", "aeroplane", "bicycle", "bird", "boat",
       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
       "sofa", "train", "tvmonitor"] 

print('starting... model reading...')
net = cv2.dnn.readNetFromCaffe('/var/isaax/project/camera/processor/MobileNetSSD_deploy.prototxt',
        '/var/isaax/project/camera/processor/MobileNetSSD_deploy.caffemodel')
print('read ok.')

vs = WebcamVideoStream().start()
time.sleep(2.0)

data_list = []
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    person_id = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
    
        if confidence < 0.2:
            continue
    
        idx = int(detections[0, 0, i, 1])
        if idx != 15:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        label = '{}: {:.2f}%'.format(obj[idx], confidence * 100)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
        data = {}
        data['data'] = {'x': (endX-startX)/2, 'y':(endY-startY)/2}
        data['timestamp'] = datetime.datetime.now().timestamp()
        data['device'] = os.environ['DEVICE']
        data['person_id'] = person_id
        person_id += 1        
        print('id:', person_id, 'x:', data['data']['x'], 'y:', data['data']['y'], 'Time:',datetime.datetime.now())
        
        data_list.append(data)
        
    if len(data_list) > 10:
        print(data_list)
        data_list = []