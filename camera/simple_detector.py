import cv2
import numpy as np
import imutils
from imutils.video.webcamvideostream import WebcamVideoStream
from __future__ import print_function
import os
import copy
import time
import datetime
import json
import decimal
import threading
import boto3

def insert(items):
    items = json.dumps(items)
    items = json.loads(items, parse_float=decimal.Decimal)
    #session
    session = boto3.session.Session(
                                    region_name = os.environ['REGION'],
                                    aws_access_key_id = os.environ['A_KEY'],
                                    aws_secret_access_key = os.environ['S_KEY'],
                                    )
    dynamodb = session.resource('dynamodb')
    #connect Table
    table_name = os.environ['TABLE_NAME']
    table = dynamodb.Table(table_name)

    for item in items:
        #add
        response = table.put_item(
            TableName=table_name,
            Item=item
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:#Fail
            print(response)
        else:
            print('Successed :', item['device'])
    return

#obj = ["background", "aeroplane", "bicycle", "bird", "boat",
#       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#       "sofa", "train", "tvmonitor"] 

print('starting... model reading...')
net = cv2.dnn.readNetFromCaffe(
        '/var/isaax/project/camera/processor/MobileNetSSD_deploy.prototxt',
        '/var/isaax/project/camera/processor/MobileNetSSD_deploy.caffemodel'
        )
data = {
        'device': os.environ['DEVICE'],
        'data': {}
        }
print('start detecting...')
vs = WebcamVideoStream().start()
time.sleep(2.0)

data_list = []
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=300)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    data['data']['timestamp'] = datetime.datetime.now().timestamp()
    data['data']['person_id'] = 0
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.2:
            continue    
        idx = int(detections[0, 0, i, 1])
        if idx != 15:#15:person
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
    
        data['timestamp'] = datetime.datetime.now().timestamp()
        data['data']['x'] = (endX-startX)/2
        data['data']['y'] = (endY-startY)/2

        print(data)
        data_list.append(copy.deepcopy(data))
        data['data']['person_id'] += 1
        
    if len(data_list) > 10:
        q = threading.Thread(target=insert, args=(data_list,))
        q.start()
        data_list = []