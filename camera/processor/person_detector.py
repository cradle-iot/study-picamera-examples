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

print('starting... model reading...')
net = cv2.dnn.readNetFromCaffe('/var/isaax/project/camera/processor/MobileNetSSD_deploy.prototxt',
        '/var/isaax/project/camera/processor/MobileNetSSD_deploy.caffemodel')
print('read ok.')

class PersonDetector(object):
    def __init__(self, flip = True):
        self.vs = WebcamVideoStream().start()
        #self.vs = PiVideoStream(resolution=(800, 608)).start()
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
        frame = imutils.resize(frame, width=640)
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
            if idx != 15:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            label = '{}: {:.2f}%'.format(obj[idx], confidence * 100)#('Person', confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count_list[idx] += 1
            
            
        
            data = {}
            data['data'] = {'x': (endX-startX)/2, 'y':(endY-startY)/2}
            data['timestamp'] = datetime.datetime.now().timestamp()
            data['device'] = os.environ['DEVICE']

            for i in range(21):
                if count_list[i] > 0 and i == 15:
                    print('Count_{}: {}'.format(obj[i], count_list[i]))
                    data['data'][obj[i]] = count_list[i]
                
            data = json.dumps(data)
            data = json.loads(data, parse_float=decimal.Decimal)
            
            items = [data]
            if sum(count_list) > 0:
                insert(items)
                    
        return frame
    
obj = ["background", "aeroplane", "bicycle", "bird", "boat",
       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
       "sofa", "train", "tvmonitor"] 

def insert(items):
    # データベース接続の初期化
    session = boto3.session.Session(
                                    region_name = os.environ['REGION'],
                                    aws_access_key_id = os.environ['A_KEY'],
                                    aws_secret_access_key = os.environ['S_KEY'],
                                    )
    dynamodb = session.resource('dynamodb')


    # テーブルと接続
    table_name = os.environ['TABLE_NAME']
    table = dynamodb.Table(table_name)

    for item in items:
        # 追加する
        response = table.put_item(
            TableName=table_name,
            Item=item
        )
        if response['ResponseMetadata']['HTTPStatusCode'] is not 200:
            # 失敗処理
            print(response)
        else:
            # 成功処理
            print('Successed :', item['device'])
    return
