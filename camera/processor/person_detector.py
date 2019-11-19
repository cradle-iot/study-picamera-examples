from __future__ import print_function
from imutils.video.webcamvideostream import WebcamVideoStream
#from imutils.video.pivideostream import PiVideoStream
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2

import os, time, datetime, json, copy, decimal, threading
import boto3

print('starting... model reading...')
net = cv2.dnn.readNetFromCaffe(
        'camera/processor/MobileNetSSD_deploy.prototxt',
        'camera/processor/MobileNetSSD_deploy.caffemodel')
print('read ok.')

class PersonDetector(object):
    def __init__(self, flip = True):
        self.vs = WebcamVideoStream().start()
        #self.vs = PiVideoStream(resolution=(800, 608)).start()
        self.flip = flip
        time.sleep(1.0)
        
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
        frame = imutils.resize(frame, width=500)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        data_list = []
        data = {'device': os.environ['DEVICE'], 'data': {}}
        data['data']['timestamp'] = str(datetime.datetime.now())
        data['data']['person_id'] = 0
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.2:
                continue

            idx = int(detections[0, 0, i, 1])
            if idx != 15:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            label = '{}: {:.2f}%'.format('Person', confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            data['timestamp'] = str(datetime.datetime.now())
            data['data']['x'] = str((endX+startX)/2)
            data['data']['y'] = str((endY+startY)/2)
            print(data)
            data_list.append(copy.deepcopy(data))
            data['data']['person_id'] += 1

        q = threading.Thread(target=insert, args=(data_list,))
        q.start()
        data_list = []

        return frame
    
def insert(items):
    # items = json.dumps(items)
    # items = json.loads(items, parse_float=decimal.Decimal)
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
            print('Succeeded :', item)
    return
