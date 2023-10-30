import cv2
import numpy as np
import yt_dlp
from flask import Flask, Response
from threading import Thread, Lock
from queue import Queue, Empty
from sort import Sort
import time
import math
import collections.abc

class YOLOProcessor:
    def __init__(self):
        # TODO try yolo4 tiny
        #  wget https://pjreddie.com/media/files/yolov3.weights
        #  wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
        
        # wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg        
        # wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
        
        # TODO transfer learning so tiny can detect small objects as yolov4 can do. 
        # I'll need to create a data set
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        self.output_layers = [self.layer_names[i[0] - 1] for i in unconnected_layers.reshape(-1, 1)]


        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def nms_boxes(self, boxes, scores, threshold):
        return cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.1, nms_threshold=threshold)
        
    def detect_and_annotate(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'aeroplane']
        detected_boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    label = self.classes[class_id]
                    if label not in vehicle_classes:
                        continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = x1 + w
                    y2 = y1 + h

                    detected_boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = self.nms_boxes(detected_boxes, confidences, 0.4)
        final_indices = []  

        for index in indices:
            if isinstance(index, collections.abc.Iterable):
                i = index[0]
            else:
                i = index

            x1, y1, x2, y2 = detected_boxes[i]
            label = self.classes[class_ids[i]]
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            final_indices.append(i) 

        return img, [detected_boxes[i] for i in final_indices]

    
def get_youtube_video_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

class CaptureAndProcessThread(Thread):

    def __init__(self, video_url, queue, tracker, frame_lock):
        super().__init__()
        self.cap = cv2.VideoCapture(video_url)
        self.queue = queue
        self.yolo_processor = YOLOProcessor()
        self.tracker = tracker
        self.lock = frame_lock
        self.current_vehicle_ids = set()
        self.vehicles_that_crossed = 0
        self.persistent_detection = {}  
        self.frame_width = int(self.cap.get(3))  
        self.detection_gone_threshold = 3 
        self.last_time = time.time()

    def run(self):
        frame_skip = 2
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            frame_count += 1

            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (640, 480))
                annotated_frame, detected_boxes = self.yolo_processor.detect_and_annotate(frame)

                current_detected_ids = []
                if detected_boxes:
                    trackers = self.tracker.update(np.array(detected_boxes))
                    for track in trackers:
                        tid = int(track[4])
                        current_detected_ids.append(tid)

                        if tid not in self.persistent_detection:
                            self.persistent_detection[tid] = {
                                'bbox': track[:4], 
                                'frames_since_detected': 0
                            }
                        else:
                            self.persistent_detection[tid]['bbox'] = track[:4]
                            self.persistent_detection[tid]['frames_since_detected'] = 0

                self.current_vehicle_ids.update(current_detected_ids)

                for tid, data in list(self.persistent_detection.items()):
                    if tid not in current_detected_ids:
                        data['frames_since_detected'] += 1
                        x1, y1, x2, y2 = data['bbox']

                        touching_border = x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= frame.shape[0]

                        if touching_border and data['frames_since_detected'] >= self.detection_gone_threshold:
                            self.vehicles_that_crossed += 1
                            del self.persistent_detection[tid]

                if time.time() - self.last_time >= 60:
                    print(f"Cars that crossed in the last minute: {self.vehicles_that_crossed}")
                    print(f"Cars present during the last minute: {len(self.current_vehicle_ids)}")
                    
                    self.vehicles_that_crossed = 0
                    self.current_vehicle_ids.clear()
                    self.last_time = time.time()

                with self.lock:
                    if not self.queue.full():
                        self.queue.put(annotated_frame)

app = Flask(__name__)

video_url = get_youtube_video_url("https://www.youtube.com/watch?v=6dp-bvQ7RWo")
tracker = Sort()
frame_lock = Lock()
frame_queue = Queue(maxsize=86400)

capture_process_thread = CaptureAndProcessThread(video_url, frame_queue, tracker, frame_lock)
capture_process_thread.start()

# TODO IMPLEMENT 
# Implement a database
# Implement a newwer YOLO
# WAIT UNTIL NEXT FRAME don't reset the video
# Vehicular Flux Per Minute: This represents the number of cars that completely crossed the field of view in the video feed within a minute. It counts the vehicles that entered the frame from one side and exited from the other side within that minute.
# Cars Present in a Given Minute: This represents the number of unique cars that were detected in the frame during a particular minute, regardless of whether they crossed completely or just partially appeared in the frame.
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            except Empty:
                print("empty")
                continue

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
