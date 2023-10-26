import cv2
import numpy as np
import yt_dlp
from flask import Flask, Response
from threading import Thread, Lock
from queue import Queue, Empty
from sort import Sort
import time

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
        
    def detect_and_annotate(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'aeroplane']
        detected_boxes = []

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

                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img, detected_boxes
    
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
        self.vehicle_count = 0
        self.last_time = time.time()
        self.persistent_detection = {}  # Dictionary to track persistent detections
        
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

                trackers = self.tracker.update(np.array(detected_boxes))

                for track in trackers:
                    tid = int(track[4])
                    if tid not in self.persistent_detection:
                        self.persistent_detection[tid] = 1
                    else:
                        self.persistent_detection[tid] += 1

                    # Count car if it has been persistently detected in 3 or more frames
                    if self.persistent_detection[tid] == 3:
                        self.vehicle_count += 1

                if time.time() - self.last_time >= 60:
                    print(f"Moving cars counted in last minute: {self.vehicle_count}")
                    self.vehicle_count = 0
                    self.last_time = time.time()

                with self.lock:
                    if not self.queue.full():
                        self.queue.put(annotated_frame)
                    
app = Flask(__name__)

video_url = get_youtube_video_url("https://www.youtube.com/embed/NRd9c9HSZSk")
tracker = Sort()
frame_lock = Lock()
frame_queue = Queue(maxsize=86400)

# Start the combined capture and processing thread
capture_process_thread = CaptureAndProcessThread(video_url, frame_queue, tracker, frame_lock)
capture_process_thread.start()

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
