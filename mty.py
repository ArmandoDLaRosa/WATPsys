import cv2
import numpy as np
import yt_dlp
from flask import Flask, Response
import time 

def get_youtube_video_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',  # You can adjust the format as needed
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']
       
class YOLOProcessor:
    def __init__(self):
        # TODO try yolo4 tiny
        #  wget https://pjreddie.com/media/files/yolov3.weights
        #  wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
        
        # wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg        
        # wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
        
        # TODO transfer learning so tiny can detect small objects as yolov4 can do. 
        # I'll need to create a data set
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        self.output_layers = [self.layer_names[i[0] - 1] for i in unconnected_layers.reshape(-1, 1)]


        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def detect_vehicles(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return outs

    def annotate_frame(self, img, outs):
        height, width, channels = img.shape
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1: # Setting a threshold
                    label = str(processor.classes[class_id])
                    print(label)                    
                                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        
        vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'aeroplane']
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                if label not in vehicle_classes:
                  continue
                confidence = confidences[i]
                color = (0, 255, 0)
                
                x, y, w, h = boxes[i]
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                #cv2.putText(img, label, (x, y + 30), self.font, 2, color, 3)

        return img


video_url = get_youtube_video_url("https://www.youtube.com/embed/NRd9c9HSZSk")

cap = cv2.VideoCapture(video_url)
processor = YOLOProcessor()
        

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
                continue
            outs = processor.detect_vehicles(frame)
            annotated_frame = processor.annotate_frame(frame, outs)
            cv2.imwrite('test/test_annotated.jpg', annotated_frame) # TODO delete
            
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)