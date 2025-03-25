import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response
from ultralytics import YOLO
import base64
import os

app = Flask(__name__)

class TrafficLightDetector:
    def __init__(self):
        # Load YOLO model
        model_path = Path("src/models/best_traffic_small_yolo.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(str(model_path))
        
        # Check if we're running on Render
        self.is_render = os.getenv('RENDER', False)
        if self.is_render:
            # In production, use a test image instead of camera
            self.test_mode = True
            self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.test_frame, 
                       "Camera access not available in production", 
                       (50, 240),
                       cv2.FONT_HERSHEY_DUPLEX, 
                       0.8, 
                       (255, 255, 255), 
                       2)
        else:
            # Initialize camera - try different camera indices if 0 doesn't work
            camera_index = int(os.getenv('CAMERA_INDEX', '0'))
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                self.test_mode = True
                self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(self.test_frame, 
                           "No camera detected", 
                           (50, 240),
                           cv2.FONT_HERSHEY_DUPLEX, 
                           1.0, 
                           (255, 255, 255), 
                           2)
            else:
                self.test_mode = False

    def get_color_for_class(self, class_name):
        # Define vibrant colors for each class (in BGR format)
        color_map = {
            'red': (0, 0, 255),      # Bright Red
            'yellow': (0, 255, 255),  # Bright Yellow
            'green': (0, 255, 0),     # Bright Green
            'off': (128, 128, 128)    # Gray
        }
        return color_map.get(class_name, (0, 255, 0))

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame, conf=0.25)
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Process results
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results[0].names[cls].lower()
                
                # Get color based on class
                color = self.get_color_for_class(class_name)
                
                # Draw thicker bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 4)
                
                # Create more visible label background
                label = f"{class_name.upper()}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                cv2.rectangle(processed_frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
                
                # Add larger label with confidence
                cv2.putText(processed_frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                
                # Draw attention-grabbing corner markers
                marker_length = 20
                thickness = 4
                # Top-left
                cv2.line(processed_frame, (x1, y1), (x1 + marker_length, y1), color, thickness)
                cv2.line(processed_frame, (x1, y1), (x1, y1 + marker_length), color, thickness)
                # Top-right
                cv2.line(processed_frame, (x2, y1), (x2 - marker_length, y1), color, thickness)
                cv2.line(processed_frame, (x2, y1), (x2, y1 + marker_length), color, thickness)
                # Bottom-left
                cv2.line(processed_frame, (x1, y2), (x1 + marker_length, y2), color, thickness)
                cv2.line(processed_frame, (x1, y2), (x1, y2 - marker_length), color, thickness)
                # Bottom-right
                cv2.line(processed_frame, (x2, y2), (x2 - marker_length, y2), color, thickness)
                cv2.line(processed_frame, (x2, y2), (x2, y2 - marker_length), color, thickness)
        
        return processed_frame

    def get_frame(self):
        if self.test_mode or self.is_render:
            frame = self.test_frame.copy()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None, None

        # Process frame with YOLO
        processed_frame = self.process_frame(frame)
        
        # Convert frames to JPEG
        _, raw_jpeg = cv2.imencode('.jpg', frame)
        _, processed_jpeg = cv2.imencode('.jpg', processed_frame)
        
        return raw_jpeg.tobytes(), processed_jpeg.tobytes()

    def __del__(self):
        if not self.test_mode and not self.is_render:
            self.cap.release()

detector = TrafficLightDetector()

def gen_frames(raw=True):
    while True:
        raw_frame, processed_frame = detector.get_frame()
        if raw_frame is None or processed_frame is None:
            break
            
        frame = raw_frame if raw else processed_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_raw')
def video_feed_raw():
    return Response(gen_frames(raw=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed')
def video_feed_processed():
    return Response(gen_frames(raw=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 8000))
    # In production, host should be '0.0.0.0'
    host = '0.0.0.0' if os.getenv('RENDER') else 'localhost'
    app.run(host=host, port=port) 