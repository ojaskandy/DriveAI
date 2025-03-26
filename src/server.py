from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
from pathlib import Path

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class TrafficLightDetector:
    def __init__(self):
        # Try different possible model paths
        possible_paths = [
            Path("src/models/best_traffic_small_yolo.pt"),
            Path("models/best_traffic_small_yolo.pt"),
            Path(os.path.join(os.path.dirname(__file__), "models/best_traffic_small_yolo.pt"))
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            error_msg = (
                f"Model file 'best_traffic_small_yolo.pt' not found. Tried paths:\n"
                f"{chr(10).join(str(p) for p in possible_paths)}\n"
                f"Current working directory: {os.getcwd()}"
            )
            raise FileNotFoundError(error_msg)
            
        print(f"Loading model from: {model_path}")
        self.model = YOLO(str(model_path))

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
        try:
            # Ensure frame is not too large
            max_size = 640
            h, w = frame.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

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
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            raise

detector = TrafficLightDetector()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Decode base64 image from client
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Encode result as base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        result_data = f'data:image/jpeg;base64,{encoded_img}'
        
        # Send back to client
        emit('detection', result_data)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True) 