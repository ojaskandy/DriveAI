import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import base64
import os
import re

app = Flask(__name__)

class TrafficLightDetector:
    def __init__(self):
        # Load YOLO model
        model_path = Path("src/models/best_traffic_small_yolo.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
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

detector = TrafficLightDetector()

@app.route('/')
def index():
    return render_template('index.html')

def decode_base64_image(base64_string):
    # Extract the base64 encoded binary data from the image data URL
    image_data = re.sub('^data:image/.+;base64,', '', base64_string)
    # Decode base64 string
    image_bytes = base64.b64decode(image_data)
    # Convert to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def encode_frame_to_base64(frame):
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert to base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    # Return as data URL
    return f'data:image/jpeg;base64,{base64_string}'

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the frame data from the request
        data = request.get_json()
        frame = decode_base64_image(data['image'])
        
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Convert processed frame to base64
        processed_image = encode_frame_to_base64(processed_frame)
        
        return jsonify({'image': processed_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.getenv('PORT', 8000))
    # In production, host should be '0.0.0.0'
    host = '0.0.0.0'
    app.run(host=host, port=port) 