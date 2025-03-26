from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

class TrafficLightDetector:
    def __init__(self):
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        try:
            # List all possible model paths
            possible_paths = [
                Path("src/models/best_traffic_small_yolo.pt"),
                Path("models/best_traffic_small_yolo.pt"),
                Path(os.path.join(os.path.dirname(__file__), "models/best_traffic_small_yolo.pt")),
                Path("/opt/render/project/src/models/best_traffic_small_yolo.pt"),
                Path("/opt/render/project/src/src/models/best_traffic_small_yolo.pt")
            ]
            
            # Log current working directory and PYTHONPATH
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH', 'Not set')}")
            logger.info(f"Directory contents of current location: {os.listdir('.')}")
            
            # Try to find the model file
            model_path = None
            for path in possible_paths:
                logger.info(f"Checking path: {path} (exists: {path.exists()})")
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                error_msg = (
                    f"Model file 'best_traffic_small_yolo.pt' not found. Tried paths:\n"
                    f"{chr(10).join(str(p) for p in possible_paths)}"
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(str(model_path))
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            raise

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
            if self.model is None:
                logger.error("Model not initialized!")
                return frame

            logger.info(f"Processing frame with shape: {frame.shape}")
            
            # Ensure frame is not too large
            max_size = 640
            h, w = frame.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                logger.debug(f"Resized frame to: {frame.shape}")

            # Run YOLO detection
            results = self.model(frame, conf=0.25)
            
            # Create a copy of the frame for visualization
            processed_frame = frame.copy()
            
            # Process results
            if results and len(results) > 0:
                boxes = results[0].boxes
                logger.info(f"Found {len(boxes)} detections")
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = results[0].names[cls].lower()
                    
                    logger.info(f"Detection: {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
                    
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
            logger.error(f"Error in process_frame: {str(e)}", exc_info=True)
            return frame

detector = TrafficLightDetector()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Decode base64 image from client
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        logger.info(f"Received frame with shape: {frame.shape if frame is not None else 'None'}")
        
        if frame is None:
            logger.error("Failed to decode frame")
            emit('error', {'message': 'Failed to decode frame'})
            return
        
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Encode result as base64
        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        result_data = f'data:image/jpeg;base64,{encoded_img}'
        
        logger.info("Frame processed and sent back to client")
        
        # Send back to client
        emit('detection', result_data)
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    # In production, let gunicorn handle the serving
    if os.getenv('RENDER'):
        app.logger.setLevel(logging.INFO)
        # Don't run the server here - let gunicorn do it
    else:
        socketio.run(app, host='0.0.0.0', port=port, debug=True) 