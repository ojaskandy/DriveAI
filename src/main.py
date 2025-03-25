import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

class DriveAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DriveAI - Traffic Light Detection")
        self.setGeometry(100, 100, 1280, 720)

        # Load YOLO model
        model_path = Path("src/models/best_traffic_small_yolo.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(str(model_path))
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create labels for displaying video feeds
        self.raw_feed_label = QLabel()
        self.processed_feed_label = QLabel()
        
        # Set fixed size for labels
        self.raw_feed_label.setFixedSize(640, 480)
        self.processed_feed_label.setFixedSize(640, 480)
        
        # Add labels to layout
        layout.addWidget(self.raw_feed_label)
        layout.addWidget(self.processed_feed_label)

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")

        # Set up timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (approximately 33 fps)

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame, conf=0.25)  # Lower confidence threshold for better detection
        
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
                
                # Draw bounding box
                color = (0, 255, 0)  # Default green color
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{results[0].names[cls]}: {conf:.2f}"
                cv2.putText(processed_frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return processed_frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame with YOLO
            processed_frame = self.process_frame(frame)
            
            # Convert frames to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Convert frames to QImage for display
            h, w, ch = rgb_frame.shape
            raw_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            processed_img = QImage(processed_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            
            # Scale images to fit labels
            raw_pixmap = QPixmap.fromImage(raw_img).scaled(
                self.raw_feed_label.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
            processed_pixmap = QPixmap.fromImage(processed_img).scaled(
                self.processed_feed_label.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
            
            # Update labels
            self.raw_feed_label.setPixmap(raw_pixmap)
            self.processed_feed_label.setPixmap(processed_pixmap)

    def closeEvent(self, event):
        self.cap.release()

def main():
    app = QApplication(sys.argv)
    window = DriveAI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 