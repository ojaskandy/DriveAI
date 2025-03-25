# DriveAI

A real-time hazard detection system for drivers using computer vision and deep learning.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained YOLO weights for traffic light detection and place them in the `src/models` directory.

3. Run the application:
```bash
python src/main.py
```

## Project Structure

- `src/`: Source code directory
  - `main.py`: Main application file with GUI and camera handling
  - `models/`: Directory for storing pre-trained models and weights
- `requirements.txt`: Python dependencies

## Features

- Real-time camera feed display
- Traffic light detection and color classification
- Split-screen view showing both raw and processed camera feeds 