# Traffic Light Detection Web Application

A real-time traffic light detection application using YOLO and Flask. The application processes video feed from a camera and detects traffic lights, highlighting them with color-coded bounding boxes.

## Features

- Real-time traffic light detection
- Color-coded detection boxes (red, yellow, green)
- Side-by-side raw and processed video feeds
- Modern, responsive web interface
- Camera permission handling
- Production-ready with Render deployment support

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python src/web_app.py
```

3. Open http://localhost:8000 in your browser

## Deployment on Render

1. Fork/Clone this repository
2. Sign up for a [Render account](https://render.com)
3. Create a new Web Service and connect your repository
4. Use the following configuration:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --chdir src wsgi:app`
   - Python Version: 3.9.0

### Environment Variables

Add these in Render's dashboard:
- `PYTHON_VERSION`: 3.9.0
- `RENDER`: true
- `CAMERA_INDEX`: 0 (only used in local development)

## Notes

- The application requires camera access for local development
- In production (Render), the application will display a test frame as camera access isn't available
- For local development, make sure your camera is properly connected and accessible 