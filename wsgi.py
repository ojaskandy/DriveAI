from src.server import app, socketio

# This is needed for gunicorn
application = socketio.middleware(app)

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True) 