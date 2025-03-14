"""
This is the main script responsible for starting the flask server and providing the API endpoints
"""

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

from ultralytics import YOLO

from utils import get_comm_cards, get_n_cards, process_raw_image

MODEL_PATH = "./models/best_60_23.pt"

# Init Flask app and socketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') #use eventlet for better websocket support
CORS(app)

# Init AI model
model = YOLO(MODEL_PATH)


# --- Endpoints ---
@socketio.on("connect")
def handle_connect():
    print('Client connected')

@socketio.on("disconnect")
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    """
    Gets a frame from one of the users and checks if cards are detected
    :param data: dict with keys "n" for amount of cards "needed" and "image" with the raw image data
    :return: dict with predictions and boolean value saying if amount of cards were found
    """
    num_cards = data["n"]
    image_raw = data["image"]
    image = process_raw_image(image_raw)
    preds = get_n_cards(model, image, num_cards)

    return {
        'predictions': preds,
        'found': len(preds) >= num_cards,
    }

@socketio.on('comm_cards')
def handle_comm_cards(data):
    """
    gets the community cards from the webcam and returns them
    :param data: dict with key n
    :return: dict with predictions and boolean value saying if amount of cards were found
    """
    num_cards = data["n"]

    comm_cards = get_comm_cards(model, num_cards)

    return {
        'predictions': comm_cards,
        'found': len(comm_cards) >= num_cards,
    }

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)