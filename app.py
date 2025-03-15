"""
This is the main script responsible for starting the flask server and providing the API endpoints
with HTTPS support for secure WebSocket connections.
"""

import os
import ssl
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

from ultralytics import YOLO

from utils import get_comm_cards, get_n_cards, process_raw_image

MODEL_PATH = "./models/best_60_23.pt"

# Certificate paths - update these with your actual certificate paths
CERT_FILE = os.environ.get('CERT_FILE', './certificates/cert.pem')
KEY_FILE = os.environ.get('KEY_FILE', './certificates/key.pem')

# Init Flask app and socketIO
app = Flask(__name__)
# Remove SSL config from SocketIO init - we'll apply it only in socketio.run()
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet'  # use eventlet for better websocket support
)
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

# Generate a self-signed certificate for development (run this once)
def generate_self_signed_cert():
    """Generate a self-signed certificate for development purposes."""
    from OpenSSL import crypto

    # Create a directory for certificates if it doesn't exist
    os.makedirs(os.path.dirname(CERT_FILE), exist_ok=True)

    # Only generate if the certificate doesn't already exist
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        # Create a key pair
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)

        # Create a self-signed cert
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "State"
        cert.get_subject().L = "City"
        cert.get_subject().O = "Organization"
        cert.get_subject().OU = "Organizational Unit"
        # Set the Common Name to match your IP address or domain
        cert.get_subject().CN = "192.168.178.112"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, 'sha256')

        # Write certificate and key to files
        with open(CERT_FILE, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(KEY_FILE, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

        print(f"Self-signed certificate generated at {CERT_FILE} and {KEY_FILE}")

# Run the app
if __name__ == '__main__':
    # Check if certificates exist, or generate for development
    if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
        try:
            generate_self_signed_cert()
        except ImportError:
            print("PyOpenSSL not installed. Install with: pip install pyopenssl")
            print("Alternatively, provide your own certificates using environment variables:")
            print("CERT_FILE=/path/to/cert.pem KEY_FILE=/path/to/key.pem python app.py")
            exit(1)

    # Start the socketio server with SSL
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5001,
    )