import atexit
import os
import pickle
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import quote_plus, urlencode

import cv2
import mediapipe as mp
import numpy as np
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, render_template, session, url_for
from flask_socketio import SocketIO
from pymongo import MongoClient
from pymongo.errors import PyMongoError

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated.")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
TEMPLATE_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"
MODEL_PATH = REPO_ROOT / "models" / "model.p"

load_dotenv(REPO_ROOT / ".env")

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.secret_key = os.getenv("APP_SECRET_KEY") or os.getenv("AUTH0_SECRET") or "dev-secret-change-me"
app.config["SECRET_KEY"] = app.secret_key
app.config["AUTH0_CALLBACK_URL"] = os.getenv("AUTH0_CALLBACK_URL", "http://127.0.0.1:5000/callback")
app.config["AUTH0_LOGOUT_URL"] = os.getenv("AUTH0_LOGOUT_URL", "http://127.0.0.1:5000/signin")

# Auth0 local development reminder:
# Allowed Callback URLs:
# http://127.0.0.1:5000/callback, http://localhost:5000/callback
# Allowed Logout URLs:
# http://127.0.0.1:5000/signin, http://localhost:5000/signin, http://127.0.0.1:5000, http://localhost:5000
# Allowed Web Origins:
# http://127.0.0.1:5000, http://localhost:5000
os.environ.setdefault("AUTHLIB_INSECURE_TRANSPORT", "1")

socketio = SocketIO(app, async_mode="threading")

issuer_base_url = os.getenv("AUTH0_ISSUER_BASE_URL", "").strip()
auth0_domain = os.getenv("AUTH0_DOMAIN")
if not auth0_domain and issuer_base_url:
    auth0_domain = urlparse(issuer_base_url).netloc or issuer_base_url.replace("https://", "").replace("http://", "")
auth0_client_id = os.getenv("AUTH0_CLIENT_ID")
auth0_client_secret = os.getenv("AUTH0_CLIENT_SECRET")

oauth = OAuth(app)
auth0 = None
if auth0_domain and auth0_client_id and auth0_client_secret:
    auth0 = oauth.register(
        "auth0",
        client_id=auth0_client_id,
        client_secret=auth0_client_secret,
        client_kwargs={"scope": "openid profile email"},
        server_metadata_url=f"https://{auth0_domain}/.well-known/openid-configuration",
    )
else:
    print("Warning: Auth0 environment variables are incomplete. Login routes will be unavailable.")

mongo_predictions = None
last_saved_prediction = None
last_saved_time = 0.0
prediction_lock = threading.Lock()

mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db = os.getenv("MONGODB_DB", "signlingo_db")

if mongodb_uri:
    try:
        mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
        mongo_client.admin.command("ping")
        mongo_predictions = mongo_client[mongodb_db]["predictions"]
        print(f"MongoDB connected: {mongodb_db}.predictions")
    except Exception as exc:
        print(f"Warning: MongoDB unavailable, prediction logging disabled. {exc}")
else:
    print("Warning: MONGODB_URI not set, prediction logging disabled.")

try:
    with MODEL_PATH.open("rb") as model_file:
        model_dict = pickle.load(model_file)
    model = model_dict["model"]
except Exception as exc:
    print(f"Error loading the model: {exc}")
    model = None

LABELS_DICT = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
    19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: "Hello",
    27: "Done", 28: "Thank You", 29: "I Love you", 30: "Sorry", 31: "Please",
    32: "You are welcome.",
}

STREAM_WIDTH = 640
STREAM_HEIGHT = 480
JPEG_QUALITY = 70
PREDICT_EVERY_N_FRAMES = 2
SOCKET_EMIT_INTERVAL_SECONDS = 0.75
FPS_LOG_EVERY_FRAMES = 60
CAPTURE_FPS = 60
HANDS_PROCESS_SCALE = 0.75

CONFIDENCE_THRESHOLD = 0.60
VIDEO_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
camera_stream = None
camera_stream_lock = threading.Lock()


def save_prediction_if_needed(predicted_label, confidence):
    global last_saved_prediction, last_saved_time

    if mongo_predictions is None or not predicted_label or predicted_label == "Uncertain":
        return

    now = time.monotonic()

    with prediction_lock:
        should_save = predicted_label != last_saved_prediction or now - last_saved_time >= 2.0
        if not should_save:
            return

        try:
            mongo_predictions.insert_one(
                {
                    "detected_sign": predicted_label,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow(),
                    "source": "live_camera",
                }
            )
            last_saved_prediction = predicted_label
            last_saved_time = now
        except PyMongoError as exc:
            print(f"Warning: Failed to save prediction to MongoDB. {exc}")


def serialize_prediction(document):
    return {
        "_id": str(document.get("_id", "")),
        "detected_sign": document.get("detected_sign", ""),
        "confidence": document.get("confidence", 0.0),
        "timestamp": str(document.get("timestamp", "")),
        "source": document.get("source", ""),
    }


def create_fallback_frame(line1, line2):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        line1,
        (50, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line2,
        (50, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return frame


FINGER_LANDMARK_COLORS = {
    0: (70, 70, 225),
    1: (255, 120, 40),
    2: (255, 120, 40),
    3: (255, 120, 40),
    4: (255, 120, 40),
    5: (70, 220, 70),
    6: (70, 220, 70),
    7: (70, 220, 70),
    8: (70, 220, 70),
    9: (0, 230, 255),
    10: (0, 230, 255),
    11: (0, 230, 255),
    12: (0, 230, 255),
    13: (220, 90, 255),
    14: (220, 90, 255),
    15: (220, 90, 255),
    16: (220, 90, 255),
    17: (0, 140, 255),
    18: (0, 140, 255),
    19: (0, 140, 255),
    20: (0, 140, 255),
}


def draw_hand_landmarks_colored(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = []

    for index, landmark in enumerate(hand_landmarks.landmark):
        x_pos = int(landmark.x * width)
        y_pos = int(landmark.y * height)
        points.append((x_pos, y_pos))

        color = FINGER_LANDMARK_COLORS.get(index, (200, 200, 200))
        radius = 8 if index in {4, 8, 12, 16, 20} else 6
        cv2.circle(frame, (x_pos, y_pos), radius + 3, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(frame, (x_pos, y_pos), radius, color, -1, cv2.LINE_AA)

    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        start_point = points[start_idx]
        end_point = points[end_idx]
        color = FINGER_LANDMARK_COLORS.get(end_idx, (160, 160, 160))
        cv2.line(frame, start_point, end_point, color, 3, cv2.LINE_AA)


class CameraStream:
    def __init__(self, src=0):
        self._src = src
        self.cap = None
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._read_failed_logged = False
        self._open()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _open(self):
        print("Opening webcam...")
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self._src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("ERROR: Could not open webcam")
            self.cap.release()
            self.cap = None
            return

        print("Webcam opened successfully")

    def _update(self):
        while not self._stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._open()
                continue

            if not self.cap.grab():
                ret, raw = False, None
            else:
                ret, raw = self.cap.retrieve()
            if not ret or raw is None:
                if not self._read_failed_logged:
                    print("Frame read failed")
                    self._read_failed_logged = True
                time.sleep(0.02)
                continue

            self._read_failed_logged = False
            with self._lock:
                self.frame = raw

    def read(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._stopped = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def get_camera_stream():
    global camera_stream

    with camera_stream_lock:
        if camera_stream is None or camera_stream._stopped:
            camera_stream = CameraStream(0)
        return camera_stream


def stop_camera_stream():
    global camera_stream

    with camera_stream_lock:
        if camera_stream is not None:
            camera_stream.stop()
            camera_stream = None


atexit.register(stop_camera_stream)


@app.route("/signin")
def signin():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("signin"))
    return render_template("index.html", user=session["user"])


@app.route("/login")
def login():
    if auth0 is None:
        return redirect(url_for("signin"))
    return auth0.authorize_redirect(redirect_uri=app.config["AUTH0_CALLBACK_URL"])


@app.route("/signup")
def signup():
    if auth0 is None:
        return redirect(url_for("signin"))
    return auth0.authorize_redirect(
        redirect_uri=app.config["AUTH0_CALLBACK_URL"],
        screen_hint="signup",
    )


@app.route("/callback")
def callback():
    if auth0 is None:
        return redirect(url_for("signin"))

    token = auth0.authorize_access_token()
    user_info = token.get("userinfo")
    if not user_info:
        user_info = auth0.get("userinfo").json()

    session["user"] = user_info
    return redirect(url_for("index"))


@app.route("/camera_test")
def camera_test():
    test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    opened = test_cap.isOpened()
    if opened:
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        test_cap.set(cv2.CAP_PROP_FPS, 30)
        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = test_cap.read() if opened else (False, None)
    test_cap.release()

    return {
        "camera_opened": bool(opened),
        "frame_read": bool(ret),
        "frame_shape": list(frame.shape) if frame is not None else None,
    }


@app.route("/logout")
def logout():
    session.clear()
    if not auth0_domain or not auth0_client_id:
        return redirect(url_for("signin"))

    params = {
        "returnTo": app.config["AUTH0_LOGOUT_URL"],
        "client_id": auth0_client_id,
    }
    return redirect(f"https://{auth0_domain}/v2/logout?{urlencode(params, quote_via=quote_plus)}")


@app.route("/api/predictions/recent")
def recent_predictions():
    if mongo_predictions is None:
        return jsonify([])

    try:
        documents = list(
            mongo_predictions.find({}, {"detected_sign": 1, "confidence": 1, "timestamp": 1, "source": 1})
            .sort("timestamp", -1)
            .limit(20)
        )
        return jsonify([serialize_prediction(document) for document in documents])
    except PyMongoError as exc:
        print(f"Warning: Failed to read recent predictions from MongoDB. {exc}")
        return jsonify([])


@socketio.on("connect")
def handle_connect():
    print("Client connected")


def generate_frames():
    print("Opening webcam for /video_feed...")
    stream = get_camera_stream()
    if stream.cap is None or not stream.cap.isOpened():
        print("ERROR: Could not open webcam in /video_feed")
        return

    print("Webcam opened successfully for /video_feed")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_count = 0
    last_label = ""
    last_confidence = 0.0
    last_box = None
    last_box_color = (0, 220, 0)
    streak_label = ""
    streak_count = 0
    last_emitted_label = None
    last_emitted_confidence = 0.0
    last_emit_time = 0.0
    fps_window_started_at = time.monotonic()
    missing_frame_logged = False

    try:
        while True:
            frame = stream.read()
            if frame is None:
                if not missing_frame_logged:
                    print("Frame read pending in /video_feed")
                    missing_frame_logged = True
                time.sleep(0.01)
                continue
            missing_frame_logged = False

            frame = cv2.flip(frame, 1)
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if HANDS_PROCESS_SCALE != 1.0:
                frame_rgb_for_hands = cv2.resize(
                    frame_rgb,
                    None,
                    fx=HANDS_PROCESS_SCALE,
                    fy=HANDS_PROCESS_SCALE,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_rgb_for_hands = frame_rgb

            results = hands.process(frame_rgb_for_hands)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                draw_hand_landmarks_colored(frame, hand_landmarks)

                xs = [landmark.x for landmark in hand_landmarks.landmark]
                ys = [landmark.y for landmark in hand_landmarks.landmark]
                height, width = frame.shape[:2]

                x1 = max(0, int(min(xs) * width) - 20)
                y1 = max(0, int(min(ys) * height) - 20)
                x2 = min(width, int(max(xs) * width) + 20)
                y2 = min(height, int(max(ys) * height) + 20)

                last_box = (x1, y1, x2, y2)

                should_predict = model is not None and frame_count % PREDICT_EVERY_N_FRAMES == 0
                if should_predict:
                    try:
                        feature_vector = []
                        min_x = min(xs)
                        min_y = min(ys)
                        for landmark in hand_landmarks.landmark:
                            feature_vector.append(landmark.x - min_x)
                            feature_vector.append(landmark.y - min_y)

                        vector = np.asarray(feature_vector).reshape(1, -1)
                        prediction = model.predict(vector)

                        confidence = 1.0
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(vector)
                            confidence = float(np.max(probabilities[0]))

                        predicted_character = LABELS_DICT.get(int(prediction[0]), "Unknown")
                        current_label = predicted_character if confidence >= CONFIDENCE_THRESHOLD else "Uncertain"

                        if current_label == streak_label:
                            streak_count += 1
                        else:
                            streak_label = current_label
                            streak_count = 1

                        last_box_color = (0, 220, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 185, 255)

                        should_commit = confidence >= 0.85 or streak_count >= 3
                        if should_commit or not last_label:
                            last_label = current_label
                            last_confidence = confidence

                        if should_commit and current_label != "Uncertain":
                            save_prediction_if_needed(predicted_character, confidence)

                        now = time.monotonic()
                        should_emit = should_commit and (
                            current_label != last_emitted_label
                            or now - last_emit_time >= SOCKET_EMIT_INTERVAL_SECONDS
                        )
                        if should_emit:
                            emitted_text = "" if current_label == "Uncertain" else current_label
                            socketio.emit("prediction", {"text": emitted_text, "confidence": confidence})
                            last_emitted_label = current_label
                            last_emitted_confidence = confidence
                            last_emit_time = now
                    except Exception as exc:
                        print(f"Prediction error in /video_feed: {exc}")
            else:
                last_box = None
                streak_label = ""
                streak_count = 0

            cv2.putText(
                frame,
                "Camera Online",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if last_box is not None:
                x1, y1, x2, y2 = last_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3, cv2.LINE_AA)
                display_label = "" if not last_label or last_label == "Uncertain" else last_label
                label_color = (0, 220, 0) if last_label and last_label != "Uncertain" else (0, 185, 255)
                if display_label:
                    overlay_text = f"{display_label} ({last_confidence * 100:.0f}%)"
                    text_origin = (x1, max(40, y1 - 10))
                    cv2.putText(
                        frame,
                        overlay_text,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 0, 0),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        overlay_text,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        label_color,
                        2,
                        cv2.LINE_AA,
                    )

            ok, buffer = cv2.imencode(".jpg", frame, VIDEO_ENCODE_PARAMS)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            if frame_count % FPS_LOG_EVERY_FRAMES == 0:
                now = time.monotonic()
                elapsed = max(now - fps_window_started_at, 1e-6)
                approx_fps = FPS_LOG_EVERY_FRAMES / elapsed
                print(f"Stream FPS: {approx_fps:.1f} | Prediction: {last_label or 'None'}")
                fps_window_started_at = now

    finally:
        hands.close()


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(app.url_map)
    socketio.run(app, host=host, port=port, debug=True, use_reloader=False)
