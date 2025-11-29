import math
import os
import threading
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from twilio.rest import Client

from detection import AccidentDetectionModel

load_dotenv()

MODEL_JSON_PATH = os.getenv("ACCIDENT_MODEL_JSON", "model.json")
MODEL_WEIGHTS_PATH = os.getenv("ACCIDENT_MODEL_WEIGHTS", "model_weights.keras")
PROBABILITY_THRESHOLD = float(os.getenv("ACCIDENT_PROB_THRESHOLD", "0.95"))
DISPLAY_TILE_HEIGHT = int(os.getenv("ACCIDENT_TILE_HEIGHT", "360"))
DISPLAY_TILE_WIDTH = int(os.getenv("ACCIDENT_TILE_WIDTH", "520"))

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+17652343207")
EMERGENCY_PHONE_NUMBER = os.getenv("EMERGENCY_PHONE_NUMBER", "+916005971380")

VIDEO_SOURCES = [
    ("camera1.mp4", "Camera 1"),
    ("camera2.mp4", "Camera 2"),
    ("camera3.mp4", "Camera 3"),
    ("camera4.mp4", "Camera 4"),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def save_accident_photo(frame, camera_name):
    """Store the frame that triggered the alarm for later review."""
    try:
        current_date_time = time.strftime("%Y-%m-%d-%H%M%S")
        directory = "accident_photos"
        os.makedirs(directory, exist_ok=True)
        safe_name = "".join(c if c.isalnum() else "_" for c in camera_name)
        filename = os.path.join(directory, f"{safe_name}_{current_date_time}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[{camera_name}] Accident photo saved at {filename}")
    except Exception as exc:
        print(f"[{camera_name}] Error saving accident photo: {exc}")


def call_emergency_services(camera_name):
    """Trigger a Twilio voice call that announces which camera saw the crash."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, EMERGENCY_PHONE_NUMBER]):
        print(f"[{camera_name}] Twilio credentials missing, skipping call.")
        return

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = (
            f"Accident detected on {camera_name}. "
            "Please dispatch emergency services immediately."
        )
        client.calls.create(
            twiml=f"<Response><Say voice='alice'>{message}</Say></Response>",
            to=EMERGENCY_PHONE_NUMBER,
            from_=TWILIO_FROM_NUMBER,
        )
        print(f"[{camera_name}] Emergency services notified through Twilio.")
    except Exception as exc:
        print(f"[{camera_name}] Error while contacting emergency services: {exc}")


def _load_video_captures(video_sources):
    captures = []
    for source, name in video_sources:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[{name}] Unable to open video source: {source}")
            captures.append(None)
            continue
        captures.append(cap)
    return captures


def _build_grid(frames, tile_size=None):
    """Combine frames into a single grid image."""
    tile_size = tile_size or (DISPLAY_TILE_HEIGHT, DISPLAY_TILE_WIDTH)
    if not frames:
        return None

    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)

    tile_h, tile_w = tile_size
    grid_image = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        if frame is None:
            continue
        resized = cv2.resize(frame, (tile_w, tile_h))
        row = idx // cols
        col = idx % cols
        grid_image[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = resized

    return grid_image


def _predict_accident(model, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(rgb_frame, (250, 250))
    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    try:
        probability = float(prob[0][0])
    except Exception:
        probability = float(prob) if prob is not None else 0.0
    return pred, probability


def startapplication(video_sources=None):
    """Launch the multi-camera simulation window."""
    sources = video_sources or VIDEO_SOURCES
    model = AccidentDetectionModel(MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)
    captures = _load_video_captures(sources)
    alarm_state = {name: False for _, name in sources}

    if not any(cap for cap in captures):
        print("No valid video sources available. Exiting simulation.")
        return

    window_name = "Accident Monitoring Wall"

    while True:
        frames_for_grid = []

        for idx, (source, camera_name) in enumerate(sources):
            cap = captures[idx]
            if cap is None:
                blank = np.zeros((DISPLAY_TILE_HEIGHT, DISPLAY_TILE_WIDTH, 3), dtype=np.uint8)
                frames_for_grid.append(blank)
                continue

            ret, frame = cap.read()
            if not ret:
                # Loop demo clips indefinitely
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    blank = np.zeros((DISPLAY_TILE_HEIGHT, DISPLAY_TILE_WIDTH, 3), dtype=np.uint8)
                    frames_for_grid.append(blank)
                    continue

            pred, probability = _predict_accident(model, frame)
            probability_percent = probability * 100
            label = f"{camera_name}: {pred} {probability_percent:.1f}%"
            color = (0, 0, 255) if pred == "Accident" and probability >= PROBABILITY_THRESHOLD else (0, 200, 0)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 30), FONT, 0.8, color, 2, cv2.LINE_AA)

            alarm_ready = probability >= PROBABILITY_THRESHOLD and pred == "Accident"
            if alarm_ready and not alarm_state[camera_name]:
                alarm_state[camera_name] = True
                save_accident_photo(frame, camera_name)
                threading.Thread(
                    target=call_emergency_services, args=(camera_name,), daemon=True
                ).start()
            elif not alarm_ready:
                alarm_state[camera_name] = False

            display_frame = cv2.resize(frame, (DISPLAY_TILE_WIDTH, DISPLAY_TILE_HEIGHT))
            frames_for_grid.append(display_frame)

        grid = _build_grid(frames_for_grid, (DISPLAY_TILE_HEIGHT, DISPLAY_TILE_WIDTH))
        if grid is None:
            break

        cv2.imshow(window_name, grid)
        # Press q to close the monitoring wall
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in captures:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    startapplication()
