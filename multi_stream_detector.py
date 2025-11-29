import cv2
import numpy as np
import os
import winsound
import threading
import time
import tkinter as tk
from twilio.rest import Client
from PIL import Image, ImageTk
from dotenv import load_dotenv

# your detection model import (unchanged)
from detection import AccidentDetectionModel

load_dotenv()

# Twilio Secrets
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")

# Global control for clean shutdown
stop_event = threading.Event()


def save_accident_photo(frame, source_name):
    try:
        current_date_time = time.strftime("%Y-%m-%d-%H%M%S")
        directory = "accident_photos"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # include source name to distinguish streams
        safe_name = "".join(c if c.isalnum() else "_" for c in source_name)
        filename = f"{directory}/{safe_name}_{current_date_time}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[{source_name}] Accident photo saved as {filename}")
    except Exception as e:
        print(f"[{source_name}] Error saving accident photo: {e}")

def call_ambulance(source_name):
    try:
        account_sid = TWILIO_ACCOUNT_SID
        auth_token = TWILIO_AUTH_TOKEN
        client = Client(account_sid, auth_token)
        call = client.calls.create(
            url="https://handler.twilio.com/twiml/EH7dd72f68b969250a748d2c9e8b503a1c",
            to="+91 6005971380",  # add verified ambulance number
            from_="+1 765 234 3207"
        )
        print(f"[{source_name}] Call SID: {call.sid}")
    except Exception as e:
        print(f"[{source_name}] Error calling ambulance: {e}")

def show_alert_message(source_name):
    # Play beep
    try:
        frequency = 2500
        duration = 2000
        winsound.Beep(frequency, duration)
    except Exception:
        pass

    # Note: creating a Tk() instance inside a thread can work on many systems,
    # but tkinter is not fully thread-safe across all platforms.
    alert_window = tk.Tk()
    alert_window.title(f"Alert - {source_name}")
    alert_window.geometry("400x220")
    alert_label = tk.Label(alert_window, text=f"Alert: Accident detected on {source_name}\nIs the Accident Critical?",
                           fg="black", font=("Helvetica", 14))
    alert_label.pack(pady=10)

    # GIF path optional
    gif_path = ""  # put GIF path if available
    if gif_path:
        try:
            gif = Image.open(gif_path)
            resized_gif = gif.resize((150, 100), Image.BICUBIC)
            global gif_image  # keep reference
            gif_image = ImageTk.PhotoImage(resized_gif)
            gif_label = tk.Label(alert_window, image=gif_image)
            gif_label.pack()
        except Exception as e:
            print(f"[{source_name}] Error loading GIF: {e}")

    def on_call_ambulance():
        call_ambulance(source_name)
        alert_window.destroy()

    call_button = tk.Button(alert_window, text="Call Ambulance", command=on_call_ambulance)
    call_button.pack(pady=6)

    cancel_button = tk.Button(alert_window, text="Cancel", command=alert_window.destroy)
    cancel_button.pack(pady=6)

    alert_window.mainloop()

def start_alert_thread(source_name):
    t = threading.Thread(target=show_alert_message, args=(source_name,), daemon=True)
    t.start()

def process_stream(source, source_name, model_json="model.json", model_weights="model_weights.keras", prob_threshold=95):
    """
    Worker for each video source.
    source: video path or int (camera index) or rtsp url
    source_name: friendly name for logs and filenames
    """
    global stop_all
    # create model instance inside thread (safer for concurrency)
    try:
        model = AccidentDetectionModel(model_json, model_weights)
    except Exception as e:
        print(f"[{source_name}] Error loading model: {e}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[{source_name}] Unable to open source: {source}")
        return

    window_name = f"Stream - {source_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    alarm_triggered = False

    while not stop_all:
        ret, frame = cap.read()
        if not ret:
            # If it's a file, exit; if it's a camera, retry a bit, else break
            print(f"[{source_name}] No frame (end or disconnect). Exiting worker.")
            break

        # Preprocess (same as your code)
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(rgb_frame, (250, 250))
            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        except Exception as e:
            print(f"[{source_name}] Prediction error: {e}")
            pred, prob = None, None

        if pred == "Accident":
            # adapt to your model's prob shape - original used prob[0][0]
            try:
                probability = round(prob[0][0] * 100, 2)
            except Exception:
                try:
                    probability = round(float(prob) * 100, 2)
                except Exception:
                    probability = 0.0

            if probability > prob_threshold and not alarm_triggered:
                print(f"[{source_name}] Accident detected with prob {probability}%")
                save_accident_photo(frame, source_name)
                alarm_triggered = True
                # trigger alert UI and Twilio call (threads)
                start_alert_thread(source_name)
                # also call ambulance directly (non-blocking)
                threading.Thread(target=call_ambulance, args=(source_name,), daemon=True).start()

            # overlay (for UI)
            cv2.rectangle(frame, (0, 0), (360, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {probability}%", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show frame for this stream
        cv2.imshow(window_name, frame)

        # handle keypress - q will stop everything
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[{source_name}] Q pressed - stopping all streams.")
            stop_all = True
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print(f"[{source_name}] Worker stopped.")

def start_multiple_streams(video_sources):
    """
    video_sources: list of tuples (source, source_name)
      source can be path (str) or int for webcam.
      source_name is a friendly display name.
    """
    threads = []
    for source, name in video_sources:
        t = threading.Thread(target=process_stream, args=(source, name), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.2)  # small stagger to ease I/O spikes

    try:
        # wait for threads to finish or global stop
        while any(t.is_alive() for t in threads) and not stop_all:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping all streams.")
    finally:
        # signal all threads to stop and join
        global stop_all
        stop_all = True
        for t in threads:
            t.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("All streams stopped.")

if __name__ == "__main__":
    # Example sources: adjust these to your files, camera indices, or RTSP URLs.
    # Use friendly names so saved images and logs are readable.
    video_sources = [
        ("camera1.mp4", "camera2.mp4"),
        ("camera3.mp4", "camera4.mp4"),
        (0, "Webcam0"),  # integer camera index works too
        # ("rtsp://user:pass@camera_ip/stream1", "Camera1"),
    ]

    start_multiple_streams(video_sources)
