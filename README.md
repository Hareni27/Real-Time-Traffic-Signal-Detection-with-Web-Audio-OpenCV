# Real-Time-Traffic-Signal-Detection-with-Web-Audio-OpenCV
Build a real-time intelligent system that detects traffic signal colors (Red, Green, Yellow) using a camera feed and provides audio alerts via a webpage output using OpenCV. 
"""
Real-Time Traffic Signal Detection System
Flask + OpenCV Backend
Run: python app.py
Open: http://localhost:5000
"""

import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template, jsonify
from collections import deque, Counter

app = Flask(__name__)

# ──────────────────────────────────────────────
# GLOBAL STATE
# ──────────────────────────────────────────────
detection_state = {
    "color": "UNKNOWN",
    "confidence": 0.0,
    "fps": 0.0,
    "locked": False,
    "lock_frames": 0,
}

camera = None
frame_buffer = None
buffer_lock = threading.Lock()
detection_history = deque(maxlen=15)   # smoothing buffer
lock_counter = {"color": None, "count": 0}

# ──────────────────────────────────────────────
# HSV COLOR RANGES (tuned for traffic signals)
# ──────────────────────────────────────────────
COLOR_RANGES = {
    "RED": [
        (np.array([0,   100, 100]), np.array([10,  255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),  # red wraps hue circle
    ],
    "GREEN": [
        (np.array([40, 60, 60]),  np.array([90, 255, 255])),
    ],
    "YELLOW": [
        (np.array([15, 100, 100]), np.array([35, 255, 255])),
    ],
}


# ──────────────────────────────────────────────
# DETECTION FUNCTION
# ──────────────────────────────────────────────
def detect_signal(frame):
    """
    Detects traffic signal color from a frame.
    Returns: (color, confidence, annotated_frame, coverage_dict)
    """

    # Resize for speed
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Convert BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Slight blur to reduce noise
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    coverage = {}
    masks = {}

    for color, ranges in COLOR_RANGES.items():
        # Combine all masks for this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        # Morphological cleanup — removes noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        pixel_count = cv2.countNonZero(mask)
        total       = frame.shape[0] * frame.shape[1]
        coverage[color] = pixel_count / total
        masks[color]    = mask

    # Pick dominant color
    best_color    = max(coverage, key=coverage.get)
    best_coverage = coverage[best_color]

    MIN_THRESHOLD = 0.005   # ignore tiny blobs
    if best_coverage < MIN_THRESHOLD:
        detected, confidence = "UNKNOWN", 0.0
    else:
        detected   = best_color
        confidence = min(100.0, (best_coverage / 0.15) * 100)

    # ── Annotate frame ──
    annotated = frame.copy()

    color_bgr = {
        "RED":     (0,  30, 220),
        "GREEN":   (0, 200,  60),
        "YELLOW":  (0, 200, 230),
        "UNKNOWN": (120, 120, 120),
    }
    sig_color = color_bgr.get(detected, (120, 120, 120))

    # Draw bounding box around detected region
    if detected != "UNKNOWN":
        contours, _ = cv2.findContours(
            masks[detected], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), sig_color, 2)
            cv2.drawContours(annotated, [largest], -1, sig_color, 1)

    # HUD bar at top
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 55), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    label = f"Signal: {detected}  |  Conf: {confidence:.1f}%"
    cv2.putText(annotated, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, sig_color, 2, cv2.LINE_AA)

    return detected, confidence, annotated, coverage


# ──────────────────────────────────────────────
# SMOOTHING — majority vote over last 15 frames
# ──────────────────────────────────────────────
def smooth(new_color, confidence):
    detection_history.append(new_color)
    if len(detection_history) < 3:
        return new_color, confidence
    counts = Counter(detection_history)
    best, count = counts.most_common(1)[0]
    return best, min(100.0, (count / len(detection_history)) * confidence)


# ──────────────────────────────────────────────
# SYSTEM LOCK — locks after 8 consistent frames
# ──────────────────────────────────────────────
def update_lock(color):
    if color == lock_counter["color"]:
        lock_counter["count"] += 1
    else:
        lock_counter["color"] = color
        lock_counter["count"] = 1
    locked = (color != "UNKNOWN") and (lock_counter["count"] >= 8)
    return locked, lock_counter["count"]


# ──────────────────────────────────────────────
# CAMERA CAPTURE LOOP (background thread)
# ──────────────────────────────────────────────
def capture_loop():
    global frame_buffer, detection_state, camera

    # Try webcam indices 0, 1, 2
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            camera = cap
            print(f"[INFO] Camera opened at index {idx}")
            break

    if camera is None or not camera.isOpened():
        print("[ERROR] No camera found. Check connection.")
        return

    prev_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.05)
            continue

        # Detect
        color, conf, annotated, cov = detect_signal(frame)

        # Smooth
        smooth_color, smooth_conf = smooth(color, conf)

        # Lock
        locked, lock_frames = update_lock(smooth_color)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Update global state
        detection_state.update({
            "color":       smooth_color,
            "confidence":  round(smooth_conf, 1),
            "fps":         round(fps, 1),
            "locked":      locked,
            "lock_frames": lock_frames,
            "coverage": {
                k: round(v * 100, 3) for k, v in cov.items()
            },
        })

        with buffer_lock:
            frame_buffer = annotated

        time.sleep(0.01)   # ~100fps cap, Flask throttles to 30fps


# ──────────────────────────────────────────────
# MJPEG STREAM GENERATOR
# ──────────────────────────────────────────────
def generate_frames():
    while True:
        with buffer_lock:
            frame = frame_buffer

        if frame is None:
            # Blank frame while camera loads
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Connecting to camera...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
        time.sleep(0.033)   # 30fps


# ──────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main webpage."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Live MJPEG camera stream."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """JSON API — polled every 500ms by the frontend."""
    return jsonify(detection_state)


@app.route("/speak")
def speak():
    """
    Optional: trigger Python TTS on server side.
    Called by frontend as fallback if browser TTS is unavailable.
    """
    from flask import request
    color = request.args.get("color", "UNKNOWN")

    messages = {
        "RED":     "Red signal. Stop!",
        "GREEN":   "Green signal. Go!",
        "YELLOW":  "Yellow signal. Slow down!",
        "UNKNOWN": "Signal not detected.",
    }

    def speak_async(msg):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    msg = messages.get(color, "Signal not detected.")
    t = threading.Thread(target=speak_async, args=(msg,), daemon=True)
    t.start()

    return jsonify({"spoken": msg})


# ──────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Start camera in background thread
    cam_thread = threading.Thread(target=capture_loop, daemon=True)
    cam_thread.start()

    print("=" * 45)
    print("  Traffic Signal Detection System")
    print("  Open: http://localhost:5000")
    print("=" * 45)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,      # Keep False — debug=True breaks threading
        threaded=True,
    )
