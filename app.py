import cv2
import os
import time
import mysql.connector
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import torch
import threading

app = Flask(__name__)

# --- 1. KONFIGURASI UTAMA ---
MODEL_PATH = "best.pt"
CAMERA_INDEX = "http://10.62.20.69:4747/video"
CONFIDENCE_THRESHOLD = 0.4

# --- KONFIGURASI ANTI-FALSE DETECTION ---
STABLE_FRAME_THRESHOLD = 5      # Label harus konsisten N frame berturut-turut
DETECTION_DELAY        = 0.8    # Jeda (detik) setelah part masuk sebelum mulai ngitung frame

# --- 2. SETUP AI ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- SISTEM DIMULAI ---")
print(f"Perangkat Terdeteksi: {device.upper()}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO(MODEL_PATH).to(device)

# --- 3. STATE MANAGEMENT ---
detection_state = {
    "part_present":          False,
    "already_saved":         False,
    "last_label":            None,
    "last_annotated_frame":  None,
    "camera_status":         "connecting",

    # Frame counter
    "consecutive_label":     None,
    "consecutive_count":     0,

    # Delay timer
    "detection_delay_start": None,
    "delay_passed":          False,

    # Performa
    "fps":                   0.0,
    "cycle_time":            None,
    "cycle_start":           None,
}
state_lock = threading.Lock()

# --- 4. FUNGSI DATABASE ---
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="db_produksi"
    )

def save_to_db(status, frame=None, cycle_time=None):
    try:
        db = get_db_connection()
        cursor = db.cursor()
        foto_name = ""
        if frame is not None:
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')
            foto_name = f"{status.lower()}_{int(time.time())}.jpg"
            cv2.imwrite(f"static/uploads/{foto_name}", frame)
        sql = "INSERT INTO log_deteksi (status, foto_path, cycle_time) VALUES (%s, %s, %s)"
        cursor.execute(sql, (status, foto_name, cycle_time))
        db.commit()
        cursor.close()
        db.close()
        print(f"[DB] Berhasil mencatat status: {status}, foto: {foto_name}, cycle_time: {cycle_time}s")
    except Exception as e:
        print(f"[DB ERROR]: {e}")

# --- 5. HELPER: GAMBAR BOUNDING BOX MANUAL ---
def draw_boxes(frame, results, roi_x_start, roi_y_start):
    best_label = None
    best_conf  = 0.0
    for r in results:
        if len(r.boxes) == 0:
            continue
        boxes_sorted = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)
        top_box = boxes_sorted[0]
        label = model.names[int(top_box.cls[0])]
        conf  = float(top_box.conf[0])
        if conf > best_conf:
            best_conf  = conf
            best_label = label
        color = (0, 255, 0) if label.upper() == "OK" else (0, 0, 255)
        x1, y1, x2, y2 = map(int, top_box.xyxy[0])
        x1f, y1f = x1 + roi_x_start, y1 + roi_y_start
        x2f, y2f = x2 + roi_x_start, y2 + roi_y_start
        cv2.rectangle(frame, (x1f, y1f), (x2f, y2f), color, 3)
        label_text = f"{label.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1f, y1f - th - 10), (x1f + tw + 6, y1f), color, -1)
        cv2.putText(frame, label_text, (x1f + 3, y1f - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame, best_label, best_conf

# --- 6. CORE LOGIKA VIDEO & DETEKSI ---
def generate_frames():
    reconnect_delay = 3

    while True:
        print(f"[KAMERA] Mencoba konek ke: {CAMERA_INDEX}")
        with state_lock:
            detection_state["camera_status"] = "connecting"

        cap = cv2.VideoCapture(CAMERA_INDEX) if isinstance(CAMERA_INDEX, str) \
              else cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"[KAMERA] Gagal konek. Retry dalam {reconnect_delay}s...")
            with state_lock:
                detection_state["camera_status"] = "disconnected"
            time.sleep(reconnect_delay)
            continue

        print("[KAMERA] Berhasil konek!")
        with state_lock:
            detection_state["camera_status"] = "ok"

        consecutive_fail = 0
        MAX_FAIL         = 30

        # FPS tracking
        fps_counter = 0
        fps_timer   = time.time()

        while True:
            success, frame = cap.read()

            # Hitung FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                with state_lock:
                    detection_state["fps"] = round(fps_counter / (time.time() - fps_timer), 1)
                fps_counter = 0
                fps_timer   = time.time()

            if not success:
                consecutive_fail += 1
                if consecutive_fail >= MAX_FAIL:
                    print("[KAMERA] Koneksi terputus. Reconnecting...")
                    with state_lock:
                        detection_state["camera_status"] = "disconnected"
                    break
                blank = _make_blank_frame(640, 480, "Menghubungkan ulang kamera...")
                _, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            consecutive_fail = 0

            # --- LOGIKA ROI ---
            height, width, _ = frame.shape
            roi_size    = int(min(width, height) * 0.35)
            roi_x_start = (width  - roi_size) // 2
            roi_y_start = (height - roi_size) // 2
            roi_x_end   = roi_x_start + roi_size
            roi_y_end   = roi_y_start + roi_size
            roi_frame   = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            results       = model(roi_frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
            part_detected = len(results[0].boxes) > 0
            now           = time.time()

            if part_detected:
                frame, best_label, best_conf = draw_boxes(frame, results, roi_x_start, roi_y_start)
                border_color = (0, 255, 0) if best_label and best_label.upper() == "OK" else (0, 0, 255)
                cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), border_color, 2)

                with state_lock:
                    # Transisi STANDBY → DETECTED
                    if not detection_state["part_present"]:
                        detection_state["part_present"]          = True
                        detection_state["already_saved"]         = False
                        detection_state["detection_delay_start"] = now
                        detection_state["delay_passed"]          = False
                        detection_state["consecutive_count"]     = 0
                        detection_state["consecutive_label"]     = None
                        detection_state["cycle_start"]           = now
                        print("[STATE] Part masuk. Delay dimulai...")

                    detection_state["last_label"]           = best_label
                    detection_state["last_annotated_frame"] = frame.copy()

                    # Cek delay
                    delay_start  = detection_state["detection_delay_start"]
                    delay_passed = detection_state["delay_passed"]
                    if not delay_passed and (now - delay_start) >= DETECTION_DELAY:
                        detection_state["delay_passed"] = True
                        delay_passed = True
                        print(f"[STATE] Delay {DETECTION_DELAY}s lewat. Mulai hitung frame konsisten.")

                    # Hitung frame konsisten — hanya setelah delay lewat
                    if delay_passed and not detection_state["already_saved"] and best_label is not None:
                        if best_label == detection_state["consecutive_label"]:
                            detection_state["consecutive_count"] += 1
                        else:
                            # Label berubah → reset counter
                            detection_state["consecutive_label"] = best_label
                            detection_state["consecutive_count"] = 1
                            print(f"[STATE] Label berubah ke {best_label}. Counter reset.")

                        count = detection_state["consecutive_count"]
                        print(f"[STATE] Frame konsisten: {count}/{STABLE_FRAME_THRESHOLD} ({best_label})")

                        if count >= STABLE_FRAME_THRESHOLD:
                            cycle_time = round(now - detection_state["cycle_start"], 2)
                            detection_state["cycle_time"]    = cycle_time
                            detection_state["already_saved"] = True
                            save_to_db(best_label, frame.copy(), cycle_time)
                            print(f"[STATE] ✅ {best_label} | Cycle time: {cycle_time}s")

                    # Teks status di bawah ROI
                    if not detection_state["already_saved"]:
                        if not detection_state["delay_passed"]:
                            sisa     = max(0, DETECTION_DELAY - (now - delay_start))
                            txt, col = f"Tunggu tangan... {sisa:.1f}s", (0, 200, 255)
                        else:
                            count    = detection_state["consecutive_count"]
                            txt, col = f"Verifikasi: {count}/{STABLE_FRAME_THRESHOLD}", (0, 255, 180)
                        cv2.putText(frame, txt, (roi_x_start, roi_y_end + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                    else:
                        cv2.putText(frame, "TERSIMPAN", (roi_x_start, roi_y_end + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            else:
                # STANDBY
                cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
                cv2.putText(frame, "STANDBY - MASUKKAN PART",
                            (roi_x_start, roi_y_start - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                with state_lock:
                    if detection_state["part_present"]:
                        print("[STATE] Part diangkat. State di-reset.")
                    detection_state["part_present"]          = False
                    detection_state["already_saved"]         = False
                    detection_state["last_label"]            = None
                    detection_state["consecutive_label"]     = None
                    detection_state["consecutive_count"]     = 0
                    detection_state["detection_delay_start"] = None
                    detection_state["delay_passed"]          = False

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()
        time.sleep(reconnect_delay)

def _make_blank_frame(w, h, text):
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(blank, text, (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return blank

# --- 7. ROUTES FLASK ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    try:
        db = get_db_connection()
        cursor = db.cursor(buffered=True)
        cursor.execute("SELECT status, COUNT(*) FROM log_deteksi GROUP BY status")
        stats = dict(cursor.fetchall())
        cursor.close()
        db.close()
        with state_lock:
            s = detection_state
        return jsonify({
            "ok":            stats.get("OK", 0) or stats.get("ok", 0),
            "ng":            stats.get("NG", 0) or stats.get("ng", 0),
            "camera_status": s["camera_status"],
            "last_label":    s["last_label"],
            "already_saved": s["already_saved"],
            "part_present":  s["part_present"],
            "fps":           s["fps"],
            "cycle_time":    s["cycle_time"],
        })
    except Exception as e:
        print(f"[STATS ERROR]: {e}")
        return jsonify({"ok": 0, "ng": 0, "camera_status": "disconnected",
                        "last_label": None, "already_saved": False,
                        "part_present": False, "fps": 0, "cycle_time": None})

@app.route('/manual_capture', methods=['POST'])
def manual_capture():
    with state_lock:
        part_present  = detection_state["part_present"]
        already_saved = detection_state["already_saved"]
        last_label    = detection_state["last_label"]
        last_frame    = detection_state["last_annotated_frame"]
        cycle_start   = detection_state["cycle_start"]

    if not part_present:
        return jsonify({"success": False, "message": "Tidak ada part di depan kamera."})
    if already_saved:
        return jsonify({"success": False, "message": "Sudah tersimpan otomatis untuk part ini."})
    if last_label is None or last_frame is None:
        return jsonify({"success": False, "message": "Belum ada hasil deteksi."})

    save_to_db(last_label, last_frame, cycle_time)
    cycle_time = round(time.time() - cycle_start, 2) if cycle_start else None
    with state_lock:
        detection_state["already_saved"] = True
        detection_state["cycle_time"]    = cycle_time

    return jsonify({"success": True, "message": f"Manual capture berhasil: {last_label}"})

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    print("Aplikasi Berjalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)