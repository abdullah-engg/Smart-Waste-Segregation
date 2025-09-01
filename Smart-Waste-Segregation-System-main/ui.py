import time
import sys, torch, cv2, sounddevice as sd, requests
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton,
    QVBoxLayout, QWidget, QTextBrowser, QComboBox, QHBoxLayout, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from torchvision import transforms
from threading import Thread, Lock
import socket

CAMERA_INDEX_TRY = list(range(0, 6))
MIN_CONF = 0.60
STABLE_NEED = 3         
SEND_COOLDOWN_SEC = 5.0 

IMG_CLASSES = ["plastic", "organic_waste", "metal_cans", "glass"]
AUD_CLASSES = ["bottle", "paper", "can"]
BIN_IDS = {"plastic": 1, "organic_waste": 2, "metal_cans": 3, "glass": 4}
AUDIO_TO_IMG = {"bottle": "plastic", "paper": "organic_waste", "can": "metal_cans"}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

print("Loading Torch image model...")
try:
    image_model = torch.jit.load(
        r"C:\D disc\proto1\waste_classifier_scripted300.pt", map_location="cpu"
    )
    print("TorchScript image model loaded (CPU)")
except Exception as e:
    print("TorchScript load failed:", e)
    image_model = torch.load(
        r"C:\D disc\proto1\waste_classifier_final.pt", map_location="cpu"
    )
    print("Regular PyTorch model loaded (CPU)")
image_model.eval()
print("Loading TF audio model...")
audio_model = tf.keras.models.load_model(r"C:\D disc\proto1\audio_classifier.h5")
print("TF audio model loaded")

def _waveform_to_mel_logspec(x_np: np.ndarray, sr: int, n_mels: int = 64, target_frames: int = 2154) -> np.ndarray:
    x = tf.convert_to_tensor(x_np, dtype=tf.float32)
    stft = tf.signal.stft(x, frame_length=1024, frame_step=256, fft_length=1024, window_fn=tf.signal.hann_window)
    spec = tf.abs(stft)
    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=sr,
        lower_edge_hertz=20.0,
        upper_edge_hertz=sr / 2.0
    )
    mel_spec = tf.matmul(tf.square(spec), mel_w)
    mel_log = tf.math.log1p(mel_spec)
    mel_log = tf.transpose(mel_log, perm=[1, 0])
    frames = tf.shape(mel_log)[1]

    def pad():
        pad_amt = target_frames - tf.shape(mel_log)[1]
        return tf.pad(mel_log, paddings=[[0, 0], [0, pad_amt]], mode="CONSTANT")

    def crop():
        return mel_log[:, :target_frames]

    mel_fixed = tf.cond(frames < target_frames, pad, crop)
    mel_fixed = tf.expand_dims(mel_fixed, axis=0)
    mel_fixed = tf.expand_dims(mel_fixed, axis=-1)
    return mel_fixed.numpy()

def run_tf_audio_inference(audio_chunk_1d: np.ndarray, samplerate: int):
    try:
        if audio_chunk_1d.dtype != np.float32:
            audio_chunk_1d = audio_chunk_1d.astype(np.float32)
        if np.max(np.abs(audio_chunk_1d)) > 0:
            audio_chunk_1d = audio_chunk_1d / (np.max(np.abs(audio_chunk_1d)) + 1e-8)

        x = _waveform_to_mel_logspec(audio_chunk_1d, sr=samplerate, n_mels=64, target_frames=2154)
        pred = audio_model.predict(x, verbose=0).squeeze()
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        aud_name = AUD_CLASSES[idx] if 0 <= idx < len(AUD_CLASSES) else f"class_{idx}"
        return f"AUD: {aud_name}", conf
    except Exception as e:
        print("Audio model error:", e)
        return "AUD: Error", 0.0
def run_torch_image_inference(frame):
    try:
        x = transform(frame).unsqueeze(0)
        with torch.no_grad():
            out = image_model(x)
        pred = out.detach().cpu().numpy().squeeze()
        if pred.ndim == 1:
            exp = np.exp(pred - np.max(pred))
            probs = exp / (np.sum(exp) + 1e-8)
        else:
            probs = pred
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        class_name = IMG_CLASSES[idx] if 0 <= idx < len(IMG_CLASSES) else f"class_{idx}"
        return f"IMG: {class_name}", conf
    except Exception as e:
        print("Image model error:", e)
        return "IMG: Error", 0.0

def decide_final_class(img_pred, aud_pred):
    img_label, img_conf = img_pred
    aud_label, aud_conf = aud_pred

    img_cls = img_label.split(":", 1)[1].strip() if img_label.startswith("IMG:") else None
    aud_raw = aud_label.split(":", 1)[1].strip() if aud_label.startswith("AUD:") else None
    aud_cls = AUDIO_TO_IMG.get(aud_raw, None)

    if img_cls and aud_cls and img_cls == aud_cls:
        final_cls = img_cls
        final_conf = min(img_conf, aud_conf)
        return final_cls, final_conf
    else:
        return None, 0.0
ESP_IP = "192.168.213.193"
ESP_PORT = 8080             
def send_to_esp(bin_class_name):
    try:
        # Send the name directly
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ESP_IP, ESP_PORT))
            s.sendall(bin_class_name.encode())   
            response = s.recv(1024).decode().strip()
            print(f"Sent to ESP: {bin_class_name} | ESP replied: {response}")
            return True
    except Exception as e:
        print("ESP send error:", e)
        return False

# -----------------------------
# UI
# -----------------------------
class WasteClassifierUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("♻ Smart Waste Segregation Bin")
        self.resize(980, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #ECF0F1; }
            QLabel { font-size: 14px; }
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 8px 18px;
                border-radius: 8px;
                background-color: #27AE60;
                color: white;
            }
            QPushButton:hover { background-color: #1E8449; }
            QTextBrowser {
                background: #FDFEFE;
                border: 1px solid #ccc;
                padding: 6px;
                font-size: 13px;
            }
        """)

        layout = QVBoxLayout()

        cam_group = QGroupBox("Live Camera (Place object fully inside the green box)")
        cam_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        cam_layout = QVBoxLayout()
        self.cam_label = QLabel("Camera feed")
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("background:black; border:2px solid gray; border-radius: 8px;")
        self.cam_label.setScaledContents(True)
        cam_layout.addWidget(self.cam_label, alignment=Qt.AlignmentFlag.AlignCenter)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group, alignment=Qt.AlignmentFlag.AlignCenter)

        self.pred_label = QLabel("Predictions: IMG - | AUD -")
        self.pred_label.setStyleSheet("font-size:18px; font-weight:bold; color:#154360; padding:6px;")
        layout.addWidget(self.pred_label, alignment=Qt.AlignmentFlag.AlignCenter)

        device_group = QGroupBox("Device Selection")
        device_layout = QHBoxLayout()
        self.cam_select = QComboBox()
        self.mic_select = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(QLabel("Camera:"))
        device_layout.addWidget(self.cam_select)
        device_layout.addWidget(QLabel("Microphone:"))
        device_layout.addWidget(self.mic_select)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_all)
        self.stop_btn.clicked.connect(self.stop_all)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        self.log = QTextBrowser()
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.running_audio = False
        self.last_image_pred = ("IMG: -", 0.0)
        self.last_audio_pred = ("AUD: -", 0.0)
        self.lock = Lock()

        self._stable_class = None
        self._stable_count = 0
        self._last_sent_bin = None
        self._last_sent_time = 0.0

        self.box_frac = 0.50
        self.overlay_font = QFont("Arial", 12, QFont.Weight.Bold)
        self.frame_count = 0

    def refresh_devices(self):
        self.cam_select.clear()
        for idx in CAMERA_INDEX_TRY:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cam_select.addItem(f"Camera {idx}", idx)
                cap.release()
        if self.cam_select.count() == 0:
            self.cam_select.addItem("No Camera", None)

        self.mic_select.clear()
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                self.mic_select.addItem(f"{i}: {dev['name']}", i)
        if self.mic_select.count() == 0:
            self.mic_select.addItem("No Microphone", None)

    def start_all(self):
        cam_idx = self.cam_select.currentData()
        mic_idx = self.mic_select.currentData()

        if cam_idx is not None:
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.timer.start(1)
                self.log.append(f"Camera started at index {cam_idx}")
            else:
                self.log.append("Camera failed to start")

        if mic_idx is not None:
            self.running_audio = True
            Thread(target=self.audio_loop, args=(mic_idx,), daemon=True).start()
            self.log.append(f"Audio started on device {mic_idx}")

    def stop_all(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running_audio = False
        self.log.append("Stopped all")

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return

        self.frame_count += 1
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        bw = int(w * self.box_frac)
        bh = int(h * self.box_frac)
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        x2 = x1 + bw
        y2 = y1 + bh

        if self.frame_count % 5 == 0:
            roi = frame[y1:y2, x1:x2].copy()
            Thread(target=self.infer_and_update, args=(roi,), daemon=True).start()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, lineType=cv2.LINE_AA)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        label_text = "Place object fully inside the box"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = max(10, x1)
        ty = max(25, y1 - 10)
        cv2.rectangle(frame, (tx-6, ty-th-6), (tx+tw+6, ty+6), (0,0,0), -1)
        cv2.putText(frame, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        qt_img = QImage(rgb.data, w2, h2, ch * w2, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qt_img))

        with self.lock:
            text = f"{self.last_image_pred[0]} ({self.last_image_pred[1]*100:.1f}%) | " \
                   f"{self.last_audio_pred[0]} ({self.last_audio_pred[1]*100:.1f}%)"
            final_cls, final_conf = decide_final_class(self.last_image_pred, self.last_audio_pred)

        self.pred_label.setText("Predictions: " + text)

        now = time.time()
        if final_cls and final_conf >= MIN_CONF:
            if final_cls == self._stable_class:
                self._stable_count += 1
            else:
                self._stable_class = final_cls
                self._stable_count = 1

            if self._stable_count >= STABLE_NEED:
                bin_id = BIN_IDS[final_cls]
                if not (self._last_sent_bin == bin_id and (now - self._last_sent_time) < SEND_COOLDOWN_SEC):
                    ok = send_to_esp(final_cls)
                    if ok:
                        self.log.append(f"Final: {final_cls} ({final_conf*100:.1f}%) → Bin {bin_id} (sent)")
                        self._last_sent_bin = bin_id
                        self._last_sent_time = now
        else:
            self._stable_class = None
            self._stable_count = 0

    def infer_and_update(self, roi_frame):
        pred = run_torch_image_inference(roi_frame)
        with self.lock:
            self.last_image_pred = pred

    def audio_loop(self, mic_idx):
        samplerate = 44100
        try:
            device_info = sd.query_devices(mic_idx)
            samplerate = int(device_info['default_samplerate'])
            print(f"Using mic {mic_idx} ({device_info['name']}) at {samplerate} Hz")
        except Exception as e:
            print("Mic query error, using fallback samplerate 44100:", e)

        def callback(indata, frames, time_info, status):
            if not self.running_audio:
                raise sd.CallbackStop()
            try:
                pred = run_tf_audio_inference(indata[:, 0], samplerate)
                with self.lock:
                    self.last_audio_pred = pred
            except Exception as e:
                print("Audio callback error:", e)
                with self.lock:
                    self.last_audio_pred = ("AUD: Error", 0.0)

        try:
            with sd.InputStream(device=mic_idx,
                                channels=1,
                                samplerate=samplerate,
                                callback=callback,
                                blocksize=samplerate):
                while self.running_audio:
                    time.sleep(0.05)
        except Exception as e:
            print("Audio stream error:", e)
            with self.lock:
                self.last_audio_pred = ("AUD: Error", 0.0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WasteClassifierUI()
    win.show()
    sys.exit(app.exec())
