import sys
import cv2
import time
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VideoApp(QWidget):
    def __init__(self, video1_path, video2_path, dino_path, model_path):
        super().__init__()
        self.setWindowTitle("Hat Projesi - YOLO Detection")
        self.setGeometry(200, 100, 1280, 720)
        self.setStyleSheet("background-color: #2b0426;")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.tracker1 = DeepSort(max_age=10, nn_budget=50)
        self.tracker2 = DeepSort(max_age=10, nn_budget=50)
        
        self.conf_threshold = 0.25
        self.line_position = 320
        self.line_tolerance = 15
        
        self.cross_count1 = 0
        self.cross_count2 = 0
        self.total_production = 0  # Bağımsız toplam sayaç
        self.counted_ids1 = set()
        self.counted_ids2 = set()

        # ===   TREXXXX MERT SOFTWARE ===
        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(10, 10, 100, 100)
        img = cv2.imread(dino_path)
        if img is not None:
            b, g, r = cv2.split(img)
            alpha = cv2.inRange(img, (200,200,200), (255,255,255))
            alpha = 255 - alpha
            rgba = cv2.merge([b, g, r, alpha])
            h, w, ch = rgba.shape
            bytes_per_line = ch * w
            qimg = QImage(rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            qimg = qimg.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(QPixmap.fromImage(qimg))

        # === Ürün kutuları ===
        self.product_box = QLabel(f"1. Hat Üretimi: {self.cross_count1}", self)
        self.product_box.setGeometry(50, 300, 250, 70)
        self.product_box.setAlignment(Qt.AlignCenter)
        self.product_box.setFont(QFont("Arial", 12, QFont.Bold))
        self.product_box.setStyleSheet("""
            border-radius: 15px;
            color: white;
            padding: 5px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #2196F3, stop:1 #3a0d52
            );
        """)

        self.daily_box = QLabel(f"2. Hat Üretimi: {self.cross_count2}", self)
        self.daily_box.setGeometry(340, 300, 250, 70)
        self.daily_box.setAlignment(Qt.AlignCenter)
        self.daily_box.setFont(QFont("Arial", 12, QFont.Bold))
        self.daily_box.setStyleSheet("""
            border-radius: 15px;
            color: white;
            padding: 5px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #2196F3, stop:1 #3a0d52
            );
        """)

        self.total_box = QLabel(f"Toplam Üretim: {self.cross_count1 + self.cross_count2}", self)
        self.total_box.setGeometry(50, 190, 250, 70)
        self.total_box.setAlignment(Qt.AlignCenter)
        self.total_box.setFont(QFont("Arial", 12, QFont.Bold))
        self.total_box.setStyleSheet("""
            border-radius: 15px;
            color: white;
            padding: 5px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #2196F3, stop:1 #3a0d52
            );
        """)

        self.target_box = QLabel("Hedeflenen Üretim: 10", self)
        self.target_box.setGeometry(340, 190, 250, 70)
        self.target_box.setAlignment(Qt.AlignCenter)
        self.target_box.setFont(QFont("Arial", 12, QFont.Bold))
        self.target_box.setStyleSheet("""
            border-radius: 15px;
            color: white;
            padding: 5px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #F44336, stop:1 #E57373
            );
        """)

        # Video alanları 
        self.video1_x, self.video1_y = 650, 30
        self.video2_x, self.video2_y = 650, 400
        self.video_w, self.video_h = 600, 300

        self.video1_label = QLabel(self)
        self.video1_label.setGeometry(self.video1_x, self.video1_y, self.video_w, self.video_h)
        self.video1_label.setAlignment(Qt.AlignCenter)
        self.video1_label.setStyleSheet("border: 2px solid none; border-radius: 15px; background-color: black;")
        self.video1_text = QLabel("Kamera 1", self.video1_label)
        self.video1_text.setGeometry(10, 10, 150, 40)
        self.video1_text.setFont(QFont("Arial", 16, QFont.Bold))
        self.video1_text.setStyleSheet("color: white; background-color: rgba(0,0,0,0); border: none;")
        self.fps1_label = QLabel("FPS: 0", self.video1_label)
        self.fps1_label.setGeometry(self.video_w - 100, 10, 90, 30)
        self.fps1_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fps1_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.fps1_label.setStyleSheet("color: yellow; background-color: rgba(0,0,0,0); border: none;")

        self.video2_label = QLabel(self)
        self.video2_label.setGeometry(self.video2_x, self.video2_y, self.video_w, self.video_h)
        self.video2_label.setAlignment(Qt.AlignCenter)
        self.video2_label.setStyleSheet("border: 2px solid none; border-radius: 15px; background-color: black;")
        self.video2_text = QLabel("Kamera 2", self.video2_label)
        self.video2_text.setGeometry(10, 10, 150, 40)
        self.video2_text.setFont(QFont("Arial", 16, QFont.Bold))
        self.video2_text.setStyleSheet("color: white; background-color: rgba(0,0,0,0); border: none;")
        self.fps2_label = QLabel("FPS: 0", self.video2_label)
        self.fps2_label.setGeometry(self.video_w - 100, 10, 90, 30)
        self.fps2_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fps2_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.fps2_label.setStyleSheet("color: yellow; background-color: rgba(0,0,0,0); border: none;")

        # Sistem saati ve geçen süre 
        self.datetime_label = QLabel("", self)
        self.datetime_label.setGeometry(10, 670, 400, 40)
        self.datetime_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.datetime_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.datetime_label.setStyleSheet("color: white; background-color: transparent; border: none;")

        # Alt kutular cartlar curtlar
        self.cam1_label = QLabel("Kamera 1", self)
        self.cam1_label.setGeometry(50, 520, 120, 40)
        self.cam1_label.setAlignment(Qt.AlignCenter)
        self.cam1_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.cam1_label.setStyleSheet("color: white; background-color: rgba(0,255,0,0.5); border-radius: 8px;")

        self.cam2_label = QLabel("Kamera 2", self)
        self.cam2_label.setGeometry(340, 520, 120, 40)
        self.cam2_label.setAlignment(Qt.AlignCenter)
        self.cam2_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.cam2_label.setStyleSheet("color: white; background-color: rgba(0,255,0,0.5); border-radius: 8px;")

        # Sıfırlama butonları
        self.reset1_btn = QPushButton("1. Hattı Sıfırla", self)
        self.reset1_btn.setGeometry(180, 520, 140, 40)
        self.reset1_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.reset1_btn.setStyleSheet("""
             QPushButton {
                background-color: #D32F2F;
                color: white;
                border-radius: 10px;
            }
            QPushButton:pressed {
                background-color: #E64A19;
            }
        """)
        self.reset1_btn.clicked.connect(self.reset_hat1)

        self.reset2_btn = QPushButton("2. Hattı Sıfırla", self)
        self.reset2_btn.setGeometry(470, 520, 140, 40)
        self.reset2_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.reset2_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                border-radius: 10px;
            }
            QPushButton:pressed {
                background-color: #E64A19;
            }
        """)
        self.reset2_btn.clicked.connect(self.reset_hat2)

        # === Video açma ===
        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)

        self.prev_time1 = time.time()
        self.prev_time2 = time.time()
        self.start_time = time.time()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def detect_and_count(self, frame, tracker, counted_ids):
        """YOLO detection ve DeepSORT tracking"""
        frame_resized = cv2.resize(frame, (640, 480))
        
        results = self.model(frame_resized, conf=self.conf_threshold, 
                           device=self.device, imgsz=640, verbose=False)

        detections = []
        for r in results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                for box, score in zip(r.boxes.xyxy, r.boxes.conf):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    score = float(score.cpu().numpy())
                    detections.append(([x1, y1, x2, y2], score))

        tracks = tracker.update_tracks(detections, frame=frame_resized)

        cross_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            cx = int((x1 + x2) / 2)

            if track_id not in counted_ids and (self.line_position - self.line_tolerance <= cx <= self.line_position + self.line_tolerance):
                cross_count += 1
                counted_ids.add(track_id)

        # Çizgi ve sayımı frame'e ekle
        cv2.line(frame_resized, (self.line_position, 0), (self.line_position, 480), (0, 255, 0), 2)
        cv2.putText(frame_resized, f'Count: {cross_count}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame_resized, cross_count

    def reset_hat1(self):
        self.cross_count1 = 0
        self.counted_ids1 = set()
        self.product_box.setText("1. Hat Üretimi: 0")
        # Toplam değişmez!

    def reset_hat2(self):
        self.cross_count2 = 0
        self.counted_ids2 = set()
        self.daily_box.setText("2. Hat Üretimi: 0")
        # Toplam değişmez!

    def update_frame(self):
        # Video 1
        ret1, frame1 = self.cap1.read()
        if ret1:
            frame1_processed, count_increment1 = self.detect_and_count(frame1, self.tracker1, self.counted_ids1)
            self.cross_count1 += count_increment1
            self.total_production += count_increment1  # Toplama ekle
            
            frame1_rgb = cv2.cvtColor(frame1_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = frame1_rgb.shape
            bytes_per_line = ch * w
            qimg1 = QImage(frame1_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video1_label.setPixmap(QPixmap.fromImage(qimg1))
            
            curr_time1 = time.time()
            fps1 = 1 / (curr_time1 - self.prev_time1)
            self.prev_time1 = curr_time1
            self.fps1_label.setText(f"FPS: {int(fps1)}")
            self.cam1_label.setStyleSheet("color: white; background-color: rgba(0,255,175,0.5); border-radius: 8px;")
        else:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cam1_label.setStyleSheet("color: white; background-color: rgba(255,0,0,0.5); border-radius: 8px;")

        # Video 2
        ret2, frame2 = self.cap2.read()
        if ret2:
            frame2_processed, count_increment2 = self.detect_and_count(frame2, self.tracker2, self.counted_ids2)
            self.cross_count2 += count_increment2
            self.total_production += count_increment2  # Toplama ekle
            
            frame2_rgb = cv2.cvtColor(frame2_processed, cv2.COLOR_BGR2RGB)
            h, w, ch = frame2_rgb.shape
            bytes_per_line = ch * w
            qimg2 = QImage(frame2_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video2_label.setPixmap(QPixmap.fromImage(qimg2))
            
            curr_time2 = time.time()
            fps2 = 1 / (curr_time2 - self.prev_time2)
            self.prev_time2 = curr_time2
            self.fps2_label.setText(f"FPS: {int(fps2)}")
            self.cam2_label.setStyleSheet("color: white; background-color: rgba(0,255,175,0.5); border-radius: 8px;")
        else:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.cam2_label.setStyleSheet("color: white; background-color: rgba(255,0,0,0.5); border-radius: 8px;")

        # Üretim kutularını güncelle
        self.product_box.setText(f"1. Hat Üretimi: {self.cross_count1}")
        self.daily_box.setText(f"2. Hat Üretimi: {self.cross_count2}")
        
        # Toplam bağımsız olarak artmaya devam eder
        self.total_box.setText(f"Toplam Üretim: {self.total_production}")

        try:
            target_production = int(self.target_box.text().split(":")[1].strip())
            percentage = self.total_production / target_production
        except:
            percentage = 0

        # renk belirleme
        if percentage <= 0.33:
            color_start, color_end = "#F44336", "#E57373"
        elif percentage <= 0.66:
            color_start, color_end = "#FF9800", "#FFB74D"
        elif percentage < 1.0:
            color_start, color_end = "#FFEB3B", "#FFF176"
        else:
            color_start, color_end = "#4CAF50", "#81C784"

        self.target_box.setStyleSheet(f"""
            border-radius: 15px;
            color: white;
            padding: 5px;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 {color_start}, stop:1 {color_end}
            );
        """)

        # sistem saati
        current_time = datetime.now().strftime("%H:%M:%S")
        elapsed_sec = int(time.time() - self.start_time)
        hours = elapsed_sec // 3600
        minutes = (elapsed_sec % 3600) // 60
        seconds = elapsed_sec % 60
        elapsed_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.datetime_label.setText(f"Saat: {current_time} | Geçen Süre: {elapsed_time_str}")

    def closeEvent(self, event):
        if self.cap1.isOpened():
            self.cap1.release()
        if self.cap2.isOpened():
            self.cap2.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    video1_path="data/video.mp4", 
    video2_path="data/video.mp4", 
    dino_path="assets/dino.png", 
    model_path="models/best.pt"
    win = VideoApp(video1_path, video2_path, dino_path, model_path)
    win.show()
    sys.exit(app.exec_())
