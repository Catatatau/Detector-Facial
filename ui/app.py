import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QFormLayout, QGroupBox, QListWidget, QListWidgetItem, QSplitter, QProgressBar
)
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QColor

from camera.capture import CameraStream
from recognition.pipeline import RecognitionPipeline
from registration.workflow import RegistrationWorkflow
from config.settings import config
from utils.logger import logger

class FacePlatformApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enterprise Local Face Recognition Platform")
        self.resize(1024, 768)
        
        # Core components
        self.camera = CameraStream(src=config.camera.index, resolution=config.camera.resolution, fps=config.camera.frame_rate)
        self.camera.start()
        
        self.recognition_pipeline = RecognitionPipeline()
        self.registration_workflow = RegistrationWorkflow(required_samples=5)
        
        self.is_registering = False
        
        self.setup_ui()
        
        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / config.camera.frame_rate))
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: Camera Feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel("Camera Feed Initializing...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.video_label, stretch=1)
        
        splitter.addWidget(left_panel)
        
        # Right side: Controls & Info
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Registration Group
        reg_group = QGroupBox("User Registration")
        reg_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter full name")
        
        self.btn_register = QPushButton("Start Registration")
        self.btn_register.clicked.connect(self.toggle_registration)
        
        reg_layout.addRow("Name:", self.name_input)
        reg_layout.addRow("", self.btn_register)
        
        self.reg_status = QLabel("")
        self.reg_status.setStyleSheet("color: blue;")
        reg_layout.addRow(self.reg_status)
        
        self.reg_progress = QProgressBar()
        self.reg_progress.setRange(0, self.registration_workflow.required_samples)
        self.reg_progress.setValue(0)
        reg_layout.addRow(self.reg_progress)
        
        reg_group.setLayout(reg_layout)
        right_layout.addWidget(reg_group)
        
        # History Group
        hist_group = QGroupBox("Recent Recognitions")
        hist_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        hist_layout.addWidget(self.history_list)
        
        hist_group.setLayout(hist_layout)
        right_layout.addWidget(hist_group, stretch=1)
        
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([700, 324])

    @Slot()
    def toggle_registration(self):
        if not self.is_registering:
            name = self.name_input.text().strip()
            if not name:
                self.reg_status.setText("Error: Name cannot be empty.")
                self.reg_status.setStyleSheet("color: red;")
                return
                
            self.is_registering = True
            self.btn_register.setText("Cancel Registration")
            self.name_input.setEnabled(False)
            self.reg_progress.setValue(0)
            self.registration_workflow.start_registration(name)
            self.reg_status.setText(f"Registering {name}...")
            self.reg_status.setStyleSheet("color: blue;")
        else:
            self.is_registering = False
            self.btn_register.setText("Start Registration")
            self.name_input.setEnabled(True)
            self.name_input.clear()
            self.reg_progress.setValue(0)
            self.reg_status.setText("Registration cancelled.")
            self.reg_status.setStyleSheet("color: red;")

    @Slot()
    def update_frame(self):
        grabbed, frame = self.camera.read()
        if not grabbed or frame is None:
            return
            
        display_frame = frame.copy()
        
        if self.is_registering:
            # Optionally get faces just to draw box in registration mode
            faces = self.registration_workflow.detector.detect(frame)
            if faces:
                face = faces[0]
                x, y, w, h = face["bbox"]
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
            success, msg, count = self.registration_workflow.process_frame(frame)
            self.reg_status.setText(f"{msg}")
            self.reg_progress.setValue(count)
            
            if count >= self.registration_workflow.required_samples:
                if self.registration_workflow.complete_registration():
                    self.reg_status.setText("Registration Complete!")
                    self.reg_status.setStyleSheet("color: green;")
                    self.recognition_pipeline.reload_matcher()
                else:
                    self.reg_status.setText("Registration Failed!")
                    self.reg_status.setStyleSheet("color: red;")
                    
                self.is_registering = False
                self.btn_register.setText("Start Registration")
                self.name_input.setEnabled(True)
                self.name_input.clear()
                self.reg_progress.setValue(0)
                
            # Draw simple text for registration mode
            cv2.putText(display_frame, "Registration Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
        else:
            # Recognition mode
            results = self.recognition_pipeline.process_frame(frame)
            access_granted = False
            granted_name = ""
            
            for res in results:
                bbox = res["bbox"]
                landmarks = res.get("landmarks")
                identity = res["identity"]
                score = res["score"]
                
                x, y, w, h = bbox
                
                # Draw bounding box
                color = (0, 255, 0) if identity else (0, 0, 255)
                
                # Draw corner brackets for a nicer look
                thickness = 2
                length = 20
                # Top left
                cv2.line(display_frame, (x, y), (x + length, y), color, thickness)
                cv2.line(display_frame, (x, y), (x, y + length), color, thickness)
                # Top right
                cv2.line(display_frame, (x + w, y), (x + w - length, y), color, thickness)
                cv2.line(display_frame, (x + w, y), (x + w, y + length), color, thickness)
                # Bottom left
                cv2.line(display_frame, (x, y + h), (x + length, y + h), color, thickness)
                cv2.line(display_frame, (x, y + h), (x, y + h - length), color, thickness)
                # Bottom right
                cv2.line(display_frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
                cv2.line(display_frame, (x + w, y + h), (x + w, y + h - length), color, thickness)
                
                # Draw faint full rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 1)
                
                # Draw landmarks (eyes, nose, mouth corners)
                if landmarks is not None:
                    for pt in landmarks:
                        cx, cy = int(pt[0]), int(pt[1])
                        cv2.circle(display_frame, (cx, cy), 3, (0, 255, 255), -1)
                
                label = f"{identity['name']} ({score:.2f})" if identity else f"Unknown ({score:.2f})"
                
                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
                cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update history list if recognized and high confidence
                if identity and score > config.models.recognition_threshold:
                    self._add_to_history(f"{identity['name']} matched with {score:.2f}")
                    access_granted = True
                    granted_name = identity['name']

            if access_granted:
                cv2.putText(display_frame, f"ACCESS GRANTED: {granted_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Convert to QPixmap
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale pixmap to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def _add_to_history(self, text: str):
        # Check if last item is the same (avoid spamming)
        if self.history_list.count() > 0:
            last_item = self.history_list.item(0).text()
            if text in last_item:
                return
                
        item = QListWidgetItem(text)
        self.history_list.insertItem(0, item)
        if self.history_list.count() > 50:
            self.history_list.takeItem(50)

    def closeEvent(self, event):
        self.camera.stop()
        event.accept()

def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Modern look
    window = FacePlatformApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
