import cv2
import time
import numpy as np
from datetime import datetime
import os
import multiprocessing as mp
import sqlite3
import json
from collections import deque
import mediapipe as mp_lib
import torch
import logging
from threading import Thread
from queue import Queue
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("behavior_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    SKIP_FRAMES = 3  # Process every 3rd frame normally
    MOTION_THRESHOLD = 5000  # Process all frames with significant motion
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    MIN_FACE_CONFIDENCE = 0.7
    PHONE_CONFIDENCE_THRESHOLD = 0.6
    DISTRACTION_THRESHOLD = 3  # Seconds
    BEHAVIOR_CONFIRMATION_FRAMES = 15  # Frames to confirm behavior
    DATABASE_FILE = "behavior_monitor.db"
    LOG_DIR = "behavior_logs"
    MODEL_DIR = "models"
    GPU_ENABLED = torch.cuda.is_available()
    
    # Attention thresholds
    YAWN_THRESHOLD = 0.5  # Adjusted for more reliable mouth detection
    EYE_CLOSED_THRESHOLD = 0.2  # Adjusted for more reliable eye detection
    HEAD_TILT_THRESHOLD = 20  # Degrees
    PHONE_HAND_DISTANCE = 0.15  # Normalized distance
    
    # Visualization
    DISPLAY_WIDTH = 1280
    HEATMAP_ALPHA = 0.3
    TIMELINE_HEIGHT = 100
    
    # Performance
    USE_THREADING = True  # Use threading for parallel processing
    
    @classmethod
    def initialize(cls):
        """Create necessary directories"""
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        
        # Log configuration
        logger.info(f"GPU Enabled: {cls.GPU_ENABLED}")
        logger.info(f"Display Width: {cls.DISPLAY_WIDTH}")
        logger.info(f"Frame Width: {cls.FRAME_WIDTH}")

# Initialize modules with error handling and fallbacks
class Detectors:
    def __init__(self):
        self.object_detector = None
        self.mp_pose = None
        self.mp_face_mesh = None
        self.mp_holistic = None
        self.mp_drawing = mp_lib.solutions.drawing_utils
        self.mp_drawing_styles = mp_lib.solutions.drawing_styles
        self.custom_models = {}
        self.face_mesh_loaded = False
        self.pose_loaded = False

        # Initialize detectors
        self._init_object_detector()
        self._init_media_pipe()
        self.load_custom_models()
        
    def _init_object_detector(self):
        """Initialize YOLO object detector with error handling"""
        try:
            from ultralytics import YOLO
            self.object_detector = YOLO("yolov8n.pt")
            if Config.GPU_ENABLED:
                self.object_detector.to('cuda')
            logger.info("✅ YOLO object detector loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO object detector: {e}")
            logger.info("Using fallback detection methods")
    
    def _init_media_pipe(self):
        """Initialize MediaPipe models with error handling"""
        try:
            # Initialize pose detector
            self.mp_pose = mp_lib.solutions.pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                static_image_mode=False
            )
            self.pose_loaded = True
            logger.info("✅ MediaPipe Pose loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load MediaPipe Pose: {e}")
        
        try:
            # Initialize face mesh
            self.mp_face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_faces=1,
                static_image_mode=False
            )
            self.face_mesh_loaded = True
            logger.info("✅ MediaPipe Face Mesh loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load MediaPipe Face Mesh: {e}")
        
        try:
            # Initialize holistic
            self.mp_holistic = mp_lib.solutions.holistic.Holistic(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            logger.info("✅ MediaPipe Holistic loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load MediaPipe Holistic: {e}")
    
    def detect_faces(self, frame):
        """Detect faces using MediaPipe face mesh as primary method"""
        faces = {}
        
        if not self.face_mesh_loaded:
            # Fallback to OpenCV haar cascade if MediaPipe is not available
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for i, (x, y, w, h) in enumerate(detected_faces):
                    faces[i] = {
                        'facial_area': [x, y, x+w, y+h],
                        'score': 0.8,
                        'landmarks': {}
                    }
                return faces
            except Exception as e:
                logger.error(f"❌ Failed to detect faces with OpenCV: {e}")
                return {}
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            
            results = self.mp_face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extract face bounding box from landmarks
                    x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
                    y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
                    
                    x_min = int(min(x_coordinates) * width)
                    y_min = int(min(y_coordinates) * height)
                    x_max = int(max(x_coordinates) * width)
                    y_max = int(max(y_coordinates) * height)
                    
                    # Define specific facial landmarks
                    landmarks = {}
                    
                    # Extract specific landmark indices based on MediaPipe Face Mesh
                    left_eye_top = (int(face_landmarks.landmark[159].x * width), 
                                   int(face_landmarks.landmark[159].y * height))
                    left_eye_bottom = (int(face_landmarks.landmark[145].x * width), 
                                      int(face_landmarks.landmark[145].y * height))
                    left_eye_left = (int(face_landmarks.landmark[33].x * width), 
                                    int(face_landmarks.landmark[33].y * height))
                    left_eye_right = (int(face_landmarks.landmark[133].x * width), 
                                     int(face_landmarks.landmark[133].y * height))
                    
                    right_eye_top = (int(face_landmarks.landmark[386].x * width), 
                                    int(face_landmarks.landmark[386].y * height))
                    right_eye_bottom = (int(face_landmarks.landmark[374].x * width), 
                                       int(face_landmarks.landmark[374].y * height))
                    right_eye_left = (int(face_landmarks.landmark[362].x * width), 
                                     int(face_landmarks.landmark[362].y * height))
                    right_eye_right = (int(face_landmarks.landmark[263].x * width), 
                                      int(face_landmarks.landmark[263].y * height))
                    
                    mouth_top = (int(face_landmarks.landmark[13].x * width), 
                                int(face_landmarks.landmark[13].y * height))
                    mouth_bottom = (int(face_landmarks.landmark[14].x * width), 
                                   int(face_landmarks.landmark[14].y * height))
                    mouth_left = (int(face_landmarks.landmark[78].x * width), 
                                 int(face_landmarks.landmark[78].y * height))
                    mouth_right = (int(face_landmarks.landmark[308].x * width), 
                                  int(face_landmarks.landmark[308].y * height))
                    
                    landmarks['left_eye_top'] = left_eye_top
                    landmarks['left_eye_bottom'] = left_eye_bottom
                    landmarks['left_eye_left'] = left_eye_left
                    landmarks['left_eye_right'] = left_eye_right
                    
                    landmarks['right_eye_top'] = right_eye_top
                    landmarks['right_eye_bottom'] = right_eye_bottom
                    landmarks['right_eye_left'] = right_eye_left
                    landmarks['right_eye_right'] = right_eye_right
                    
                    landmarks['mouth_top'] = mouth_top
                    landmarks['mouth_bottom'] = mouth_bottom
                    landmarks['mouth_left'] = mouth_left
                    landmarks['mouth_right'] = mouth_right
                    
                    faces[i] = {
                        'facial_area': [x_min, y_min, x_max, y_max],
                        'score': 0.9,
                        'landmarks': landmarks
                    }
            return faces
        except Exception as e:
            logger.error(f"❌ Failed to detect faces with MediaPipe: {e}")
            return {}
    
    def detect_objects(self, frame):
        """Detect objects using YOLO"""
        objects = []
        
        if self.object_detector is None:
            return objects
        
        try:
            if Config.GPU_ENABLED:
                frame_tensor = torch.from_numpy(frame).to('cuda')
                detections = self.object_detector(frame_tensor)
            else:
                detections = self.object_detector(frame)
            
            for det in detections[0].boxes:
                try:
                    objects.append({
                        'class': int(det.cls.item()),
                        'class_name': self.get_class_name(int(det.cls.item())),
                        'box': det.xyxy[0].cpu().numpy(),
                        'confidence': float(det.conf.item())
                    })
                except Exception as e:
                    logger.error(f"Error processing detection: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to detect objects: {e}")
        
        return objects
    
    def get_class_name(self, class_id):
        """Get readable class name from YOLO class ID"""
        coco_names = {
            0: 'person', 67: 'cell phone', 62: 'tv', 63: 'laptop', 
            64: 'mouse', 65: 'remote', 66: 'keyboard', 73: 'book'
        }
        return coco_names.get(class_id, f"class_{class_id}")
    
    def detect_pose(self, frame):
        """Detect human pose using MediaPipe"""
        if not self.pose_loaded:
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_frame)
            return results.pose_landmarks
        except Exception as e:
            logger.error(f"❌ Failed to detect pose: {e}")
            return None

    def detect_gaze(self, frame, face_landmarks):
        """Simple gaze detection using eye landmarks"""
        if not face_landmarks or not face_landmarks.get('left_eye_top'):
            return {
                'left': False,
                'right': False,
                'center': False,
                'blinking': False
            }
            
        try:
            left_eye_vert = np.linalg.norm(
                np.array(face_landmarks['left_eye_top']) - np.array(face_landmarks['left_eye_bottom']))
            left_eye_horiz = np.linalg.norm(
                np.array(face_landmarks['left_eye_left']) - np.array(face_landmarks['left_eye_right']))
                
            right_eye_vert = np.linalg.norm(
                np.array(face_landmarks['right_eye_top']) - np.array(face_landmarks['right_eye_bottom']))
            right_eye_horiz = np.linalg.norm(
                np.array(face_landmarks['right_eye_left']) - np.array(face_landmarks['right_eye_right']))
            
            if left_eye_horiz == 0 or right_eye_horiz == 0:
                return {
                    'left': False,
                    'right': False,
                    'center': True,
                    'blinking': False
                }
                
            left_ratio = left_eye_vert / left_eye_horiz
            right_ratio = right_eye_vert / right_eye_horiz
            
            blinking = (left_ratio < Config.EYE_CLOSED_THRESHOLD and 
                        right_ratio < Config.EYE_CLOSED_THRESHOLD)
            
            left_eye_center_x = (face_landmarks['left_eye_left'][0] + face_landmarks['left_eye_right'][0]) / 2
            right_eye_center_x = (face_landmarks['right_eye_left'][0] + face_landmarks['right_eye_right'][0]) / 2
            
            looking_left = left_eye_center_x < face_landmarks['left_eye_left'][0] + 0.3 * (face_landmarks['left_eye_right'][0] - face_landmarks['left_eye_left'][0])
            looking_right = left_eye_center_x > face_landmarks['left_eye_left'][0] + 0.7 * (face_landmarks['left_eye_right'][0] - face_landmarks['left_eye_left'][0])
            
            looking_center = not (looking_left or looking_right or blinking)
            
            return {
                'left': looking_left,
                'right': looking_right,
                'center': looking_center,
                'blinking': blinking
            }
        except Exception as e:
            logger.error(f"❌ Failed to detect gaze: {e}")
            return {
                'left': False,
                'right': False,
                'center': True,
                'blinking': False
            }
    
    def load_custom_models(self):
        """Load custom YOLO models from model directory"""
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)
            return
            
        try:
            from ultralytics import YOLO
            for model_file in os.listdir(Config.MODEL_DIR):
                if model_file.endswith('.pt'):
                    try:
                        model_name = os.path.splitext(model_file)[0]
                        model_path = os.path.join(Config.MODEL_DIR, model_file)
                        self.custom_models[model_name] = YOLO(model_path)
                        if Config.GPU_ENABLED:
                            self.custom_models[model_name].to('cuda')
                        logger.info(f"✅ Loaded custom model: {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to load model {model_file}: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to import YOLO for custom models: {e}")

# Database handler
class Database:
    def __init__(self):
        try:
            self.conn = sqlite3.connect(Config.DATABASE_FILE)
            self.create_tables()
            logger.info(f"✅ Database initialized: {Config.DATABASE_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            self.conn = sqlite3.connect(":memory:")
            self.create_tables()
            logger.info("Using in-memory database as fallback")
    
    def create_tables(self):
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration INTEGER,
                summary TEXT
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                first_seen DATETIME NOT NULL,
                last_seen DATETIME NOT NULL,
                face_encoding BLOB,
                name TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )''')
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {e}")
    
    def log_event(self, session_id, event_type, event_data=None):
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if event_data and not isinstance(event_data, str):
                event_data = json.dumps(event_data)
            
            cursor.execute('''
            INSERT INTO events (session_id, timestamp, event_type, event_data)
            VALUES (?, ?, ?, ?)
            ''', (session_id, timestamp, event_type, event_data))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"❌ Failed to log event: {e}")
            return None
    
    def start_session(self):
        try:
            cursor = self.conn.cursor()
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
            INSERT INTO sessions (start_time) VALUES (?)
            ''', (start_time,))
            
            self.conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Started new session with ID: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"❌ Failed to start session: {e}")
            return 0
    
    def end_session(self, session_id, summary=None):
        try:
            cursor = self.conn.cursor()
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('SELECT start_time FROM sessions WHERE session_id = ?', (session_id,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Session {session_id} not found")
                return
                
            start_time = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            duration = (datetime.now() - start_time).total_seconds()
            
            cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, duration = ?, summary = ?
            WHERE session_id = ?
            ''', (end_time, duration, summary, session_id))
            
            self.conn.commit()
            logger.info(f"Ended session {session_id} with duration {duration:.1f} seconds")
        except Exception as e:
            logger.error(f"❌ Failed to end session: {e}")
    
    def close(self):
        try:
            self.conn.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"❌ Failed to close database: {e}")

# Attention State Machine
class AttentionState:
    def __init__(self):
        self.state = "INITIALIZING"
        self.state_history = deque(maxlen=100)
        self.state_start_time = time.time()
        self.transitions = {
            "INITIALIZING": self._from_initializing,
            "FOCUSED": self._from_focused,
            "DISTRACTED": self._from_distracted,
            "AWAY": self._from_away,
            "TIRED": self._from_tired
        }
        logger.info("Attention state machine initialized")
    
    def update(self, detection_data):
        handler = self.transitions.get(self.state, self._default_handler)
        new_state = handler(detection_data)
        
        if new_state != self.state:
            now = time.time()
            self.state_history.append((now, self.state, new_state))
            logger.info(f"State transition: {self.state} -> {new_state} (after {now - self.state_start_time:.1f}s)")
            self.state = new_state
            self.state_start_time = now
        
        return self.state
    
    def _from_initializing(self, data):
        if data.get('face_detected', False):
            return "FOCUSED"
        return "INITIALIZING"
    
    def _from_focused(self, data):
        if not data.get('face_detected', False):
            if data.get('away_time', 0) > Config.DISTRACTION_THRESHOLD:
                return "AWAY"
            return "FOCUSED"
        
        if data.get('phone_detected', False):
            return "DISTRACTED"
        
        if data.get('drowsy', False) or data.get('yawn_count', 0) > 2:
            return "TIRED"
        
        if data.get('gaze_direction', 'center') not in ['center']:
            if data.get('gaze_away_time', 0) > Config.DISTRACTION_THRESHOLD:
                return "DISTRACTED"
        
        return "FOCUSED"
    
    def _from_distracted(self, data):
        if not data.get('face_detected', False):
            if data.get('away_time', 0) > Config.DISTRACTION_THRESHOLD:
                return "AWAY"
            return "DISTRACTED"
        
        if data.get('phone_detected', False):
            return "DISTRACTED"
        
        if data.get('focused_time', 0) > Config.DISTRACTION_THRESHOLD:
            return "FOCUSED"
        
        return "DISTRACTED"
    
    def _from_away(self, data):
        if data.get('face_detected', False):
            return "FOCUSED"
        return "AWAY"
    
    def _from_tired(self, data):
        if not data.get('face_detected', False):
            return "AWAY"
        
        if not data.get('drowsy', False) and data.get('yawn_count', 0) == 0:
            if data.get('focused_time', 0) > Config.DISTRACTION_THRESHOLD:
                return "FOCUSED"
        
        return "TIRED"
    
    def _default_handler(self, data):
        return "FOCUSED"

    def get_state_summary(self):
        """Get summary of time spent in each state"""
        if not self.state_history:
            return {"INITIALIZING": 0, "FOCUSED": 0, "DISTRACTED": 0, "AWAY": 0, "TIRED": 0}
            
        summary = {"INITIALIZING": 0, "FOCUSED": 0, "DISTRACTED": 0, "AWAY": 0, "TIRED": 0}
        
        summary[self.state] = time.time() - self.state_start_time
        
        for i in range(len(self.state_history)):
            current_time, state, new_state = self.state_history[i]
            
            if i == 0:
                continue
            else:
                prev_time = self.state_history[i-1][0]
                duration = current_time - prev_time
                summary[state] += duration
                
        return summary

# Frame processor class
class FrameProcessor(Thread):
    def __init__(self, detectors, frame_queue, result_queue):
        super().__init__(daemon=True)
        self.detectors = detectors
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True
        self.frame_count = 0
        self.last_frame = None
        self.motion_level = 0
        logger.info("Frame processor initialized")
    
    def run(self):
        """Frame processing worker running in separate thread"""
        logger.info("Frame processor started")
        
        while self.running:
            try:
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                except:
                    continue
                    
                if frame_data is None:
                    self.running = False
                    break
                    
                frame, frame_time = frame_data
                
                if self.last_frame is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(gray, last_gray)
                        self.motion_level = np.sum(diff) / (frame.shape[0] * frame.shape[1])
                    except Exception as e:
                        logger.error(f"Error in motion detection: {e}")
                        self.motion_level = 0
                
                if self.motion_level > Config.MOTION_THRESHOLD or self.frame_count % Config.SKIP_FRAMES == 0:
                    result = self._process_frame(frame, frame_time)
                    self.result_queue.put(result)
                
                self.last_frame = frame.copy()
                self.frame_count += 1
                
            except Exception as e:
                logger.error(f"Error in frame processor: {e}")
        
        logger.info("Frame processor stopped")
    
    def _process_frame(self, frame, frame_time):
        """Process a single frame for all detections"""
        results = {
            'faces': {},
            'objects': [],
            'pose': None,
            'gaze': None,
            'timestamp': frame_time,
            'motion_level': self.motion_level
        }
        
        try:
            results['faces'] = self.detectors.detect_faces(frame)
            results['objects'] = self.detectors.detect_objects(frame)
            results['pose'] = self.detectors.detect_pose(frame)
            
            if results['faces'] and len(results['faces']) > 0:
                first_face_key = list(results['faces'].keys())[0]
                first_face = results['faces'][first_face_key]
                results['gaze'] = self.detectors.detect_gaze(frame, first_face.get('landmarks', {}))
            else:
                results['gaze'] = {
                    'left': False,
                    'right': False,
                    'center': False,
                    'blinking': False
                }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
        
        return results
    
    def stop(self):
        """Stop the processor thread"""
        self.running = False

# Main Application
class BehaviorMonitor:
    def __init__(self):
        Config.initialize()
        self.detectors = Detectors()
        self.database = Database()
        self.session_id = self.database.start_session()
        self.attention_state = AttentionState()
        
        self.frame_count = 0
        self.last_face_time = time.time()
        self.last_focused_time = time.time()
        self.last_gaze_center_time = time.time()
        self.yawn_count = 0
        self.drowsy_frames = 0
        
        self.face_tracks = {}
        self.next_face_id = 1
        
        self.heatmap = None
        self.timeline = None
        
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        self.csv_log_file = os.path.join(Config.LOG_DIR, f"behavior_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.csv_log_file, 'w') as f:
            f.write("timestamp,face_id,attention_state,gaze_direction,posture_status,behavior,phone_detected\n")
        
        if Config.USE_THREADING:
            self.frame_queue = Queue(maxsize=5)
            self.result_queue = Queue(maxsize=5)
            self.processor = FrameProcessor(self.detectors, self.frame_queue, self.result_queue)
            self.processor.start()
        else:
            self.frame_queue = mp.Queue(maxsize=5)
            self.result_queue = mp.Queue(maxsize=5)
            self.process = mp.Process(
                target=self._process_frames_mp,
                args=(self.frame_queue, self.result_queue)
            )
            self.process.daemon = True
            self.process.start()
        
        logger.info("BehaviorMonitor initialized")
    
    def _process_frames_mp(self, frame_queue, result_queue):
        """Multiprocessing frame processor function"""
        processor = FrameProcessor(Detectors(), frame_queue, result_queue)
        processor.run()

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2
        
        xi1 = max(x1, x1_t)
        yi1 = max(y1, y1_t)
        xi2 = min(x2, x2_t)
        yi2 = min(y2, y2_t)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_t - x1_t) * (y2_t - y1_t)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update_tracking(self, detections):
        """Update face tracks with new detections"""
        current_time = time.time()
        matched_tracks = set()
        updated_tracks = {}
        
        for face_id, face in detections.get('faces', {}).items():
            best_match = None
            best_iou = 0.4
            
            for track_id, track in self.face_tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self.calculate_iou(face['facial_area'], track['last_box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = track_id
            
            if best_match is not None:
                updated_tracks[best_match] = {
                    'last_box': face['facial_area'],
                    'last_seen': current_time,
                    'landmarks': face.get('landmarks', {}),
                    'score': face.get('score', 0.9),
                    'name': self.face_tracks[best_match].get('name', f'Person_{best_match}')
                }
                matched_tracks.add(best_match)
            else:
                # Create new track
                new_id = self.next_face_id
                self.next_face_id += 1
                updated_tracks[new_id] = {
                    'last_box': face['facial_area'],
                    'last_seen': current_time,
                    'landmarks': face.get('landmarks', {}),
                    'score': face.get('score', 0.9),
                    'name': f'Person_{new_id}',
                    'first_seen': current_time
                }
                # Log new face
                self.database.log_event(
                    self.session_id,
                    'new_face',
                    {'face_id': new_id, 'first_seen': current_time}
                )

        # Remove lost tracks
        for track_id in list(self.face_tracks.keys()):
            if track_id not in matched_tracks:
                if current_time - self.face_tracks[track_id]['last_seen'] > 5.0:  # 5 second timeout
                    del self.face_tracks[track_id]
                    # Log lost face
                    self.database.log_event(
                        self.session_id,
                        'lost_face',
                        {'face_id': track_id}
                    )

        self.face_tracks.update(updated_tracks)

    def analyze_attention(self, detections):
        """Analyze attention based on all available data"""
        current_time = time.time()
        analysis = {
            'face_detected': len(detections.get('faces', {})) > 0,
            'phone_detected': False,
            'drowsy': False,
            'yawn_count': self.yawn_count,
            'away_time': current_time - self.last_face_time,
            'focused_time': current_time - self.last_focused_time,
            'gaze_direction': 'unknown',
            'gaze_away_time': 0
        }

        # Check for phones
        for obj in detections.get('objects', []):
            if obj['class_name'] == 'cell phone' and obj['confidence'] > Config.PHONE_CONFIDENCE_THRESHOLD:
                analysis['phone_detected'] = True
                break

        # Check face states
        if analysis['face_detected']:
            self.last_face_time = current_time
            
            for face_id, face in detections['faces'].items():
                # Check for yawning
                if self._is_mouth_open(face.get('landmarks', {})):
                    self.yawn_count += 1
                else:
                    self.yawn_count = max(0, self.yawn_count - 1)
                
                # Check for drowsiness
                if self._are_eyes_closed(face.get('landmarks', {})):
                    self.drowsy_frames += 1
                    if self.drowsy_frames > Config.BEHAVIOR_CONFIRMATION_FRAMES:
                        analysis['drowsy'] = True
                else:
                    self.drowsy_frames = max(0, self.drowsy_frames - 1)
            
            # Check gaze direction
            if detections.get('gaze'):
                gaze = detections['gaze']
                if gaze['blinking']:
                    analysis['gaze_direction'] = 'blinking'
                elif gaze['left']:
                    analysis['gaze_direction'] = 'left'
                elif gaze['right']:
                    analysis['gaze_direction'] = 'right'
                elif gaze['center']:
                    analysis['gaze_direction'] = 'center'
                    self.last_gaze_center_time = current_time
                
                analysis['gaze_away_time'] = current_time - self.last_gaze_center_time

        # Update attention state
        state = self.attention_state.update(analysis)
        
        # Log state changes
        if hasattr(self, 'last_state') and state != self.last_state:
            self.database.log_event(
                self.session_id,
                'state_change',
                {'from': self.last_state, 'to': state}
            )
            self._log_to_csv(current_time, state, analysis)
        
        self.last_state = state
        return state, analysis

    def _is_mouth_open(self, landmarks):
        """Check if mouth is open using mouth aspect ratio"""
        if not landmarks or not all(k in landmarks for k in ['mouth_top', 'mouth_bottom', 'mouth_left', 'mouth_right']):
            return False
            
        try:
            vertical = np.linalg.norm(
                np.array(landmarks['mouth_top']) - np.array(landmarks['mouth_bottom']))
            horizontal = np.linalg.norm(
                np.array(landmarks['mouth_left']) - np.array(landmarks['mouth_right']))
            
            if horizontal == 0:
                return False
                
            ratio = vertical / horizontal
            return ratio > Config.YAWN_THRESHOLD
        except Exception as e:
            logger.error(f"Error calculating mouth openness: {e}")
            return False

    def _are_eyes_closed(self, landmarks):
        """Check if eyes are closed using eye aspect ratio"""
        if not landmarks or not all(k in landmarks for k in ['left_eye_top', 'left_eye_bottom', 'right_eye_top', 'right_eye_bottom']):
            return False
            
        try:
            left_eye_vert = np.linalg.norm(
                np.array(landmarks['left_eye_top']) - np.array(landmarks['left_eye_bottom']))
            left_eye_horiz = np.linalg.norm(
                np.array(landmarks['left_eye_left']) - np.array(landmarks['left_eye_right']))
            
            right_eye_vert = np.linalg.norm(
                np.array(landmarks['right_eye_top']) - np.array(landmarks['right_eye_bottom']))
            right_eye_horiz = np.linalg.norm(
                np.array(landmarks['right_eye_left']) - np.array(landmarks['right_eye_right']))
            
            if left_eye_horiz == 0 or right_eye_horiz == 0:
                return False
                
            left_ratio = left_eye_vert / left_eye_horiz
            right_ratio = right_eye_vert / right_eye_horiz
            
            return (left_ratio < Config.EYE_CLOSED_THRESHOLD and 
                    right_ratio < Config.EYE_CLOSED_THRESHOLD)
        except Exception as e:
            logger.error(f"Error calculating eye closure: {e}")
            return False

    def _log_to_csv(self, timestamp, state, analysis):
        """Log data to CSV file"""
        try:
            with open(self.csv_log_file, 'a') as f:
                for face_id, face in self.face_tracks.items():
                    f.write(f"{datetime.fromtimestamp(timestamp).isoformat()},")
                    f.write(f"{face_id},")
                    f.write(f"{state},")
                    f.write(f"{analysis.get('gaze_direction', 'unknown')},")
                    f.write(f"{'Good' if analysis.get('posture_good', True) else 'Bad'},")
                    f.write(f"{'Drowsy' if analysis.get('drowsy', False) else 'Alert'},")
                    f.write(f"{'Yes' if analysis.get('phone_detected', False) else 'No'}\n")
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")

    def visualize(self, frame, detections, state):
        """Visualize all detections and state information"""
        # Resize frame for display
        display_frame = cv2.resize(frame, (Config.DISPLAY_WIDTH, 
                                         int(Config.DISPLAY_WIDTH * frame.shape[0] / frame.shape[1])))
        
        # Draw face tracks
        for track_id, track in self.face_tracks.items():
            box = track['last_box']
            # Scale box to display size
            box = [
                int(box[0] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[1] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[2] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[3] * Config.DISPLAY_WIDTH / frame.shape[1])
            ]
            
            color = (0, 255, 0)  # Green for normal
            if state == "DISTRACTED":
                color = (0, 165, 255)  # Orange
            elif state == "TIRED":
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw face info
            info_text = f"ID: {track_id}"
            if track.get('name'):
                info_text += f" ({track['name']})"
            
            cv2.putText(display_frame, info_text, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw objects
        for obj in detections.get('objects', []):
            box = obj['box']
            # Scale box to display size
            box = [
                int(box[0] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[1] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[2] * Config.DISPLAY_WIDTH / frame.shape[1]),
                int(box[3] * Config.DISPLAY_WIDTH / frame.shape[1])
            ]
            
            color = (0, 0, 255)  # Red for objects
            label = f"{obj['class_name']} ({obj['confidence']:.2f})"
            
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(display_frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw pose
        if detections.get('pose'):
            self.detectors.mp_drawing.draw_landmarks(
                display_frame,
                detections['pose'],
                self.detectors.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.detectors.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw state information
        state_colors = {
            "FOCUSED": (0, 255, 0),
            "DISTRACTED": (0, 165, 255),
            "TIRED": (0, 0, 255),
            "AWAY": (255, 255, 255),
            "INITIALIZING": (128, 128, 128)
        }
        
        color = state_colors.get(state, (255, 255, 255))
        cv2.putText(display_frame, f"State: {state}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw FPS
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame

    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Error: Could not open video source")
            return
        
        cv2.namedWindow("Behavior Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Behavior Monitor", Config.DISPLAY_WIDTH, 
                         int(Config.DISPLAY_WIDTH * 9/16))
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Error: Could not read frame")
                    break
                
                # Send frame to processing
                if Config.USE_THREADING:
                    if self.frame_queue.qsize() < 2:  # Don't overload the queue
                        self.frame_queue.put((frame, start_time))
                else:
                    if self.frame_queue.empty():  # MP queue blocks when full
                        self.frame_queue.put((frame, start_time))
                
                # Get results if available
                if not self.result_queue.empty():
                    results = self.result_queue.get()
                    
                    # Update tracking
                    self.update_tracking(results)
                    
                    # Analyze attention
                    state, analysis = self.analyze_attention(results)
                    
                    # Visualize
                    display_frame = self.visualize(frame, results, state)
                    cv2.imshow("Behavior Monitor", display_frame)
                
                # Calculate FPS
                self.frame_times.append(time.time())
                if len(self.frame_times) > 30:
                    self.frame_times.popleft()
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Signal processing thread to exit
            if Config.USE_THREADING:
                self.processor.stop()
                self.frame_queue.put(None)
            else:
                self.frame_queue.put(None)
                self.process.join()
            
            # Close database
            summary = self.attention_state.get_state_summary()
            self.database.end_session(self.session_id, json.dumps(summary))
            self.database.close()
            
            logger.info("✅ Application shutdown complete")

if __name__ == "__main__":
    monitor = BehaviorMonitor()
    monitor.run()