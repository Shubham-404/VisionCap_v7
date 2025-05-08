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
                        'score': 0.8,  # Default confidence for haar cascade
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
                    # Eyes
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
                    
                    # Mouth
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
                        'score': 0.9,  # Default confidence for MediaPipe
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
        # COCO dataset class names
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
            # Convert to RGB for MediaPipe
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
            # Calculate eye aspect ratios
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
            
            # Determine blinking
            blinking = (left_ratio < Config.EYE_CLOSED_THRESHOLD and 
                        right_ratio < Config.EYE_CLOSED_THRESHOLD)
            
            # Calculate pupil positions (simplified)
            left_eye_center_x = (face_landmarks['left_eye_left'][0] + face_landmarks['left_eye_right'][0]) / 2
            right_eye_center_x = (face_landmarks['right_eye_left'][0] + face_landmarks['right_eye_right'][0]) / 2
            
            face_center_x = (face_landmarks['left_eye_center_x'] + face_landmarks['right_eye_center_x']) / 2 if \
                'left_eye_center_x' in face_landmarks and 'right_eye_center_x' in face_landmarks else \
                (left_eye_center_x + right_eye_center_x) / 2
            
            # Simple left/right gaze detection
            looking_left = left_eye_center_x < face_landmarks['left_eye_left'][0] + 0.3 * left_eye_horiz and \
                          right_eye_center_x < face_landmarks['right_eye_left'][0] + 0.3 * right_eye_horiz
            
            looking_right = left_eye_center_x > face_landmarks['left_eye_left'][0] + 0.7 * left_eye_horiz and \
                           right_eye_center_x > face_landmarks['right_eye_left'][0] + 0.7 * right_eye_horiz
            
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
            # Create fallback in-memory database
            self.conn = sqlite3.connect(":memory:")
            self.create_tables()
            logger.info("Using in-memory database as fallback")
    
    def create_tables(self):
        try:
            cursor = self.conn.cursor()
            
            # Sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration INTEGER,
                summary TEXT
            )''')
            
            # Events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )''')
            
            # Faces table
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
            
            # Calculate duration
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
        
        # Add current state duration
        summary[self.state] = time.time() - self.state_start_time
        
        # Add previous state durations
        for i in range(len(self.state_history)):
            current_time, state, new_state = self.state_history[i]
            
            # Find the start time of this state
            if i == 0:
                # For the first recorded state, we don't know when it started
                # So we'll just count the transition
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
                # Get frame data from queue with timeout
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                except:
                    continue
                    
                if frame_data is None:  # Termination signal
                    self.running = False
                    break
                    
                frame, frame_time = frame_data
                
                # Motion detection
                if self.last_frame is not None:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                        diff = cv2.absdiff(gray, last_gray)
                        self.motion_level = np.sum(diff) / (frame.shape[0] * frame.shape[1])
                    except Exception as e:
                        logger.error(f"Error in motion detection: {e}")
                        self.motion_level = 0
                
                # Only process if significant motion or every SKIP_FRAMES
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
            # Face detection
            results['faces'] = self.detectors.detect_faces(frame)
            
            # Object detection
            results['objects'] = self.detectors.detect_objects(frame)
            
            # Pose estimation
            results['pose'] = self.detectors.detect_pose(frame)
            
            # Gaze estimation
            if results['faces'] and len(results['faces']) > 0:
                # Get the first face's landmarks
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
        # Initialize configuration
        Config.initialize()
        
        # Initialize components
        self.detectors = Detectors()
        self.database = Database()
        self.session_id = self.database.start_session()
        self.attention_state = AttentionState()
        
        # State variables
        self.frame_count = 0
        self.last_face_time = time.time()
        self.last_focused_time = time.time()
        self.last_gaze_center_time = time.time()
        self.yawn_count = 0
        self.drowsy_frames = 0
        
        # Face tracking
        self.face_tracks = {}
        self.next_face_id = 1
        
        # Visualization
        self.heatmap = None
        self.timeline = None
        
        # FPS calculation
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Initialize processing queues and thread
        self.frame_queue = mp.Queue(maxsize=5) if not Config.USE_THREADING else None
        self.result_queue = mp.Queue(maxsize=5) if not Config.USE_THREADING else None
        
        if Config.USE_THREADING:
            from queue import Queue
            self.frame_queue = Queue(maxsize=5)
            self.result_queue = Queue(maxsize=5)
            self.processor = Frame