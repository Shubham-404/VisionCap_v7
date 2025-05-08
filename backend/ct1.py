import cv2
import time
import numpy as np
from datetime import datetime
import os
import threading
import csv
import argparse
from collections import defaultdict

# Import detection and tracking modules with error handling
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from gaze_tracking import GazeTracking
    gaze = GazeTracking()
    GAZE_AVAILABLE = True
except ImportError:
    GAZE_AVAILABLE = False
    gaze = None

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    pose = None
    face_mesh = None

try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO_MODEL = None

# Constants
SKIP_FRAMES = 3  # Process every 3rd frame for performance
FRAME_WIDTH = 1280
YAWN_THRESHOLD = 25
EYE_CLOSED_THRESHOLD = 0.01
EYE_DISTANCE_THRESHOLD = 30
EYE_CLOSED_FRAMES = 10
DISTRACTION_THRESHOLD = 3  # Seconds
PHONE_CONFIDENCE_THRESHOLD = 0.5
BEHAVIOR_THRESHOLD = 15  # Frames to confirm behavior
MOUTH_OPEN_THRESHOLD = 0.03
FACE_TRACKING_THRESHOLD = 0.6  # IoU threshold for face tracking
MAX_FACE_DISTANCE = 0.6  # Max distance for face recognition matches

# Create directories for logs
LOG_DIR = "behavior_logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"behavior_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

class FaceTracker:
    """Tracks faces across frames using IoU and face recognition"""
    def __init__(self):
        self.faces = {}
        self.next_id = 1
        self.known_encodings = []
        self.known_names = []
    
    def load_known_faces(self, directory="known_faces"):
        if not FACE_RECOGNITION_AVAILABLE:
            return
        
        for filename in os.listdir(directory):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                try:
                    path = os.path.join(directory, filename)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    print(f"Error loading face {filename}: {e}")
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / float(box1_area + box2_area - inter_area)
    
    def update_faces(self, current_faces):
        """Update tracked faces with new detections"""
        updated_faces = {}
        used_ids = set()
        
        # First try to match existing faces with new detections
        for face_key, new_face in current_faces.items():
            best_match_id = None
            best_iou = FACE_TRACKING_THRESHOLD
            
            for face_id, old_face in self.faces.items():
                iou = self._calculate_iou(new_face['box'], old_face['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing face
                updated_face = new_face.copy()
                updated_face['id'] = best_match_id
                updated_face['name'] = self.faces[best_match_id].get('name', 'Unknown')
                updated_faces[best_match_id] = updated_face
                used_ids.add(best_match_id)
            else:
                # New face - assign ID and try to recognize
                face_id = self.next_id
                self.next_id += 1
                updated_face = new_face.copy()
                updated_face['id'] = face_id
                
                # Face recognition
                if FACE_RECOGNITION_AVAILABLE and 'encoding' in new_face:
                    matches = face_recognition.compare_faces(
                        self.known_encodings, new_face['encoding'], tolerance=0.5
                    )
                    if True in matches:
                        face_distances = face_recognition.face_distance(
                            self.known_encodings, new_face['encoding']
                        )
                        best_match_idx = np.argmin(face_distances)
                        if matches[best_match_idx] and face_distances[best_match_idx] < MAX_FACE_DISTANCE:
                            updated_face['name'] = self.known_names[best_match_idx]
                
                updated_faces[face_id] = updated_face
                used_ids.add(face_id)
        
        # Carry forward any untracked faces for a few frames
        for face_id, old_face in self.faces.items():
            if face_id not in used_ids and old_face.get('frames_since_seen', 0) < 5:
                updated_face = old_face.copy()
                updated_face['frames_since_seen'] = old_face.get('frames_since_seen', 0) + 1
                updated_faces[face_id] = updated_face
        
        self.faces = updated_faces
        return self.faces

class BehaviorAnalyzer:
    """Analyzes behavior patterns over time"""
    def __init__(self):
        self.face_behaviors = defaultdict(lambda: {
            'sleep_counter': 0,
            'talk_counter': 0,
            'phone_counter': 0,
            'drowsy_counter': 0,
            'distracted_counter': 0
        })
    
    def update_behavior(self, face_id, behavior_type, increment=True):
        """Update behavior counters for a face"""
        if increment:
            self.face_behaviors[face_id][f'{behavior_type}_counter'] += 1
        else:
            self.face_behaviors[face_id][f'{behavior_type}_counter'] = max(
                0, self.face_behaviors[face_id][f'{behavior_type}_counter'] - 1
            )
        
        # Determine current behavior based on counters
        if self.face_behaviors[face_id]['sleep_counter'] > BEHAVIOR_THRESHOLD:
            return "Sleeping"
        elif self.face_behaviors[face_id]['phone_counter'] > BEHAVIOR_THRESHOLD:
            return "Using Phone"
        elif self.face_behaviors[face_id]['talk_counter'] > BEHAVIOR_THRESHOLD:
            return "Talking"
        elif self.face_behaviors[face_id]['drowsy_counter'] > BEHAVIOR_THRESHOLD:
            return "Drowsy"
        elif self.face_behaviors[face_id]['distracted_counter'] > BEHAVIOR_THRESHOLD:
            return "Distracted"
        else:
            return "Attentive"

class State:
    """Maintains the current state of the monitoring system"""
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_phone_detected = None
        self.attention_states = {}
        self.fps = 0
        self.phone_detected = False
        self.phone_box = None
        self.phone_confidence = 0
        self.gaze_status = "Unknown"
        self.posture_status = "Unknown"
        self.face_tracker = FaceTracker()
        self.behavior_analyzer = BehaviorAnalyzer()

def process_frame(frame, state):
    """Process a single frame for face, behavior, and phone detection"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_faces = {}
    
    # Face detection with RetinaFace or fallback
    if RETINAFACE_AVAILABLE:
        try:
            faces = RetinaFace.detect_faces(frame)
            if isinstance(faces, dict):
                for idx, key in enumerate(faces):
                    identity = faces[key]
                    facial_area = identity["facial_area"]
                    landmarks = identity.get("landmarks", {})
                    
                    x1, y1, x2, y2 = facial_area
                    face_img = frame[y1:y2, x1:x2]
                    
                    face_data = {
                        'box': (x1, y1, x2, y2),
                        'landmarks': landmarks,
                        'yawning': is_mouth_open(landmarks),
                        'drowsy': False,
                        'sleeping': False,
                        'behavior': "Attentive",
                        'emotion': "Unknown",
                        'frames_since_seen': 0
                    }
                    
                    # Face recognition encoding
                    if FACE_RECOGNITION_AVAILABLE and face_img.size > 0:
                        try:
                            face_encoding = face_recognition.face_encodings(face_img)
                            if face_encoding:
                                face_data['encoding'] = face_encoding[0]
                        except Exception as e:
                            print(f"Face encoding error: {e}")
                    
                    # Emotion detection
                    if DEEPFACE_AVAILABLE and face_img.size > 0:
                        try:
                            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                            if isinstance(analysis, list):
                                face_data['emotion'] = analysis[0]['dominant_emotion']
                        except Exception as e:
                            print(f"Emotion detection error: {e}")
                    
                    current_faces[key] = face_data
        except Exception as e:
            print(f"Face detection error: {e}")
    
    # Update face tracking
    tracked_faces = state.face_tracker.update_faces(current_faces)
    
    # Phone detection
    phone_detected, phone_box, phone_confidence = detect_phone(frame)
    state.phone_detected = phone_detected
    state.phone_box = phone_box
    state.phone_confidence = phone_confidence
    
    # MediaPipe processing for posture and gaze
    if MEDIAPIPE_AVAILABLE:
        # Pose estimation
        pose_result = pose.process(rgb)
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            state.posture_status = "Bad Posture" if shoulder_diff > 0.1 else "Good Posture"
        
        # Face mesh for eye tracking
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                # Eye closure detection
                left_eye_top = face_landmarks.landmark[159]
                left_eye_bottom = face_landmarks.landmark[145]
                eyes_closed = abs(left_eye_top.y - left_eye_bottom.y) < EYE_CLOSED_THRESHOLD
                
                # Update face data if we can match with detected faces
                for face_id, face_data in tracked_faces.items():
                    face_box = face_data['box']
                    face_center = ((face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2)
                    
                    # Simple heuristic to match face mesh with detected face
                    mesh_nose = face_landmarks.landmark[1]
                    mesh_x = mesh_nose.x * frame.shape[1]
                    mesh_y = mesh_nose.y * frame.shape[0]
                    
                    if (abs(mesh_x - face_center[0]) < 100 and abs(mesh_y - face_center[1]) < 100):
                        face_data['eyes_closed'] = eyes_closed
                        face_data['talking'] = is_mouth_open(face_landmarks)
    
    # Gaze tracking (global for now)
    if GAZE_AVAILABLE:
        gaze.refresh(frame)
        if gaze.is_blinking():
            state.gaze_status = "Blinking"
        elif gaze.is_right():
            state.gaze_status = "Looking right"
        elif gaze.is_left():
            state.gaze_status = "Looking left"
        elif gaze.is_center():
            state.gaze_status = "Looking center"
        else:
            state.gaze_status = "Gaze undetected"
    
    # Update behavior states for each face
    for face_id, face_data in tracked_faces.items():
        # Update behavior counters
        if face_data.get('eyes_closed', False):
            state.behavior_analyzer.update_behavior(face_id, 'sleep')
        else:
            state.behavior_analyzer.update_behavior(face_id, 'sleep', False)
        
        if face_data.get('talking', False):
            state.behavior_analyzer.update_behavior(face_id, 'talk')
        else:
            state.behavior_analyzer.update_behavior(face_id, 'talk', False)
        
        if face_data.get('yawning', False):
            state.behavior_analyzer.update_behavior(face_id, 'drowsy')
        else:
            state.behavior_analyzer.update_behavior(face_id, 'drowsy', False)
        
        # Phone usage detection
        if state.phone_detected and state.phone_box and 'box' in face_data:
            if detect_phone_use(face_data['box'], state.phone_box):
                state.behavior_analyzer.update_behavior(face_id, 'phone')
            else:
                state.behavior_analyzer.update_behavior(face_id, 'phone', False)
        
        # Get current behavior state
        face_data['behavior'] = state.behavior_analyzer.update_behavior(face_id, '')
        
        # Determine attention state for this face
        attention_state, color = analyze_attention_state(face_data, state)
        face_data['attention_state'] = attention_state
        face_data['attention_color'] = color
    
    return tracked_faces

def analyze_attention_state(face_data, state):
    """Determine attention state for a single face"""
    if face_data.get('sleeping', False):
        return "SLEEPING", (0, 0, 255)
    
    if face_data.get('behavior', "") == "Using Phone":
        return "DISTRACTED: Phone", (0, 0, 255)
    
    if face_data.get('yawning', False):
        return "TIRED: Yawning", (0, 165, 255)
    
    if face_data.get('drowsy', False):
        return "TIRED: Drowsy", (0, 0, 255)
    
    if face_data.get('behavior', "") == "Talking":
        return "TALKING", (0, 255, 255)
    
    if state.posture_status == "Bad Posture":
        return "DISTRACTED: Posture", (0, 165, 255)
    
    if state.gaze_status in ["Looking left", "Looking right", "Blinking"]:
        return f"DISTRACTED: {state.gaze_status}", (0, 165, 255)
    
    emotion = face_data.get('emotion', 'Unknown')
    if emotion in ["sad", "angry", "fear"]:
        return f"CONCERNED: {emotion}", (0, 165, 255)
    
    return "FOCUSED", (0, 255, 0)

def draw_face_info(frame, face_data):
    """Draw face information on the frame"""
    box = face_data.get('box', None)
    if not box:
        return
    
    x1, y1, x2, y2 = [int(coord) for coord in box]
    name = face_data.get('name', 'Unknown')
    behavior = face_data.get('behavior', 'Unknown')
    emotion = face_data.get('emotion', 'Unknown')
    attention_state = face_data.get('attention_state', 'Unknown')
    color = face_data.get('attention_color', (0, 255, 0))
    
    # Draw face box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw info box above face
    info_text = f"{name}: {attention_state}"
    (text_width, text_height), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, info_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw behavior and emotion below face
    detail_text = f"{behavior} | {emotion}"
    cv2.putText(frame, detail_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main(video_path):
    state = State()
    state.face_tracker.load_known_faces()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Create window
    cv2.namedWindow("Multi-Face Behavior Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Face Behavior Monitor", FRAME_WIDTH, int(FRAME_WIDTH * 9/16))
    
    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write("timestamp,face_id,name,attention_state,behavior,emotion,phone_detected\n")
    
    frame_count = 0
    last_log_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            continue
        
        # Process frame
        start_time = time.time()
        tracked_faces = process_frame(frame, state)
        processing_time = time.time() - start_time
        
        # Calculate FPS
        if state.frame_count == 0:
            state.start_time = time.time()
        state.frame_count += 1
        elapsed_time = time.time() - state.start_time
        if elapsed_time > 1.0:
            state.fps = state.frame_count / elapsed_time
            state.frame_count = 0
            state.start_time = time.time()
        
        # Draw all face information
        for face_id, face_data in tracked_faces.items():
            draw_face_info(frame, face_data)
        
        # Draw phone box if detected
        if state.phone_detected and state.phone_box:
            x1, y1, x2, y2 = [int(coord) for coord in state.phone_box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Phone {state.phone_confidence:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Draw system info
        info_y = 30
        cv2.putText(frame, f"FPS: {state.fps:.1f} | Process: {processing_time*1000:.1f}ms", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(frame, f"Posture: {state.posture_status}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(frame, f"Gaze: {state.gaze_status}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Multi-Face Behavior Monitor", frame)
        
        # Log data periodically
        current_time = time.time()
        if current_time - last_log_time > 5:  # Log every 5 seconds
            for face_id, face_data in tracked_faces.items():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_FILE, 'a') as f:
                    f.write(f"{timestamp},{face_id},{face_data.get('name', 'Unknown')}," +
                            f"{face_data.get('attention_state', 'Unknown')}," +
                            f"{face_data.get('behavior', 'Unknown')}," +
                            f"{face_data.get('emotion', 'Unknown')}," +
                            f"{state.phone_detected}\n")
            last_log_time = current_time
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Face Behavior Monitoring from Video')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    args = parser.parse_args()
    
    main(args.video_path)