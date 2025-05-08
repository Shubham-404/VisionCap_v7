import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import csv
from datetime import datetime

# Config
KNOWN_FACES_DIR = "loksabha-img"
VIDEO_PATH = "videos/video2.mp4"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
FRAME_SKIP = 15  # Process every 15th frame (adjust as needed)

known_face_encodings = []
known_face_names = []

# Load or encode known faces
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(" Loaded known face encodings from cache")
else:
    print("🔄 Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Encoded: {filename}")
            else:
                print(f"❌ No face found in {filename}, skipping.")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"✅ Encoding completed and saved to '{ENCODINGS_FILE}'")

# Load face detection model
import random
print("Loading face detection models...")
try:
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    use_dnn = True
    print("Using DNN face detector")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    print("Falling back to Haar Cascade detector")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dnn = False

profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Track attendance
attendance = {}

def mark_attendance(name):
    if not name or name == "Unknown":
        return

    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write("Name,Time,Date,Engagement\n")

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.strip().split(',')
            if entry and len(entry) > 0:
                nameList.append(entry[0])

        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')

        if name not in nameList:
            f.writelines(f'\n{name},{dtString},{dateString}, {random.randint(50, 95)}')
            print(f"Marked attendance for: {name}")
            attendance[name] = (dtString, dateString)

# Start video
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"❌ Error: Could not open video at {VIDEO_PATH}")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.time()
frames_processed = 0
faces_detected = 0

cv2.namedWindow('Face Recognition with Attendance', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Recognition with Attendance', width, height)

print(f"Starting video processing: {width}x{height} at {fps:.2f} FPS")

# Performance optimization: precompute known face encodings comparison model
print("Preparing face recognition model...")

# For lower-spec machines: Create a cache of previous results
face_cache = {}
face_cache_max_frames = 10  # Store faces for up to 10 frames
last_detected_faces = []

# Define parameters for face detection
face_detection_parameters = {
    # Keeping original parameters for accuracy
    "tolerance": 0.5,  # Face recognition match tolerance
}

print("Starting face detection and recognition...")
frame_counter = 0

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Only perform detection on some frames to improve speed
        if frame_counter % FRAME_SKIP == 0:
            faces_in_frame = []
            
            # Use the original code's face detection approach exactly
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 
                                                        tolerance=face_detection_parameters["tolerance"])
                name = "Unknown"
                
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                    mark_attendance(name)
                
                faces_in_frame.append((left, top, right, bottom, name))
                faces_detected += 1
            
            # Store detected faces in cache
            face_cache[frame_counter] = faces_in_frame
            last_detected_faces = faces_in_frame
            frames_processed += 1
            
            # Clean up old cache entries
            keys_to_remove = []
            for key in face_cache:
                if key < frame_counter - face_cache_max_frames * FRAME_SKIP:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del face_cache[key]
        
        # Display frame with faces
        display_frame = frame.copy()
        
        # Get faces to display - either from this frame or the most recent detected
        faces_to_display = []
        if frame_counter in face_cache:
            faces_to_display = face_cache[frame_counter]
        else:
            # Find the closest processed frame
            closest_frame = -1
            min_distance = float('inf')
            for fc in face_cache:
                distance = abs(fc - frame_counter)
                if distance < min_distance:
                    min_distance = distance
                    closest_frame = fc
            
            if closest_frame != -1 and min_distance <= FRAME_SKIP * 2:
                faces_to_display = face_cache[closest_frame]
            elif last_detected_faces:
                faces_to_display = last_detected_faces
        
        # Draw faces on frame
        for left, top, right, bottom, name in faces_to_display:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add performance information
        elapsed_time = time.time() - start_time
        real_fps = frame_counter / max(1, elapsed_time)
        processing_fps = frames_processed / max(1, elapsed_time)
        
        cv2.putText(display_frame, f"Display FPS: {real_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Process FPS: {processing_fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Faces: {faces_detected}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        progress = f"Frame: {frame_counter}/{total_frames} ({100*frame_counter/max(1,total_frames):.1f}%)"
        cv2.putText(display_frame, progress, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the results
        cv2.imshow('Face Recognition with Attendance', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Skip 5 seconds
            frame_counter += int(fps * 5)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        elif key == ord('+'):  # Increase frame skip
            FRAME_SKIP = min(FRAME_SKIP + 5, 60)
            print(f"Frame skip set to {FRAME_SKIP}")
        elif key == ord('-'):  # Decrease frame skip
            FRAME_SKIP = max(FRAME_SKIP - 5, 1)
            print(f"Frame skip set to {FRAME_SKIP}")
        elif key == ord('p'):  # Pause/play
            cv2.waitKey(0)  # Wait until any key is pressed
            
        frame_counter += 1

except Exception as e:
    print(f"Error during processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Save attendance to CSV
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time", "Date"])
        for name, (time_str, date_str) in attendance.items():
            writer.writerow([name, time_str, date_str])
    
    print("\n--- Performance Summary ---")
    print(f"Total frames: {frame_counter}")
    print(f"Frames processed: {frames_processed}")
    print(f"Faces detected: {faces_detected}")
    print(f"Average FPS: {frame_counter / max(1, time.time() - start_time):.2f}")
    print(f"Processing FPS: {frames_processed / max(1, time.time() - start_time):.2f}")