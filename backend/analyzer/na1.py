# analyzer/na.py

import cv2
import face_recognition
import numpy as np
import os
import argparse
import csv
import datetime
import threading
from collections import defaultdict
from queue import Queue

# Constants
FRAME_SKIP = 3
DOWNSCALE = 0.25
ATTENDANCE_LOG = 'output/attendance.csv'

# Utility: Load and encode known faces
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []

    print("[INFO] Loading known faces from:", known_faces_dir)
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f"[ENCODED] {name}")
            else:
                print(f"[WARNING] No face found in {filename}")
    return known_encodings, known_names

# Utility: Save attendance log to CSV
def save_attendance_csv(attendance_dict):
    os.makedirs(os.path.dirname(ATTENDANCE_LOG), exist_ok=True)
    with open(ATTENDANCE_LOG, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'First Seen', 'Last Seen', 'Frames Seen'])
        for name, data in attendance_dict.items():
            writer.writerow([name, data['first_seen'], data['last_seen'], data['count']])
    print(f"[SAVED] Attendance written to {ATTENDANCE_LOG}")

# Worker thread for face recognition
def recognition_worker(frame_queue, result_queue, known_encodings, known_names):
    attendance = defaultdict(lambda: {'first_seen': None, 'last_seen': None, 'count': 0})

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_idx, small_frame, full_frame = item

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                match_idx = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
                name = known_names[match_idx]

            top, right, bottom, left = face_location
            top, right, bottom, left = [int(v / DOWNSCALE) for v in (top, right, bottom, left)]

            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            att = attendance[name]
            if att['first_seen'] is None:
                att['first_seen'] = now
            att['last_seen'] = now
            att['count'] += 1

            recognized.append((name, (top, right, bottom, left)))

        result_queue.put((frame_idx, full_frame, recognized))

    result_queue.put("DONE")
    save_attendance_csv(attendance)

# Draw labels on frame
def draw_faces(frame, faces):
    for name, (top, right, bottom, left) in faces:
        color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Main function
def main(video_path, faces_dir):
    print("[INFO] Starting real-time facial attendance system")
    known_encodings, known_names = load_known_faces(faces_dir)

    if not known_encodings:
        print("[ERROR] No known faces found. Exiting.")
        return

    frame_queue = Queue(maxsize=10)
    result_queue = Queue()

    # Start recognition thread
    threading.Thread(target=recognition_worker, args=(frame_queue, result_queue, known_encodings, known_names), daemon=True).start()

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        if frame_idx % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            frame_queue.put((frame_idx, small_frame, frame.copy()))

        while not result_queue.empty():
            result = result_queue.get()
            if result == "DONE":
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Attendance system completed.")
                return
            idx, result_frame, faces = result
            draw_faces(result_frame, faces)
            cv2.imshow("Attendance", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                frame_queue.put(None)
                cap.release()
                return

        frame_idx += 1

    frame_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

# CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Facial Recognition Attendance System")
    parser.add_argument('--video', required=True, help="Path to input video file")
    parser.add_argument('--faces', required=True, help="Path to directory of known faces")
    args = parser.parse_args()

    main(args.video, args.faces)
