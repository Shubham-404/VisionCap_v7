"""
Advanced Classroom Facial Recognition Attendance System
-------------------------------------------------------
A high-performance, multithreaded system for tracking student attendance
in a classroom of up to 180 students.

Features:
- Optimized face detection and recognition using parallel processing
- Frame skipping and resizing for smooth performance
- CSV attendance output with timestamps
- Real-time visual feedback
- Progress tracking and performance metrics

Usage:
    python classroom_attendance.py --video path/to/video.mp4 --faces path/to/known_faces/ 
"""

import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
import time
import argparse
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class FaceEncodingManager:
    """Manages the loading and encoding of known face images"""
    
    def __init__(self, faces_dir):
        self.faces_dir = Path(faces_dir)
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_faces(self, use_cache=True):
        """Load known faces from directory and encode them"""
        cache_file = self.faces_dir / "encodings_cache.npy"
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_file):
            try:
                cache_data = np.load(cache_file, allow_pickle=True)
                self.known_face_encodings = cache_data[0]
                self.known_face_names = cache_data[1].tolist()
                logger.info(f"Loaded {len(self.known_face_names)} face encodings from cache")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # If cache failed or not requested, load from images
        logger.info("Encoding faces from images (this may take a while)...")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        face_images = []
        
        # Look for images in the directory and its subdirectories
        for ext in image_extensions:
            face_images.extend(list(self.faces_dir.glob(f"**/*{ext}")))
        
        if not face_images:
            logger.error(f"No images found in {self.faces_dir}")
            return False
        
        # Process images with multithreading for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(self._process_face_image, face_images))
        
        # Filter out failures and add to our lists
        for name, encoding in results:
            if encoding is not None:
                self.known_face_names.append(name)
                self.known_face_encodings.append(encoding)
        
        logger.info(f"Encoded {len(self.known_face_encodings)} faces out of {len(face_images)} images")
        
        # Save to cache for future runs
        if len(self.known_face_encodings) > 0:
            np.save(cache_file, [
                np.array(self.known_face_encodings), 
                np.array(self.known_face_names)
            ])
            logger.info(f"Saved encodings to cache for faster loading next time")
        
        return len(self.known_face_encodings) > 0
    
    def _process_face_image(self, image_path):
        """Process a single face image and return the name and encoding"""
        try:
            # Get student name from file or directory name
            if image_path.parent.name != self.faces_dir.name:
                # If image is in a student-named subdirectory
                name = image_path.parent.name
            else:
                # Otherwise use the filename without extension
                name = image_path.stem
            
            # Load and encode image
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No face found in {image_path}")
                return name, None
            
            # Use the first face found in the image
            encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
            return name, encoding
            
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")
            return None, None


class FrameProcessor:
    """Processes video frames to detect and recognize faces"""
    
    def __init__(self, encoding_manager, frame_scale=0.25, batch_size=32):
        self.encoding_manager = encoding_manager
        self.frame_scale = frame_scale
        self.batch_size = batch_size
        self.attendance = {}  # Track attendance: {name: {'time': time, 'frame': frame_number}}
        
    def process_frame(self, frame, frame_number):
        """Process a single video frame to detect and recognize faces"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if not face_locations:
            return []
            
        # Get encodings of found faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Check each face against known faces
        recognized_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back to original size
            scaled_top = int(top / self.frame_scale)
            scaled_right = int(right / self.frame_scale)
            scaled_bottom = int(bottom / self.frame_scale)
            scaled_left = int(left / self.frame_scale)
            
            # Look for matches
            matches = face_recognition.compare_faces(
                self.encoding_manager.known_face_encodings, 
                face_encoding,
                tolerance=0.6  # Adjust tolerance as needed
            )
            
            name = "Unknown"
            
            # If we found matches
            if True in matches:
                # Use the match with the smallest distance
                face_distances = face_recognition.face_distance(
                    self.encoding_manager.known_face_encodings, 
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.encoding_manager.known_face_names[best_match_index]
                    
                    # Mark attendance if not already recorded
                    if name not in self.attendance:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        self.attendance[name] = {
                            'time': current_time,
                            'frame': frame_number
                        }
                        logger.info(f"Marked attendance for {name} at frame {frame_number}")
            
            # Add to list of recognized faces
            recognized_faces.append((scaled_left, scaled_top, scaled_right, scaled_bottom, name))
        
        return recognized_faces


class AttendanceSystem:
    """Main attendance system that processes video and outputs results"""
    
    def __init__(self, video_path, faces_dir, output_dir="./output", frame_skip=2, frame_scale=0.25):
        self.video_path = video_path
        self.faces_dir = faces_dir
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.frame_scale = frame_scale
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize managers
        self.encoding_manager = FaceEncodingManager(faces_dir)
        self.frame_processor = None
        
        # Threading variables
        self.processing_queue = queue.Queue(maxsize=100)
        self.results_queue = queue.Queue()
        self.processing_active = False
        self.display_active = False
        
        # Performance metrics
        self.start_time = None
        self.frames_processed = 0
        self.frames_displayed = 0
    
    def generate_attendance_report(self):
        """Generate and save the attendance report as CSV"""
        if not self.frame_processor or not self.frame_processor.attendance:
            logger.warning("No attendance data to report")
            return False
        
        # Create a dataframe
        attendance_data = []
        for name, data in self.frame_processor.attendance.items():
            attendance_data.append({
                'Name': name,
                'Time Detected': data['time'],
                'Frame Number': data['frame']
            })
        
        df = pd.DataFrame(attendance_data)
        df = df.sort_values('Name')
        
        # Save to CSV with timestamp
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"attendance_{date_str}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Attendance report saved to {csv_path}")
        
        # Also save summary stats
        with open(self.output_dir / f"summary_{date_str}.txt", 'w') as f:
            f.write(f"Attendance System Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Students detected: {len(self.frame_processor.attendance)}\n")
            f.write(f"Frames processed: {self.frames_processed}\n")
            f.write(f"Processing time: {time.time() - self.start_time:.2f} seconds\n")
            if self.frames_processed > 0:
                f.write(f"Average FPS: {self.frames_processed / (time.time() - self.start_time):.2f}\n")
        
        return True
    
    def processing_worker(self):
        """Thread worker for processing frames"""
        logger.info("Frame processing worker started")
        
        while self.processing_active:
            try:
                # Get a frame to process
                frame_data = self.processing_queue.get(timeout=1)
                if frame_data is None:
                    break
                
                frame_number, frame = frame_data
                
                # Process the frame
                faces = self.frame_processor.process_frame(frame, frame_number)
                
                # Put results in the queue
                self.results_queue.put((frame_number, frame, faces))
                
                # Update metrics
                self.frames_processed += 1
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
        
        logger.info("Frame processing worker stopped")
    
    def display_worker(self, window_name="Classroom Attendance"):
        """Thread worker for displaying results"""
        logger.info("Display worker started")
        
        # Initialize for progress bar
        total_frames = 0
        if os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        last_display_time = time.time()
        display_interval = 1.0 / 30.0  # Target 30 FPS display
        
        while self.display_active:
            try:
                # Get processed results
                frame_number, frame, faces = self.results_queue.get(timeout=0.1)
                
                # Control display rate for smooth playback
                now = time.time()
                if now - last_display_time < display_interval:
                    time.sleep(display_interval - (now - last_display_time))
                
                # Draw faces on frame
                for left, top, right, bottom, name in faces:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw a label with the name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                
                # Add progress and metrics
                elapsed = time.time() - self.start_time
                fps = self.frames_processed / max(1, elapsed)
                
                # Progress bar
                if total_frames > 0:
                    progress = frame_number / total_frames
                    bar_width = 400
                    bar_height = 20
                    filled_width = int(bar_width * progress)
                    
                    # Draw progress bar
                    cv2.rectangle(frame, (10, 20), (10 + bar_width, 20 + bar_height), (200, 200, 200), cv2.FILLED)
                    cv2.rectangle(frame, (10, 20), (10 + filled_width, 20 + bar_height), (0, 255, 0), cv2.FILLED)
                    
                    # Text on progress bar
                    cv2.putText(frame, f"{progress*100:.1f}%", 
                                (10 + bar_width//2 - 25, 20 + bar_height - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add FPS and frame counter
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Frame: {frame_number}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Students: {len(self.frame_processor.attendance)}/180", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow(window_name, frame)
                
                # Update metrics
                self.frames_displayed += 1
                last_display_time = time.time()
                
                # Check for keypresses
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    self.processing_active = False
                    break
                
                # Mark task as done
                self.results_queue.task_done()
                
            except queue.Empty:
                # If queue is empty, just wait a bit
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in display worker: {e}")
        
        logger.info("Display worker stopped")
        cv2.destroyAllWindows()
    
    def run(self):
        """Run the attendance system"""
        # Load known faces
        if not self.encoding_manager.load_known_faces():
            logger.error("Failed to load known faces. Exiting.")
            return False
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.encoding_manager, frame_scale=self.frame_scale)
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} at {fps:.2f} FPS, {total_frames} frames")
        
        # Create a window
        window_name = "Classroom Attendance"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width//2, height//2)  # Resize for convenience
        
        # Start processing and display threads
        self.processing_active = True
        self.display_active = True
        
        processing_thread = threading.Thread(target=self.processing_worker)
        display_thread = threading.Thread(target=self.display_worker, args=(window_name,))
        
        processing_thread.start()
        display_thread.start()
        
        # Start processing frames
        self.start_time = time.time()
        frame_number = 0
        
        try:
            while cap.isOpened() and self.processing_active:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if frame_number % self.frame_skip == 0:
                    # Wait if queue is full
                    while self.processing_queue.full() and self.processing_active:
                        time.sleep(0.01)
                    
                    # Add to processing queue
                    if self.processing_active:
                        self.processing_queue.put((frame_number, frame))
                
                frame_number += 1
                
                # Check if ESC was pressed in display thread
                if not self.display_active:
                    break
            
            # Signal threads to finish and wait
            logger.info("Video processing complete, waiting for threads to finish...")
            self.processing_active = False
            self.processing_queue.put(None)  # Sentinel value
            
            processing_thread.join()
            self.display_active = False
            display_thread.join()
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            self.processing_active = False
            self.display_active = False
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
            # Generate report
            self.generate_attendance_report()
            
            # Show final statistics
            elapsed = time.time() - self.start_time
            logger.info(f"Finished in {elapsed:.2f} seconds")
            logger.info(f"Processed {self.frames_processed} frames ({self.frames_processed/elapsed:.2f} FPS)")
            logger.info(f"Recognized {len(self.frame_processor.attendance)} students")
            
            return True


def main():
    parser = argparse.ArgumentParser(description="Classroom Attendance System")
    parser.add_argument("--video", required=True, help="Path to classroom video")
    parser.add_argument("--faces", required=True, help="Path to known faces directory")
    parser.add_argument("--output", default="./output", help="Output directory for attendance report")
    parser.add_argument("--skip", type=int, default=2, help="Process every nth frame")
    parser.add_argument("--scale", type=float, default=0.25, help="Scale factor for frame processing")
    
    args = parser.parse_args()
    
    # Run the system
    system = AttendanceSystem(
        video_path=args.video,
        faces_dir=args.faces,
        output_dir=args.output,
        frame_skip=args.skip,
        frame_scale=args.scale
    )
    
    system.run()


if __name__ == "__main__":
    main()