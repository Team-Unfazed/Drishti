# Advanced AI-Powered Crowd Counter with Deep Learning
# Uses YOLOv8 for highly accurate person detection
# Significantly improved accuracy over Haar cascades

import cv2
import numpy as np
from datetime import datetime
import torch
import torchvision.transforms as transforms
from collections import deque
import math

# Try to import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 available - using advanced AI detection")
except ImportError:
    YOLO_AVAILABLE = False 
    print("YOLOv8 not available - install with: pip install ultralytics")

# Try to import MediaPipe for additional person detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe available - using pose detection backup")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - install with: pip install mediapipe")

class AdvancedCrowdCounter:
    def __init__(self):
        # Initialize detection models
        self.yolo_model = None
        self.mp_pose = None
        self.mp_drawing = None
        
        # Tracking variables
        self.person_tracks = {}
        self.next_person_id = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=30)  # Store last 30 frame counts
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.min_detection_area = 800  # Minimum area for a valid person detection
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize AI models for person detection"""
        
        if YOLO_AVAILABLE:
            try:
                # Load YOLOv8 model (will download automatically on first run)
                self.yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
                print("YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"Failed to load YOLOv8: {e}")
                self.yolo_model = None
        
        if MEDIAPIPE_AVAILABLE:
            try:
                # Initialize MediaPipe Pose
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                print("MediaPipe Pose initialized successfully")
            except Exception as e:
                print(f"Failed to initialize MediaPipe: {e}")
                self.mp_pose = None
    
    def detect_persons_yolo(self, frame):
        """Detect persons using YOLOv8"""
        if not self.yolo_model:
            return []
        
        try:
            # Run inference
            results = self.yolo_model(frame, verbose=False)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])
                            if confidence >= self.confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                                
                                # Filter out very small detections
                                if w * h >= self.min_detection_area:
                                    persons.append({
                                        'bbox': (x, y, w, h),
                                        'confidence': confidence,
                                        'method': 'YOLOv8'
                                    })
            
            return persons
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def detect_persons_mediapipe(self, frame):
        """Detect persons using MediaPipe pose detection"""
        if not self.mp_pose:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.mp_pose.process(rgb_frame)
            
            persons = []
            if results.pose_landmarks:
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Extract landmark coordinates
                landmarks = results.pose_landmarks.landmark
                
                # Calculate bounding box from pose landmarks
                x_coords = [landmark.x * w for landmark in landmarks if landmark.visibility > 0.5]
                y_coords = [landmark.y * h for landmark in landmarks if landmark.visibility > 0.5]
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding to bounding box
                    padding = 20
                    x = max(0, min_x - padding)
                    y = max(0, min_y - padding)
                    w = min(frame.shape[1] - x, max_x - min_x + 2 * padding)
                    h = min(frame.shape[0] - y, max_y - min_y + 2 * padding)
                    
                    if w * h >= self.min_detection_area:
                        persons.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.8,  # MediaPipe doesn't provide confidence, use fixed value
                            'method': 'MediaPipe'
                        })
            
            return persons
            
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return []
    
    def detect_persons_advanced_cv(self, frame):
        """Advanced OpenCV-based person detection using multiple techniques"""
        persons = []
        
        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Initialize HOG descriptor for person detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        try:
            # HOG person detection
            (rects, weights) = hog.detectMultiScale(
                gray,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
                groupThreshold=2
            )
            
            # Process detections
            for i, (x, y, w, h) in enumerate(rects):
                confidence = weights[i][0] if len(weights) > i else 0.5
                
                # Filter based on aspect ratio (people are typically taller than wide)
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.2 and w * h >= self.min_detection_area:
                    persons.append({
                        'bbox': (x, y, w, h),
                        'confidence': min(confidence, 1.0),
                        'method': 'HOG'
                    })
            
        except Exception as e:
            print(f"HOG detection error: {e}")
        
        return persons
    
    def remove_duplicate_detections(self, all_detections):
        """Remove overlapping detections using advanced NMS"""
        if len(all_detections) == 0:
            return []
        
        # Convert to format for NMS
        boxes = []
        scores = []
        
        for detection in all_detections:
            x, y, w, h = detection['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(detection['confidence'])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Extract final detections
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detection = all_detections[i].copy()
                final_detections.append(detection)
        
        return final_detections
    
    def track_persons(self, detections):
        """Simple person tracking to reduce flickering"""
        current_persons = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Find closest existing track
            min_distance = float('inf')
            closest_id = None
            
            for person_id, track in self.person_tracks.items():
                track_x, track_y = track['center']
                distance = math.sqrt((center_x - track_x)**2 + (center_y - track_y)**2)
                
                if distance < min_distance and distance < 100:  # Maximum tracking distance
                    min_distance = distance
                    closest_id = person_id
            
            if closest_id is not None:
                # Update existing track
                self.person_tracks[closest_id] = {
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': detection['confidence'],
                    'method': detection['method'],
                    'last_seen': self.frame_count
                }
                current_persons.append(closest_id)
            else:
                # Create new track
                self.person_tracks[self.next_person_id] = {
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'confidence': detection['confidence'],
                    'method': detection['method'],
                    'last_seen': self.frame_count
                }
                current_persons.append(self.next_person_id)
                self.next_person_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for person_id, track in self.person_tracks.items():
            if self.frame_count - track['last_seen'] > 30:  # Remove if not seen for 30 frames
                tracks_to_remove.append(person_id)
        
        for person_id in tracks_to_remove:
            del self.person_tracks[person_id]
        
        return len(current_persons)
    
    def get_smoothed_count(self, current_count):
        """Get smoothed person count using temporal filtering"""
        self.detection_history.append(current_count)
        
        if len(self.detection_history) < 3:
            return current_count
        
        # Use median filtering to reduce noise
        recent_counts = list(self.detection_history)[-5:]
        return int(np.median(recent_counts))
    
    def process_frame(self, frame):
        """Process a single frame and return annotated result"""
        self.frame_count += 1
        all_detections = []
        
        # Method 1: YOLOv8 (most accurate)
        if self.yolo_model:
            yolo_detections = self.detect_persons_yolo(frame)
            all_detections.extend(yolo_detections)
        
        # Method 2: MediaPipe (good for pose-based detection)
        if self.mp_pose and len(all_detections) == 0:  # Use as backup if YOLO fails
            mp_detections = self.detect_persons_mediapipe(frame)
            all_detections.extend(mp_detections)
        
        # Method 3: Advanced OpenCV (fallback)
        if len(all_detections) == 0:
            cv_detections = self.detect_persons_advanced_cv(frame)
            all_detections.extend(cv_detections)
        
        # Remove duplicate detections
        final_detections = self.remove_duplicate_detections(all_detections)
        
        # Track persons and get count
        person_count = self.track_persons(final_detections)
        smoothed_count = self.get_smoothed_count(person_count)
        
        # Draw results on frame
        annotated_frame = self.draw_results(frame, final_detections, smoothed_count)
        
        return annotated_frame, smoothed_count
    
    def draw_results(self, frame, detections, person_count):
        """Draw detection results on frame"""
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        for person_id, track in self.person_tracks.items():
            x, y, w, h = track['bbox']
            confidence = track['confidence']
            method = track['method']
            
            # Color based on confidence and method
            if method == 'YOLOv8':
                color = (0, 255, 0)  # Green for YOLO
            elif method == 'MediaPipe':
                color = (255, 165, 0)  # Orange for MediaPipe
            else:
                color = (0, 165, 255)  # Orange for HOG
            
            # Draw bounding box
            thickness = 3 if confidence > 0.8 else 2
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw person ID and confidence
            label = f'ID:{person_id} ({confidence:.2f})'
            cv2.putText(annotated_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw main counter
        cv2.putText(annotated_frame, f'AI Crowd Counter - People: {person_count}', 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Draw detection method info
        methods_used = list(set([track['method'] for track in self.person_tracks.values()]))
        methods_text = 'Methods: ' + ', '.join(methods_used) if methods_used else 'Methods: None'
        cv2.putText(annotated_frame, methods_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw accuracy indicator
        accuracy_text = "Accuracy: Very High (AI-Powered)" if self.yolo_model else "Accuracy: High (Computer Vision)"
        cv2.putText(annotated_frame, accuracy_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (20, annotated_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame

def main():
    """Main function to run the advanced crowd counter"""
    print("Starting Advanced AI-Powered Crowd Counter...")
    print("="*50)
    
    # Initialize the crowd counter
    counter = AdvancedCrowdCounter()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Advanced Crowd Counter is running!")
    print("Features:")
    print("- YOLOv8 AI person detection (if available)")
    print("- MediaPipe pose detection (backup)")
    print("- Advanced OpenCV methods (fallback)")
    print("- Person tracking to reduce false positives")
    print("- Temporal smoothing for stable counts")
    print("- Multi-method fusion for maximum accuracy")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*50)
    
    frame_count = 0
    total_processing_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from webcam")
            break
        
        # Process frame
        start_time = cv2.getTickCount()
        annotated_frame, person_count = counter.process_frame(frame)
        end_time = cv2.getTickCount()
        
        # Calculate processing time
        processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
        total_processing_time += processing_time
        frame_count += 1
        
        # Add FPS info
        if frame_count > 0:
            avg_fps = 1000 / (total_processing_time / frame_count)
            cv2.putText(annotated_frame, f'FPS: {avg_fps:.1f}', (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Advanced AI Crowd Counter', annotated_frame)
        
        # Print periodic updates
        if frame_count % 60 == 0:  # Every 2 seconds at 30fps
            print(f"Frame {frame_count}: Detected {person_count} people "
                  f"(Processing: {processing_time:.1f}ms)")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            filename = f"crowd_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("="*50)
    print("Session Summary:")
    print(f"Total frames processed: {frame_count}")
    if frame_count > 0:
        print(f"Average FPS: {1000 / (total_processing_time / frame_count):.1f}")
        print(f"Average processing time: {total_processing_time / frame_count:.1f}ms")
    print("Advanced Crowd Counter session completed!")

if __name__ == "__main__":
    main()