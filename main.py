"""
Real-time Crowd Density Detection System
Author: AI Assistant
Description: Detects crowd density using a pre-trained PyTorch model and webcam feed

Requirements:
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0 
numpy>=1.21.0
Pillow>=8.3.0
ultralytics>=8.0.0
"""

import cv2
import torch
import numpy as np
import time
import sys
import os
from torchvision import transforms
from PIL import Image
import threading

class CrowdDensityDetector:
    def __init__(self, model_path='best.pt'):
        """
        Initialize the crowd density detector
        
        Args:
            model_path (str): Path to the pre-trained PyTorch model
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_beep_time = 0
        self.beep_cooldown = 3.0  # Cooldown period in seconds
        self.current_status = "Unknown"
        self.previous_status = "Unknown"
        self.is_ultralytics = False  # Flag to track model type
        
        # Density thresholds based on person count
        self.low_threshold = 1      # 1 person = low density
        self.medium_threshold = 3   # 2-3 people = medium density  
        self.high_threshold = 4     # 4+ people = high density
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Common input size for many models
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Initialize the model
        self.load_model()
        
    def load_model(self):
        """
        Load the pre-trained PyTorch model from the .pt file
        Handles Ultralytics models and PyTorch 2.6 security requirements
        """
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found!")
            
            # First, try to detect if this is an Ultralytics model
            try:
                # Try loading with Ultralytics library first
                from ultralytics import YOLO
                print("Detected Ultralytics model format, loading with YOLO...")
                self.model = YOLO(self.model_path)
                self.is_ultralytics = True
                print(f"Ultralytics model loaded successfully")
                return
                
            except ImportError:
                print("Ultralytics not installed, trying PyTorch loading...")
                self.is_ultralytics = False
            except Exception as e:
                print(f"Ultralytics loading failed: {e}")
                print("Trying PyTorch loading...")
                self.is_ultralytics = False
            
            # If Ultralytics fails, try PyTorch loading methods
            try:
                # Method 1: Try with weights_only=False (for trusted models)
                print("Attempting to load with PyTorch (weights_only=False)...")
                self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.is_ultralytics = False
                
            except Exception as e1:
                try:
                    # Method 2: Try with safe globals for Ultralytics
                    print("Attempting to load with safe globals...")
                    from ultralytics.nn.tasks import DetectionModel
                    
                    # Add safe globals for Ultralytics
                    torch.serialization.add_safe_globals([DetectionModel])
                    
                    self.model = torch.load(self.model_path, map_location=self.device, weights_only=True)
                    self.is_ultralytics = False
                    
                except Exception as e2:
                    try:
                        # Method 3: Try loading as state dict
                        print("Attempting to load as state dict...")
                        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                        
                        if isinstance(checkpoint, dict) and 'model' in checkpoint:
                            self.model = checkpoint['model']
                        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            # You'll need to define your model architecture here
                            # self.model = YourModelClass()
                            # self.model.load_state_dict(checkpoint['state_dict'])
                            raise Exception("State dict loading requires model architecture definition")
                        else:
                            self.model = checkpoint
                        
                        self.is_ultralytics = False
                        
                    except Exception as e3:
                        print(f"All loading methods failed:")
                        print(f"  Method 1 (weights_only=False): {e1}")
                        print(f"  Method 2 (safe_globals): {e2}")
                        print(f"  Method 3 (state_dict): {e3}")
                        raise Exception("Could not load model with any method")
            
            # Set model to evaluation mode (only for PyTorch models)
            if not self.is_ultralytics:
                self.model.eval()
                self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. If this is an Ultralytics/YOLO model, install: pip install ultralytics")
            print("2. For trusted models, the script will use weights_only=False")
            print("3. Ensure 'best.pt' exists and is a valid model file")
            sys.exit(1)
    
    def classify_density(self, person_count):
        """
        Classify density level based on person count
        
        Args:
            person_count (int): Number of people detected
            
        Returns:
            str: Density classification
        """
        if person_count == 0:
            return "no people"
        elif person_count <= self.low_threshold:
            return "low density"
        elif person_count < self.high_threshold:
            return "medium density"
        else:
            return "high density"

    # FIX 1: The misplaced code for preprocessing is now correctly placed in this method.
    def preprocess_frame(self, frame):
        """
        Preprocess the webcam frame for model inference
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model inference
        """
        try:
            # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transformations
            tensor = self.transform(pil_image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            # Move to device
            tensor = tensor.to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    # FIX 2: Renamed this method from `preprocess_frame` to `predict_density` to match its purpose and calls.
    # FIX 3: Changed the parameter name from `frame_tensor` to `processed_input` to resolve the "not defined" error.
    # FIX 4: Refactored to consistently return three values: (status, confidence, person_count).
    def predict_density(self, processed_input):
        """
        Perform inference on the preprocessed frame or raw frame.
        Handles both Ultralytics and standard PyTorch models.
        
        Args:
            processed_input: Preprocessed tensor or original frame for Ultralytics
            
        Returns:
            tuple: (status, confidence, person_count)
        """
        try:
            if self.is_ultralytics:
                # For Ultralytics models, input is the original frame
                results = self.model(processed_input, verbose=False)
                
                person_count = 0
                confidence = 0.0
                
                if len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        person_count = len(result.boxes)
                        if person_count > 0:
                            # Use the maximum confidence among detected objects
                            confidence = float(result.boxes.conf.max())
                
                status = self.classify_density(person_count)
                return status, confidence, person_count
                    
            else:
                # Standard PyTorch model inference (input is a tensor)
                with torch.no_grad():
                    outputs = self.model(processed_input)
                
                # --- This section is highly dependent on your model's output ---
                # We assume the model outputs a single value (regression) for the crowd count.
                # You may need to adjust this logic for your specific model.
                
                # Squeeze to get a single value if the tensor has extra dimensions
                raw_prediction = outputs.squeeze().item()
                
                person_count = max(0, int(round(raw_prediction))) # Ensure count is a non-negative integer
                confidence = raw_prediction # For regression, we can display the raw score
                
                status = self.classify_density(person_count)
                return status, confidence, person_count
                        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", 0.0, 0
    
    def play_beep_sound(self):
        """
        Play an audible beep sound in a separate thread
        This uses the system bell - works on most systems
        """
        def beep():
            try:
                # Method 1: System bell (most compatible)
                print('\a', flush=True)  # ASCII bell character
                    
            except Exception as e:
                print(f"Could not play beep sound: {e}")
        
        # Run beep in separate thread to avoid blocking
        beep_thread = threading.Thread(target=beep)
        beep_thread.daemon = True
        beep_thread.start()
    
    def should_beep(self, current_status):
        """
        Determine if a beep should be played based on current status and cooldown
        Now only beeps for high density (4+ people)
        
        Args:
            current_status (str): Current density status
            
        Returns:
            bool: True if beep should be played
        """
        current_time = time.time()
        
        # Beep conditions:
        # 1. Current status is "high density" (4+ people)
        # 2. Either status changed to high, or cooldown period has passed
        if current_status == "high density":
            if (self.previous_status != "high density" or 
                current_time - self.last_beep_time >= self.beep_cooldown):
                self.last_beep_time = current_time
                return True
        
        return False
    
    def draw_overlay(self, frame, status, confidence=None, person_count=None):
        """
        Draw status overlay on the frame with person count
        
        Args:
            frame: OpenCV frame
            status (str): Density status
            confidence: Model confidence (optional)
            person_count: Number of people detected (optional)
            
        Returns:
            frame: Frame with overlay
        """
        # Define colors for each density level
        if status == "high density":
            color = (0, 0, 255)  # Red in BGR
            status_text = "HIGH DENSITY"
        elif status == "medium density":
            color = (0, 165, 255)  # Orange in BGR
            status_text = "MEDIUM DENSITY"
        elif status == "low density":
            color = (0, 255, 0)  # Green in BGR
            status_text = "LOW DENSITY"
        elif status == "no people":
            color = (128, 128, 128)  # Gray in BGR
            status_text = "NO PEOPLE"
        else:
            color = (0, 255, 255)  # Yellow in BGR
            status_text = "UNKNOWN"
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), 
                     (20 + text_width, 20 + text_height + baseline), 
                     (0, 0, 0), -1)
        
        # Draw status text
        cv2.putText(frame, status_text, (15, 15 + text_height), 
                   font, font_scale, color, thickness)
        
        # Draw person count if available
        if person_count is not None:
            count_text = f"People: {person_count}"
            cv2.putText(frame, count_text, (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw confidence if available
        if confidence is not None:
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, conf_text, (15, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw thresholds info
        thresholds_text = f"Thresholds: 1=Low, 2-3=Med, 4+=High"
        cv2.putText(frame, thresholds_text, (15, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw device info
        device_text = f"Device: {self.device}"
        cv2.putText(frame, device_text, (15, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        Main execution loop for real-time crowd density detection
        """
        print("Starting crowd density detection...")
        print("Press 'q' to quit")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Check if webcam opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                
                # Prepare input for the model
                if self.is_ultralytics:
                    # For Ultralytics, pass the original frame
                    processed_input = frame
                else:
                    # For standard PyTorch models, use preprocessed tensor
                    processed_input = self.preprocess_frame(frame)
                
                if processed_input is not None:
                    # Get prediction
                    status, confidence, person_count = self.predict_density(processed_input)
                    
                    # Update status tracking
                    self.previous_status = self.current_status
                    self.current_status = status
                    
                    # Check if beep should be played (only for high density = 4+ people)
                    if self.should_beep(status):
                        self.play_beep_sound()
                        print(f"⚠️  HIGH DENSITY ALERT! {person_count} people detected (Confidence: {confidence:.3f})")
                    
                    # Draw overlay on frame
                    frame = self.draw_overlay(frame, status, confidence, person_count)
                
                # Display the frame
                cv2.imshow('Crowd Density Detection', frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup completed")

    def process_image(self, image_path):
        """
        Process a single image for crowd density detection
        
        Args:
            image_path (str): Path to the image file
        """
        try:
            print(f"Processing image: {image_path}")
            
            # Load image
            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' not found!")
                return
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not load image '{image_path}'")
                return
            
            # Prepare input for the model
            if self.is_ultralytics:
                processed_input = frame
            else:
                processed_input = self.preprocess_frame(frame)
            
            if processed_input is not None:
                # Get prediction
                status, confidence, person_count = self.predict_density(processed_input)
                
                # Draw overlay on frame
                result_frame = self.draw_overlay(frame.copy(), status, confidence, person_count)
                
                # Display results
                print(f"Result: {status.upper()} - {person_count} people detected (Confidence: {confidence:.3f})")
                
                # Show image
                cv2.imshow('Crowd Density Detection - Image', result_frame)
                print("Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Save result image
                output_path = f"result_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, result_frame)
                print(f"Result saved as: {output_path}")
                
            else:
                print("Error processing image")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def process_video(self, video_path, save_output=True):
        """
        Process a video file for crowd density detection
        
        Args:
            video_path (str): Path to the video file
            save_output (bool): Whether to save the output video
        """
        try:
            print(f"Processing video: {video_path}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Error: Video file '{video_path}' not found!")
                return
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video file '{video_path}'")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Setup video writer if saving output
            output_writer = None
            if save_output:
                output_path = f"result_{os.path.basename(video_path)}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"Output will be saved as: {output_path}")
            
            frame_count = 0
            high_density_frames = 0
            medium_density_frames = 0
            low_density_frames = 0
            no_people_frames = 0
            
            print("Processing video frames... Press 'q' to stop early")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Prepare input for the model
                if self.is_ultralytics:
                    processed_input = frame
                else:
                    processed_input = self.preprocess_frame(frame)
                
                if processed_input is not None:
                    # Get prediction
                    status, confidence, person_count = self.predict_density(processed_input)
                    
                    # Count frames by density level
                    if status == "high density":
                        high_density_frames += 1
                    elif status == "medium density":
                        medium_density_frames += 1
                    elif status == "low density":
                        low_density_frames += 1
                    elif status == "no people":
                        no_people_frames += 1
                    
                    # Update status tracking for audio alerts
                    self.previous_status = self.current_status
                    self.current_status = status
                    
                    # Play beep for high density only (4+ people)
                    if self.should_beep(status):
                        self.play_beep_sound()
                        print(f"Frame {frame_count}: HIGH DENSITY ALERT! {person_count} people detected (Confidence: {confidence:.3f})")
                    
                    # Draw overlay
                    result_frame = self.draw_overlay(frame.copy(), status, confidence, person_count)
                    
                    # Add frame counter
                    cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                               (15, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow('Crowd Density Detection - Video', result_frame)
                    
                    # Save frame if output writer is available
                    if output_writer:
                        output_writer.write(result_frame)
                    
                    # Progress indicator
                    if frame_count % (max(1, total_frames // 20)) == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Current: {status} ({person_count} people)")
                
                # Check for early exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
            
            # Cleanup
            cap.release()
            if output_writer:
                output_writer.release()
            cv2.destroyAllWindows()
            
            # Print summary
            if frame_count > 0:
                print(f"\nVideo Processing Complete!")
                print(f"Total frames processed: {frame_count}")
                print(f"📊 DENSITY BREAKDOWN:")
                print(f"   🔴 High Density (4+ people):   {high_density_frames} frames ({(high_density_frames/frame_count)*100:.1f}%)")
                print(f"   🟠 Medium Density (2-3 people): {medium_density_frames} frames ({(medium_density_frames/frame_count)*100:.1f}%)")
                print(f"   🟢 Low Density (1 person):      {low_density_frames} frames ({(low_density_frames/frame_count)*100:.1f}%)")
                print(f"   ⚪ No People (0 people):        {no_people_frames} frames ({(no_people_frames/frame_count)*100:.1f}%)")
            
            if save_output and output_writer:
                print(f"Output video saved successfully")
                
        except Exception as e:
            print(f"Error processing video: {e}")

def main():
    """
    Main function to run the crowd density detector
    Supports webcam, image, and video processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Crowd Density Detection')
    parser.add_argument('--mode', choices=['webcam', 'image', 'video'], default='webcam',
                       help='Processing mode: webcam (default), image, or video')
    parser.add_argument('--input', type=str, help='Input file path for image or video mode')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to model file')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save output video (video mode only)')
    
    args = parser.parse_args()
    
    detector = None # Initialize detector to None
    try:
        # If no command line arguments were provided, show help and default to webcam
        if len(sys.argv) == 1:
            print("\n" + "="*60)
            print("CROWD DENSITY DETECTION SYSTEM")
            print("="*60)
            print("\nUsage Examples:")
            print("  Webcam (default):  python main.py")
            print("  Image processing:  python main.py --mode image --input crowd_image.jpg")
            print("  Video processing:  python main.py --mode video --input crowd_video.mp4")
            print("\nStarting webcam mode in 3 seconds...")
            print("Press Ctrl+C to cancel")
            time.sleep(3)
            # Initialize detector for default run
            detector = CrowdDensityDetector(model_path=args.model)
            detector.run()
            return # Exit after default run

        # Initialize detector for command-line specified run
        detector = CrowdDensityDetector(model_path=args.model)
        
        if args.mode == 'webcam':
            print("Starting webcam mode...")
            detector.run()
            
        elif args.mode == 'image':
            if not args.input:
                print("Error: --input required for image mode")
                print("Usage: python main.py --mode image --input path/to/image.jpg")
                return
            detector.process_image(args.input)
            
        elif args.mode == 'video':
            if not args.input:
                print("Error: --input required for video mode")
                print("Usage: python main.py --mode video --input path/to/video.mp4")
                return
            detector.process_video(args.input, save_output=not args.no_save)
        
    except KeyboardInterrupt:
            print("\nCancelled by user")
            return
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()