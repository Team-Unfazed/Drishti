import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client
import time
from datetime import datetime, timedelta
import json
import threading
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DrishtiCrowdSafetySystem:
    """
    Project Drishti - AI-Powered Crowd Safety System
    Monitors crowd density, detects emergencies, and provides automated response
    """
    
    def __init__(self):
        # YOLO Model Setup
        self.model = YOLO("best.pt")  # Your trained model
        
        # Twilio Configuration
        self.ACCOUNT_SID = "AC8c30bcb8589dd88c706dea01e2d25bc9"
        self.AUTH_TOKEN = "d55d15fd58c739594e425c4552bfcfd7"
        self.TWILIO_PHONE_NUMBER = "+19205251279"
        
        # Emergency Contact Numbers (Security, Medical, Police)
        self.emergency_contacts = {
            'security': "+918146795946",
            'medical': "+918146795946",  # Add different numbers as needed
            'police': "+918146795946",
            'control_room': "+918146795946"
        }
        
        self.client = Client(self.ACCOUNT_SID, self.AUTH_TOKEN)
        
        # Crowd Safety Parameters
        self.crowd_density_threshold = 15  # Max people per detection zone
        self.panic_detection_threshold = 0.8  # Confidence threshold for panic detection
        self.fire_smoke_threshold = 0.7
        self.bottleneck_threshold = 10  # People count that indicates bottleneck
        
        # Alert Management
        self.last_alert_time = {}  # Track last alert time per category
        self.alert_cooldown = 120  # 2 minutes between similar alerts
        
        # Crowd Monitoring Data
        self.crowd_history = deque(maxlen=100)  # Store last 100 frames of crowd data
        self.current_crowd_count = 0
        self.risk_zones = {}
        self.sentiment_score = 0.5  # 0 = panic, 1 = calm
        
        # Emergency Categories
        self.emergency_types = {
            'high_density': 'Overcrowding Alert',
            'panic': 'Crowd Panic Detected',
            'fire': 'Fire/Smoke Emergency',
            'bottleneck': 'Crowd Bottleneck',
            'missing_person': 'Missing Person Alert',
            'medical': 'Medical Emergency'
        }
        
        logger.info("Drishti Crowd Safety System Initialized")
    
    def analyze_crowd_density(self, detections):
        """Analyze crowd density from YOLO detections"""
        person_count = 0
        crowd_positions = []
        
        for detection in detections:
            boxes = detection.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name == 'person' and confidence > 0.7:
                        person_count += 1
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        crowd_positions.append((center_x, center_y))
        
        self.current_crowd_count = person_count
        self.crowd_history.append({
            'timestamp': datetime.now(),
            'count': person_count,
            'positions': crowd_positions
        })
        
        return person_count, crowd_positions
    
    def detect_crowd_bottlenecks(self, positions, frame_shape):
        """Detect crowd bottlenecks and congestion areas"""
        if len(positions) < self.bottleneck_threshold:
            return []
        
        # Divide frame into grid zones
        height, width = frame_shape[:2]
        grid_size = 100  # pixels
        zones = {}
        
        for x, y in positions:
            zone_x = int(x // grid_size)
            zone_y = int(y // grid_size)
            zone_key = f"{zone_x}_{zone_y}"
            
            if zone_key not in zones:
                zones[zone_key] = 0
            zones[zone_key] += 1
        
        # Identify high-density zones
        bottlenecks = []
        for zone, count in zones.items():
            if count >= self.bottleneck_threshold:
                bottlenecks.append({
                    'zone': zone,
                    'count': count,
                    'risk_level': min(count / self.crowd_density_threshold, 1.0)
                })
        
        return bottlenecks
    
    def analyze_crowd_sentiment(self, detections, frame):
        """Analyze crowd sentiment and detect panic signs"""
        # Simplified sentiment analysis based on movement patterns and density
        if len(self.crowd_history) < 5:
            return 0.5
        
        recent_counts = [entry['count'] for entry in list(self.crowd_history)[-5:]]
        
        # Rapid changes in crowd count might indicate panic
        count_variance = np.var(recent_counts)
        density_factor = min(self.current_crowd_count / self.crowd_density_threshold, 1.0)
        
        # Calculate sentiment score (0 = panic, 1 = calm)
        sentiment = max(0.1, 1.0 - (count_variance * 0.1 + density_factor * 0.3))
        self.sentiment_score = sentiment
        
        return sentiment
    
    def generate_emergency_message(self, emergency_type, details):
        """Generate contextual emergency messages using AI-like intelligence"""
        current_time = datetime.now().strftime('%I:%M %p')
        location = "Event Venue"  # This could be made dynamic
        
        messages = {
            'high_density': f"""
            <Response>
                <Say voice='Polly.Aditi'>
                    Emergency Alert! Overcrowding detected at {location} at {current_time}.
                    Current crowd count: {details.get('count', 0)} people.
                    Risk level: {details.get('risk_level', 'High')}.
                </Say>
                <Pause length='1'/>
                <Say voice='Polly.Aditi'>
                    Immediate crowd control required. Deploy additional security personnel.
                    This is Project Drishti automated alert. Respond immediately.
                </Say>
            </Response>
            """,
            
            'panic': f"""
            <Response>
                <Say voice='Polly.Aditi'>
                    Critical Alert! Crowd panic detected at {location} at {current_time}.
                    Crowd sentiment score: {details.get('sentiment', 0):.2f}.
                    Immediate intervention required.
                </Say>
                <Pause length='1'/>
                <Say voice='Polly.Aditi'>
                    Deploy emergency response teams. Activate evacuation protocols if necessary.
                    This is Project Drishti critical alert.
                </Say>
            </Response>
            """,
            
            'bottleneck': f"""
            <Response>
                <Say voice='Polly.Aditi'>
                    Bottleneck Alert! Crowd congestion detected at {location} at {current_time}.
                    Zone: {details.get('zone', 'Unknown')}. 
                    People count in zone: {details.get('count', 0)}.
                </Say>
                <Pause length='1'/>
                <Say voice='Polly.Aditi'>
                    Redirect crowd flow immediately. Open alternate routes.
                    This is Project Drishti traffic management alert.
                </Say>
            </Response>
            """,
            
            'fire': f"""
            <Response>
                <Say voice='Polly.Aditi'>
                    Fire Emergency! Smoke or fire detected at {location} at {current_time}.
                    Confidence level: {details.get('confidence', 0):.2f}.
                    Evacuate area immediately.
                </Say>
                <Pause length='1'/>
                <Say voice='Polly.Aditi'>
                    Fire department and medical teams required urgently.
                    This is Project Drishti fire safety alert.
                </Say>
            </Response>
            """
        }
        
        return messages.get(emergency_type, messages['high_density'])
    
    def send_emergency_alert(self, emergency_type, details, contact_type='security'):
        """Send emergency alert via phone call"""
        current_time = datetime.now()
        alert_key = f"{emergency_type}_{contact_type}"
        
        # Check cooldown period
        if (alert_key in self.last_alert_time and 
            current_time - self.last_alert_time[alert_key] < timedelta(seconds=self.alert_cooldown)):
            logger.info(f"Alert cooldown active for {alert_key}")
            return
        
        recipient = self.emergency_contacts.get(contact_type, self.emergency_contacts['security'])
        message = self.generate_emergency_message(emergency_type, details)
        
        try:
            call = self.client.calls.create(
                twiml=message,
                to=recipient,
                from_=self.TWILIO_PHONE_NUMBER
            )
            
            logger.info(f"Emergency alert sent! Type: {emergency_type}, Contact: {contact_type}")
            logger.info(f"Call SID: {call.sid}")
            
            self.last_alert_time[alert_key] = current_time
            
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")
    
    def process_frame(self, frame):
        """Process single frame for crowd analysis"""
        # Run YOLO detection
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=0.6,
            show=False,
            verbose=False
        )
        
        # Analyze crowd
        crowd_count, positions = self.analyze_crowd_density(results)
        bottlenecks = self.detect_crowd_bottlenecks(positions, frame.shape)
        sentiment = self.analyze_crowd_sentiment(results, frame)
        
        # Check for emergency conditions
        self.check_emergency_conditions(crowd_count, bottlenecks, sentiment, results)
        
        # Create annotated frame
        annotated_frame = results[0].plot()
        
        # Add crowd analytics overlay
        self.add_analytics_overlay(annotated_frame, crowd_count, sentiment, bottlenecks)
        
        return annotated_frame
    
    def check_emergency_conditions(self, crowd_count, bottlenecks, sentiment, detections):
        """Check various emergency conditions and trigger alerts"""
        
        # High crowd density alert
        if crowd_count > self.crowd_density_threshold:
            self.send_emergency_alert('high_density', {
                'count': crowd_count,
                'risk_level': 'Critical' if crowd_count > self.crowd_density_threshold * 1.5 else 'High'
            })
        
        # Crowd panic detection
        if sentiment < 0.3:
            self.send_emergency_alert('panic', {
                'sentiment': sentiment,
                'count': crowd_count
            }, 'security')
        
        # Bottleneck alerts
        for bottleneck in bottlenecks:
            if bottleneck['count'] >= self.bottleneck_threshold:
                self.send_emergency_alert('bottleneck', bottleneck, 'control_room')
        
        # Fire/smoke detection (if your model detects fire/smoke)
        for detection in detections:
            boxes = detection.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in ['fire', 'smoke'] and confidence > self.fire_smoke_threshold:
                        self.send_emergency_alert('fire', {
                            'confidence': confidence,
                            'type': class_name
                        }, 'medical')
    
    def add_analytics_overlay(self, frame, crowd_count, sentiment, bottlenecks):
        """Add analytics information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Status panel
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # Add text information
        y_offset = 30
        cv2.putText(frame, f"Project Drishti - Crowd Safety", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Crowd Count: {crowd_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        sentiment_color = (0, 255, 0) if sentiment > 0.7 else (0, 255, 255) if sentiment > 0.3 else (0, 0, 255)
        cv2.putText(frame, f"Sentiment: {sentiment:.2f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, sentiment_color, 1)
        
        y_offset += 20
        status = "SAFE" if crowd_count < self.crowd_density_threshold and sentiment > 0.5 else "ALERT"
        status_color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        y_offset += 20
        cv2.putText(frame, f"Bottlenecks: {len(bottlenecks)}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Time stamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 200, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_monitoring(self):
        """Main monitoring loop for crowd safety"""
        logger.info("Starting Project Drishti Crowd Safety Monitoring...")
        logger.info("Press 'q' to quit, 's' to save snapshot")
        
        cap = cv2.VideoCapture(0)  # Use 0 for webcam, or video file path
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame for crowd analysis
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Project Drishti - Crowd Safety System', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    filename = f"drishti_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Snapshot saved: {filename}")
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def generate_safety_report(self):
        """Generate crowd safety report"""
        if not self.crowd_history:
            return "No crowd data available"
        
        total_time = len(self.crowd_history)
        avg_crowd = sum(entry['count'] for entry in self.crowd_history) / total_time
        max_crowd = max(entry['count'] for entry in self.crowd_history)
        
        report = f"""
        PROJECT DRISHTI - CROWD SAFETY REPORT
        ====================================
        Monitoring Duration: {total_time} frames
        Average Crowd Count: {avg_crowd:.1f}
        Peak Crowd Count: {max_crowd}
        Current Sentiment Score: {self.sentiment_score:.2f}
        Safety Status: {'SAFE' if max_crowd < self.crowd_density_threshold else 'ALERT ISSUED'}
        
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report

# Test function for the system
def test_emergency_call():
    """Test emergency calling system"""
    drishti = DrishtiCrowdSafetySystem()
    
    # Test high density alert
    test_details = {
        'count': 25,
        'risk_level': 'Critical'
    }
    
    drishti.send_emergency_alert('high_density', test_details, 'security')
    logger.info("Test emergency call sent!")

if __name__ == "__main__":
    # Initialize Drishti system
    drishti = DrishtiCrowdSafetySystem()
    
    # Choose operation mode:
    
    # Option 1: Run full crowd monitoring (main mode)
    drishti.run_monitoring()
    
    # Option 2: Test emergency calling system (uncomment to test)
    # test_emergency_call()
    
    # Option 3: Generate report (uncomment to test)
    # print(drishti.generate_safety_report())