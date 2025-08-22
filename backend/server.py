from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import asyncio
import aiohttp
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64
from io import BytesIO
from PIL import Image

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ThinkSpeak Configuration
THINKSPEAK_API_KEY = "0C1T4FVVTH6F4HNL"
THINKSPEAK_CHANNEL_ID = "3043183"
THINKSPEAK_BASE_URL = "https://api.thingspeak.com"

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Pydantic Models
class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: int
    alert_type: str  # fire, stampede, fight, crowd_overflow
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "medium"  # low, medium, high, critical

class LostPerson(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    image_path: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    found: bool = False
    found_location: Optional[str] = None
    found_timestamp: Optional[datetime] = None

class DetectionResult(BaseModel):
    camera_id: int
    detection_type: str
    confidence: float
    location: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Detection status storage
detection_status = {
    1: {"fire": False, "crowd": 0, "fight": False, "stampede": False},
    2: {"fire": False, "crowd": 0, "fight": False, "stampede": False},
    3: {"fire": False, "crowd": 0, "fight": False, "stampede": False}
}

# Camera feed endpoints
@api_router.get("/camera/{camera_id}/feed")
async def get_camera_feed(camera_id: int):
    """Get camera feed from ThinkSpeak"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{THINKSPEAK_BASE_URL}/channels/{THINKSPEAK_CHANNEL_ID}/feeds.json"
            params = {
                "api_key": THINKSPEAK_API_KEY,
                "results": 1
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "success", "data": data, "camera_id": camera_id}
                else:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch camera feed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching camera feed: {str(e)}")

@api_router.get("/camera/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Get camera stream URL for ThinkSpeak"""
    stream_url = f"https://api.thingspeak.com/channels/{THINKSPEAK_CHANNEL_ID}/video.mp4?api_key={THINKSPEAK_API_KEY}"
    return {"stream_url": stream_url, "camera_id": camera_id}

# Detection endpoints
@api_router.get("/detections/status")
async def get_detection_status():
    """Get current detection status for all cameras"""
    return detection_status

@api_router.post("/detections/process")
async def process_detection(detection: DetectionResult):
    """Process detection result and trigger alerts"""
    camera_id = detection.camera_id
    detection_type = detection.detection_type
    
    # Update detection status
    if detection_type in detection_status[camera_id]:
        if detection_type == "crowd":
            detection_status[camera_id][detection_type] = int(detection.confidence)
        else:
            detection_status[camera_id][detection_type] = detection.confidence > 0.5
    
    # Create alert if confidence is high
    if detection.confidence > 0.7:
        alert_messages = {
            "fire": f"🔥 Fire Detected on Camera {camera_id}",
            "fight": f"⚠️ Fight Detected on Camera {camera_id}",
            "stampede": f"🚨 Stampede Risk on Camera {camera_id}",
            "crowd": f"👥 High Crowd Density on Camera {camera_id} ({int(detection.confidence)} people)"
        }
        
        alert = Alert(
            camera_id=camera_id,
            alert_type=detection_type,
            message=alert_messages.get(detection_type, f"Alert on Camera {camera_id}"),
            severity="high" if detection.confidence > 0.8 else "medium"
        )
        
        # Save to database
        await db.alerts.insert_one(alert.dict())
        
        # Broadcast to connected clients
        await manager.broadcast(json.dumps({
            "type": "alert",
            "data": alert.dict()
        }))
    
    return {"status": "processed", "alert_triggered": detection.confidence > 0.7}

# Lost & Found endpoints
@api_router.post("/lost-person", response_model=LostPerson)
async def add_lost_person(name: str, description: str = None, file: UploadFile = File(...)):
    """Add a lost person to the database"""
    try:
        # Save uploaded image
        upload_dir = Path("uploads/lost_persons")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_extension = file.filename.split('.')[-1]
        filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create lost person record
        lost_person = LostPerson(
            name=name,
            description=description,
            image_path=str(file_path)
        )
        
        # Save to database
        await db.lost_persons.insert_one(lost_person.dict())
        
        return lost_person
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding lost person: {str(e)}")

@api_router.get("/lost-persons", response_model=List[LostPerson])
async def get_lost_persons():
    """Get all lost persons"""
    lost_persons = await db.lost_persons.find().to_list(1000)
    return [LostPerson(**person) for person in lost_persons]

@api_router.post("/lost-person/{person_id}/found")
async def mark_person_found(person_id: str, camera_id: int):
    """Mark a lost person as found"""
    result = await db.lost_persons.update_one(
        {"id": person_id},
        {
            "$set": {
                "found": True,
                "found_location": f"Camera {camera_id}",
                "found_timestamp": datetime.now(timezone.utc)
            }
        }
    )
    
    if result.matched_count:
        # Broadcast found notification
        await manager.broadcast(json.dumps({
            "type": "person_found",
            "data": {
                "person_id": person_id,
                "camera_id": camera_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }))
        return {"status": "success", "message": "Person marked as found"}
    else:
        raise HTTPException(status_code=404, detail="Person not found")

# Emergency endpoints
@api_router.post("/emergency/call")
async def emergency_call(alert_type: str, camera_id: int, description: str = ""):
    """Trigger emergency call"""
    try:
        # Create emergency alert
        emergency_alert = Alert(
            camera_id=camera_id,
            alert_type="emergency",
            message=f"🚨 Emergency Call: {alert_type} at Camera {camera_id}",
            severity="critical"
        )
        
        # Save to database
        await db.alerts.insert_one(emergency_alert.dict())
        
        # Broadcast emergency
        await manager.broadcast(json.dumps({
            "type": "emergency",
            "data": emergency_alert.dict()
        }))
        
        # Here you would integrate with your emergency call system
        # For now, we'll simulate the call
        
        return {
            "status": "success", 
            "message": "Emergency services have been notified",
            "alert_id": emergency_alert.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing emergency call: {str(e)}")

# Route suggestions endpoint
@api_router.get("/route-suggestions/{camera_id}")
async def get_route_suggestions(camera_id: int):
    """Get route suggestions based on current crowd density"""
    try:
        # Get current crowd data
        crowd_data = detection_status.get(camera_id, {}).get("crowd", 0)
        
        # Simple route suggestion logic
        suggestions = []
        if crowd_data > 50:
            suggestions = [
                f"Avoid Camera {camera_id} area - High crowd density",
                "Use alternative routes through less crowded areas",
                "Consider waiting 10-15 minutes for crowd to disperse"
            ]
        elif crowd_data > 30:
            suggestions = [
                f"Camera {camera_id} area moderately crowded",
                "Proceed with caution",
                "Monitor crowd levels"
            ]
        else:
            suggestions = [
                f"Camera {camera_id} area is clear",
                "Normal traffic flow",
                "Safe to proceed"
            ]
        
        return {"camera_id": camera_id, "crowd_count": crowd_data, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting route suggestions: {str(e)}")

# WebSocket endpoint
@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic status updates
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "data": detection_status
            }))
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Alerts endpoints
@api_router.get("/alerts", response_model=List[Alert])
async def get_alerts():
    """Get recent alerts"""
    alerts = await db.alerts.find().sort("timestamp", -1).limit(50).to_list(50)
    return [Alert(**alert) for alert in alerts]

@api_router.delete("/alerts/{alert_id}")
async def dismiss_alert(alert_id: str):
    """Dismiss an alert"""
    result = await db.alerts.delete_one({"id": alert_id})
    if result.deleted_count:
        return {"status": "success", "message": "Alert dismissed"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

# Health check
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Crowd Shield Backend Started")
    # Create uploads directory
    Path("uploads/lost_persons").mkdir(parents=True, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()