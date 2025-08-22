import React, { useState, useEffect } from 'react';
import { Camera, Wifi, WifiOff, AlertTriangle, Users, Flame, Zap } from 'lucide-react';

const CameraFeed = ({ cameraId, detectionStatus, apiUrl }) => {
  const [isOnline, setIsOnline] = useState(true);
  const [streamUrl, setStreamUrl] = useState(null);

  useEffect(() => {
    fetchStreamUrl();
  }, [cameraId]);

  const fetchStreamUrl = async () => {
    try {
      const response = await fetch(`${apiUrl}/camera/${cameraId}/stream`);
      const data = await response.json();
      setStreamUrl(data.stream_url);
      setIsOnline(true);
    } catch (error) {
      console.error(`Error fetching stream for camera ${cameraId}:`, error);
      setIsOnline(false);
    }
  };

  const hasAlert = () => {
    return detectionStatus.fire || detectionStatus.fight || detectionStatus.stampede || detectionStatus.crowd > 50;
  };

  const getAlertMessage = () => {
    if (detectionStatus.fire) return "🔥 Fire Detected!";
    if (detectionStatus.stampede) return "🚨 Stampede Risk!";
    if (detectionStatus.fight) return "⚠️ Fight Detected!";
    if (detectionStatus.crowd > 50) return `👥 High Crowd Density (${detectionStatus.crowd} people)`;
    return null;
  };

  const getDetectionIcon = (type) => {
    switch (type) {
      case 'fire': return Flame;
      case 'crowd': return Users;
      case 'fight': return AlertTriangle;
      case 'stampede': return Zap;
      default: return AlertTriangle;
    }
  };

  const getStatusClass = (value, type) => {
    if (type === 'crowd') {
      if (value > 50) return 'danger';
      if (value > 30) return 'warning';
      return 'normal';
    }
    return value ? 'danger' : 'normal';
  };

  return (
    <div className={`camera-feed ${hasAlert() ? 'alert' : ''}`}>
      {/* Camera Header */}
      <div className="camera-header">
        <div className="flex items-center">
          <Camera className="h-4 w-4 mr-2" />
          <span>Camera {cameraId}</span>
        </div>
        <div className={`camera-status ${!isOnline ? 'offline' : ''}`}>
          {isOnline ? (
            <>
              <Wifi className="h-3 w-3 mr-1 inline" />
              Live
            </>
          ) : (
            <>
              <WifiOff className="h-3 w-3 mr-1 inline" />
              Offline
            </>
          )}
        </div>
      </div>

      {/* Video Container */}
      <div className="video-container">
        {isOnline ? (
          <>
            {/* Simulated video feed - in production, this would be the actual stream */}
            <div className="video-placeholder">
              <div className="text-center">
                <Camera className="h-12 w-12 mb-2 mx-auto opacity-50" />
                <div className="text-sm">Camera {cameraId} Feed</div>
                <div className="text-xs mt-1 opacity-75">
                  ThinkSpeak Integration Active
                </div>
              </div>
            </div>
            
            {/* Alert Overlay */}
            {hasAlert() && (
              <div className="detection-overlay">
                <div className="alert-message">
                  {getAlertMessage()}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <div className="text-center">
              <WifiOff className="h-12 w-12 mb-2 mx-auto opacity-50" />
              <div className="text-sm">Camera Offline</div>
            </div>
          </div>
        )}
      </div>

      {/* Detection Stats */}
      <div className="detection-stats">
        <div className="stat-item">
          <span className="stat-label flex items-center">
            <Flame className="h-3 w-3 mr-1" />
            Fire
          </span>
          <span className={`stat-value ${getStatusClass(detectionStatus.fire, 'fire')}`}>
            {detectionStatus.fire ? 'Detected' : 'Clear'}
          </span>
        </div>
        
        <div className="stat-item">
          <span className="stat-label flex items-center">
            <Users className="h-3 w-3 mr-1" />
            Crowd
          </span>
          <span className={`stat-value ${getStatusClass(detectionStatus.crowd, 'crowd')}`}>
            {detectionStatus.crowd} people
          </span>
        </div>
        
        <div className="stat-item">
          <span className="stat-label flex items-center">
            <AlertTriangle className="h-3 w-3 mr-1" />
            Fight
          </span>
          <span className={`stat-value ${getStatusClass(detectionStatus.fight, 'fight')}`}>
            {detectionStatus.fight ? 'Detected' : 'Clear'}
          </span>
        </div>
        
        <div className="stat-item">
          <span className="stat-label flex items-center">
            <Zap className="h-3 w-3 mr-1" />
            Stampede Risk
          </span>
          <span className={`stat-value ${getStatusClass(detectionStatus.stampede, 'stampede')}`}>
            {detectionStatus.stampede ? 'High' : 'Low'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;