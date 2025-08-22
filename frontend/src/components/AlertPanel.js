import React from 'react';
import { AlertTriangle, Flame, Users, Zap, Phone, Clock, X } from 'lucide-react';

const AlertPanel = ({ alerts }) => {
  const getAlertIcon = (alertType) => {
    switch (alertType) {
      case 'fire': return Flame;
      case 'fight': return AlertTriangle;
      case 'stampede': return Zap;
      case 'crowd': return Users;
      case 'emergency': return Phone;
      default: return AlertTriangle;
    }
  };

  const getSeverityClass = (severity) => {
    switch (severity) {
      case 'critical': return 'critical';
      case 'high': return 'high';
      case 'medium': return 'medium';
      default: return 'medium';
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes} min ago`;
    
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString();
  };

  if (!alerts || alerts.length === 0) {
    return (
      <div className="alert-panel">
        <div className="p-8 text-center text-gray-500">
          <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium mb-2">No Active Alerts</p>
          <p className="text-sm">All systems are operating normally</p>
        </div>
      </div>
    );
  }

  return (
    <div className="alert-panel">
      <div className="max-h-96 overflow-y-auto">
        {alerts.slice(0, 10).map((alert, index) => {
          const IconComponent = getAlertIcon(alert.alert_type);
          
          return (
            <div key={alert.id || index} className="alert-item">
              <div className={`alert-icon ${getSeverityClass(alert.severity)}`}>
                <IconComponent className="h-5 w-5" />
              </div>
              
              <div className="alert-content">
                <div className="alert-message-text">
                  {alert.message}
                </div>
                <div className="alert-time flex items-center">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatTime(alert.timestamp)}
                  {alert.camera_id && (
                    <span className="ml-2 px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
                      Camera {alert.camera_id}
                    </span>
                  )}
                </div>
              </div>

              {alert.severity === 'critical' && (
                <div className="ml-auto">
                  <span className="inline-flex items-center px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium">
                    CRITICAL
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      {alerts.length > 10 && (
        <div className="p-3 bg-gray-50 text-center text-sm text-gray-600 border-t">
          Showing 10 of {alerts.length} alerts
        </div>
      )}
    </div>
  );
};

export default AlertPanel;