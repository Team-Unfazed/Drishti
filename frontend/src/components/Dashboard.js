import React, { useState, useEffect } from 'react';
import { Camera, AlertTriangle, Users, Flame, Zap, MapPin } from 'lucide-react';
import CameraFeed from './CameraFeed';
import AlertPanel from './AlertPanel';
import StatsPanel from './StatsPanel';
import axios from 'axios';

const Dashboard = ({ alerts, detectionStatus, apiUrl }) => {
  const [routeSuggestions, setRouteSuggestions] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRouteSuggestions();
    const interval = setInterval(fetchRouteSuggestions, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchRouteSuggestions = async () => {
    try {
      const promises = [1, 2, 3].map(cameraId =>
        axios.get(`${apiUrl}/route-suggestions/${cameraId}`)
      );
      const responses = await Promise.all(promises);
      
      const suggestions = {};
      responses.forEach((response, index) => {
        suggestions[index + 1] = response.data;
      });
      
      setRouteSuggestions(suggestions);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching route suggestions:', error);
      setLoading(false);
    }
  };

  const getOverallStats = () => {
    const totalCrowds = Object.values(detectionStatus).reduce((sum, status) => sum + status.crowd, 0);
    const activeAlerts = alerts.filter(alert => 
      new Date().getTime() - new Date(alert.timestamp).getTime() < 300000 // Last 5 minutes
    ).length;
    
    const hasEmergency = alerts.some(alert => 
      alert.severity === 'critical' && 
      new Date().getTime() - new Date(alert.timestamp).getTime() < 600000 // Last 10 minutes
    );

    return {
      totalCrowds,
      activeAlerts,
      hasEmergency,
      systemStatus: hasEmergency ? 'emergency' : activeAlerts > 0 ? 'warning' : 'normal'
    };
  };

  const stats = getOverallStats();

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex items-center justify-center h-64">
          <div className="loading-spinner"></div>
          <span className="ml-3 text-gray-600">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <div className="mb-8">
        <h1 className="page-title">Crowd Shield Dashboard</h1>
        <p className="page-subtitle">Real-time crowd and disaster management system</p>
      </div>

      {/* System Status Banner */}
      <div className={`mb-6 p-4 rounded-lg border-l-4 ${
        stats.systemStatus === 'emergency' 
          ? 'bg-red-50 border-red-400 text-red-800' 
          : stats.systemStatus === 'warning'
          ? 'bg-yellow-50 border-yellow-400 text-yellow-800'
          : 'bg-green-50 border-green-400 text-green-800'
      }`}>
        <div className="flex items-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          <span className="font-semibold">
            System Status: {stats.systemStatus.toUpperCase()}
          </span>
          {stats.hasEmergency && (
            <span className="ml-4 text-sm">Emergency protocols activated</span>
          )}
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <StatsPanel
          title="Total People"
          value={stats.totalCrowds}
          icon={Users}
          color="blue"
        />
        <StatsPanel
          title="Active Alerts"
          value={stats.activeAlerts}
          icon={AlertTriangle}
          color={stats.activeAlerts > 0 ? "red" : "green"}
        />
        <StatsPanel
          title="Cameras Online"
          value="3/3"
          icon={Camera}
          color="green"
        />
        <StatsPanel
          title="System Health"
          value="Optimal"
          icon={Zap}
          color="green"
        />
      </div>

      {/* Camera Feeds Grid */}
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <Camera className="h-5 w-5 mr-2" />
          Live Camera Feeds
        </h2>
        <div className="dashboard-grid">
          {[1, 2, 3].map(cameraId => (
            <CameraFeed
              key={cameraId}
              cameraId={cameraId}
              detectionStatus={detectionStatus[cameraId]}
              apiUrl={apiUrl}
            />
          ))}
        </div>
      </div>

      {/* Bottom Section - Alerts and Route Suggestions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Alerts */}
        <div>
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Recent Alerts
          </h2>
          <AlertPanel alerts={alerts} />
        </div>

        {/* Route Suggestions */}
        <div>
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <MapPin className="h-5 w-5 mr-2" />
            Route Suggestions
          </h2>
          <div className="card">
            <div className="card-body">
              {Object.entries(routeSuggestions).map(([cameraId, data]) => (
                <div key={cameraId} className="mb-6 last:mb-0">
                  <div className="flex items-center mb-3">
                    <Camera className="h-4 w-4 mr-2 text-gray-500" />
                    <span className="font-semibold">Camera {cameraId} Area</span>
                    <span className={`ml-auto px-2 py-1 rounded-full text-xs font-medium ${
                      data.crowd_count > 50 
                        ? 'bg-red-100 text-red-800' 
                        : data.crowd_count > 30
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {data.crowd_count} people
                    </span>
                  </div>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {data.suggestions?.map((suggestion, index) => (
                      <li key={index} className="flex items-start">
                        <span className="text-blue-500 mr-2">•</span>
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;