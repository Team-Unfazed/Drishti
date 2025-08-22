import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css';
import Dashboard from './components/Dashboard';
import LostFound from './components/LostFound';
import EmergencyCall from './components/EmergencyCall';
import About from './components/About';
import Navbar from './components/Navbar';
import { Toaster } from './components/ui/toaster';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [alerts, setAlerts] = useState([]);
  const [detectionStatus, setDetectionStatus] = useState({
    1: { fire: false, crowd: 0, fight: false, stampede: false },
    2: { fire: false, crowd: 0, fight: false, stampede: false },
    3: { fire: false, crowd: 0, fight: false, stampede: false }
  });

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const wsUrl = `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'status_update') {
        setDetectionStatus(data.data);
      } else if (data.type === 'alert') {
        setAlerts(prev => [data.data, ...prev.slice(0, 9)]);
      } else if (data.type === 'emergency') {
        setAlerts(prev => [data.data, ...prev.slice(0, 9)]);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="App min-h-screen bg-gray-50">
      <BrowserRouter>
        <Navbar />
        <main className="pt-16">
          <Routes>
            <Route 
              path="/" 
              element={
                <Dashboard 
                  alerts={alerts} 
                  detectionStatus={detectionStatus}
                  apiUrl={API}
                />
              } 
            />
            <Route 
              path="/lost-found" 
              element={<LostFound apiUrl={API} />} 
            />
            <Route 
              path="/emergency" 
              element={<EmergencyCall apiUrl={API} />} 
            />
            <Route 
              path="/about" 
              element={<About />} 
            />
          </Routes>
        </main>
        <Toaster />
      </BrowserRouter>
    </div>
  );
}

export default App;