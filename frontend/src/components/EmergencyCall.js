import React, { useState } from 'react';
import { Phone, AlertTriangle, Flame, Users, Zap, MapPin, Clock } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Textarea } from './ui/textarea';
import { useToast } from './ui/use-toast';
import axios from 'axios';

const EmergencyCall = ({ apiUrl }) => {
  const [calling, setCalling] = useState(false);
  const [emergencyType, setEmergencyType] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [description, setDescription] = useState('');
  const { toast } = useToast();

  const emergencyTypes = [
    { value: 'fire', label: 'Fire Emergency', icon: Flame, color: 'text-red-600' },
    { value: 'stampede', label: 'Stampede/Crowd Crush', icon: Zap, color: 'text-orange-600' },
    { value: 'fight', label: 'Violence/Fight', icon: AlertTriangle, color: 'text-yellow-600' },
    { value: 'medical', label: 'Medical Emergency', icon: AlertTriangle, color: 'text-blue-600' },
    { value: 'security', label: 'Security Threat', icon: AlertTriangle, color: 'text-purple-600' },
    { value: 'other', label: 'Other Emergency', icon: AlertTriangle, color: 'text-gray-600' }
  ];

  const cameraLocations = [
    { value: '1', label: 'Camera 1 - Main Entrance' },
    { value: '2', label: 'Camera 2 - Central Area' },
    { value: '3', label: 'Camera 3 - Exit Zone' }
  ];

  const handleEmergencyCall = async () => {
    if (!emergencyType || !cameraId) {
      toast({
        title: "Missing Information",
        description: "Please select emergency type and camera location",
        variant: "destructive"
      });
      return;
    }

    setCalling(true);

    try {
      const response = await axios.post(`${apiUrl}/emergency/call`, {
        alert_type: emergencyType,
        camera_id: parseInt(cameraId),
        description: description
      });

      toast({
        title: "Emergency Call Initiated",
        description: "Emergency services have been notified. Help is on the way!",
        variant: "default"
      });

      // Reset form
      setEmergencyType('');
      setCameraId('');
      setDescription('');

    } catch (error) {
      console.error('Error making emergency call:', error);
      toast({
        title: "Call Failed",
        description: "Failed to contact emergency services. Please try again or call directly.",
        variant: "destructive"
      });
    } finally {
      setCalling(false);
    }
  };

  const selectedEmergencyType = emergencyTypes.find(type => type.value === emergencyType);

  return (
    <div className="page-container">
      <div className="mb-8">
        <h1 className="page-title text-red-600">Emergency Call System</h1>
        <p className="page-subtitle">Immediately alert emergency services and security teams</p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Emergency Call Form */}
          <div>
            <Card className="border-red-200">
              <CardHeader className="bg-red-50">
                <CardTitle className="flex items-center text-red-800">
                  <Phone className="h-5 w-5 mr-2" />
                  Emergency Call
                </CardTitle>
                <CardDescription className="text-red-600">
                  For immediate emergency response. This will alert all emergency services.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-6">
                  {/* Emergency Type */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Emergency Type *
                    </label>
                    <Select value={emergencyType} onValueChange={setEmergencyType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select emergency type" />
                      </SelectTrigger>
                      <SelectContent>
                        {emergencyTypes.map((type) => {
                          const Icon = type.icon;
                          return (
                            <SelectItem key={type.value} value={type.value}>
                              <div className="flex items-center">
                                <Icon className={`h-4 w-4 mr-2 ${type.color}`} />
                                {type.label}
                              </div>
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Camera Location */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Location *
                    </label>
                    <Select value={cameraId} onValueChange={setCameraId}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select camera location" />
                      </SelectTrigger>
                      <SelectContent>
                        {cameraLocations.map((location) => (
                          <SelectItem key={location.value} value={location.value}>
                            <div className="flex items-center">
                              <MapPin className="h-4 w-4 mr-2 text-gray-500" />
                              {location.label}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Description */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700">
                      Additional Details (Optional)
                    </label>
                    <Textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Describe the emergency situation..."
                      rows={3}
                    />
                  </div>

                  {/* Emergency Call Button */}
                  <Button
                    onClick={handleEmergencyCall}
                    disabled={calling || !emergencyType || !cameraId}
                    className="w-full bg-red-600 hover:bg-red-700 text-white py-4 text-lg font-bold"
                    size="lg"
                  >
                    {calling ? (
                      <>
                        <div className="loading-spinner mr-2"></div>
                        Calling Emergency Services...
                      </>
                    ) : (
                      <>
                        <Phone className="h-5 w-5 mr-2" />
                        CALL EMERGENCY SERVICES
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Information Panel */}
          <div className="space-y-6">
            {/* Current Selection */}
            {(emergencyType || cameraId) && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Emergency Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {selectedEmergencyType && (
                      <div className="flex items-center">
                        <selectedEmergencyType.icon 
                          className={`h-5 w-5 mr-3 ${selectedEmergencyType.color}`} 
                        />
                        <span className="font-medium">{selectedEmergencyType.label}</span>
                      </div>
                    )}
                    
                    {cameraId && (
                      <div className="flex items-center">
                        <MapPin className="h-5 w-5 mr-3 text-gray-500" />
                        <span>
                          {cameraLocations.find(loc => loc.value === cameraId)?.label}
                        </span>
                      </div>
                    )}

                    {description && (
                      <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-700">{description}</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Emergency Contacts */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Emergency Contacts</CardTitle>
                <CardDescription>
                  Direct emergency numbers (if system is unavailable)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="font-medium">Police</span>
                    <span className="text-blue-600 font-bold">911</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="font-medium">Fire Department</span>
                    <span className="text-red-600 font-bold">911</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="font-medium">Medical Emergency</span>
                    <span className="text-green-600 font-bold">911</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="font-medium">Security Control</span>
                    <span className="text-gray-600 font-bold">(555) 0123</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Status */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center">
                  <Clock className="h-4 w-4 mr-2" />
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Emergency System</span>
                    <span className="text-green-600 font-medium">Online</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Camera Network</span>
                    <span className="text-green-600 font-medium">3/3 Active</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Response Time</span>
                    <span className="text-blue-600 font-medium">< 30 seconds</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmergencyCall;