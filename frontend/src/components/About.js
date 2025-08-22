import React from 'react';
import { Shield, Camera, Brain, Zap, Users, MapPin, Phone, Search } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';

const About = () => {
  const features = [
    {
      icon: Camera,
      title: 'Live Camera Monitoring',
      description: 'Real-time monitoring of 3 camera feeds integrated with ThinkSpeak platform for continuous surveillance.'
    },
    {
      icon: Brain,
      title: 'AI-Powered Detection',
      description: 'Advanced YOLO models for fire detection, crowd counting, fight detection, and stampede prediction.'
    },
    {
      icon: Search,
      title: 'Lost & Found System',
      description: 'Upload photos to automatically track lost persons across all camera feeds using facial recognition.'
    },
    {
      icon: Phone,
      title: 'Emergency Response',
      description: 'One-click emergency calling system with automatic agent notification and incident logging.'
    },
    {
      icon: MapPin,
      title: 'Route Suggestions',
      description: 'Smart crowd-based routing recommendations to avoid congested areas and ensure safe passage.'
    },
    {
      icon: Zap,
      title: 'Real-time Alerts',
      description: 'Instant notifications for all detected incidents with severity-based alert prioritization.'
    }
  ];

  const technologies = [
    { name: 'ThinkSpeak API', description: 'IoT platform for camera feed integration' },
    { name: 'YOLO Models', description: 'State-of-the-art object detection and classification' },
    { name: 'FastAPI', description: 'High-performance backend API framework' },
    { name: 'React', description: 'Modern responsive user interface' },
    { name: 'WebSocket', description: 'Real-time bidirectional communication' },
    { name: 'MongoDB', description: 'Flexible document database for incident storage' }
  ];

  const stats = [
    { number: '3', label: 'Camera Feeds', description: 'Monitored simultaneously' },
    { number: '6+', label: 'Detection Types', description: 'Fire, crowd, fight, stampede, etc.' },
    { number: '<30s', label: 'Response Time', description: 'Emergency alert processing' },
    { number: '24/7', label: 'Monitoring', description: 'Continuous system operation' }
  ];

  return (
    <div className="page-container">
      <div className="mb-12 text-center">
        <div className="flex justify-center mb-4">
          <Shield className="h-16 w-16 text-blue-600" />
        </div>
        <h1 className="page-title mb-4">About Crowd Shield</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Advanced crowd and disaster management system powered by AI, providing real-time monitoring, 
          intelligent detection, and rapid emergency response capabilities.
        </p>
      </div>

      {/* System Overview */}
      <div className="mb-12">
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-2">{stat.number}</div>
                  <div className="font-semibold text-gray-900 mb-1">{stat.label}</div>
                  <div className="text-sm text-gray-600">{stat.description}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Features Grid */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-blue-600" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-600">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* How It Works */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">How It Works</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Camera className="h-5 w-5 mr-2 text-blue-600" />
                Data Collection
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                    <span className="text-xs font-bold text-blue-600">1</span>
                  </div>
                  <div>
                    <p className="font-medium">Camera Integration</p>
                    <p className="text-sm text-gray-600">Three cameras connected via ThinkSpeak API provide continuous video feeds</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                    <span className="text-xs font-bold text-blue-600">2</span>
                  </div>
                  <div>
                    <p className="font-medium">Real-time Processing</p>
                    <p className="text-sm text-gray-600">YOLO models analyze video streams for various threat detection</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Brain className="h-5 w-5 mr-2 text-green-600" />
                AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                    <span className="text-xs font-bold text-green-600">3</span>
                  </div>
                  <div>
                    <p className="font-medium">Detection Models</p>
                    <p className="text-sm text-gray-600">Multiple YOLO models detect fire, fights, stampedes, and count crowds</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                    <span className="text-xs font-bold text-green-600">4</span>
                  </div>
                  <div>
                    <p className="font-medium">Alert Generation</p>
                    <p className="text-sm text-gray-600">System generates alerts based on confidence scores and severity levels</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">Technology Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {technologies.map((tech, index) => (
            <div key={index} className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow">
              <h3 className="font-semibold text-gray-900 mb-2">{tech.name}</h3>
              <p className="text-sm text-gray-600">{tech.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Contact Information */}
      <div className="text-center">
        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle>Emergency Contact Information</CardTitle>
            <CardDescription>
              For technical support or system issues
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">System Administrator</span>
                <span className="text-blue-600">(555) 0100</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b">
                <span className="font-medium">Technical Support</span>
                <span className="text-blue-600">support@crowdshield.com</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="font-medium">Emergency Services</span>
                <span className="text-red-600 font-bold">911</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default About;