import React, { useState, useEffect } from 'react';
import { Upload, Search, Camera, Clock, CheckCircle, AlertCircle, User, MapPin } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { useToast } from '../hooks/use-toast';
import axios from 'axios';

const LostFound = ({ apiUrl }) => {
  const [lostPersons, setLostPersons] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    file: null
  });
  const { toast } = useToast();

  useEffect(() => {
    fetchLostPersons();
  }, []);

  const fetchLostPersons = async () => {
    try {
      const response = await axios.get(`${apiUrl}/lost-persons`);
      setLostPersons(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching lost persons:', error);
      toast({
        title: "Error",
        description: "Failed to fetch lost persons data",
        variant: "destructive"
      });
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setFormData(prev => ({
          ...prev,
          file: file
        }));
      } else {
        toast({
          title: "Invalid File",
          description: "Please select an image file",
          variant: "destructive"
        });
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.file) {
      toast({
        title: "Missing Information",
        description: "Please provide both name and photo",
        variant: "destructive"
      });
      return;
    }

    setUploading(true);
    
    try {
      const submitData = new FormData();
      submitData.append('name', formData.name);
      submitData.append('description', formData.description);
      submitData.append('file', formData.file);

      const response = await axios.post(`${apiUrl}/lost-person`, submitData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      toast({
        title: "Success",
        description: `${formData.name} has been added to the lost persons database. Monitoring all cameras now.`
      });

      // Reset form
      setFormData({
        name: '',
        description: '',
        file: null
      });
      
      // Reset file input
      const fileInput = document.getElementById('photo-upload');
      if (fileInput) fileInput.value = '';
      
      // Refresh list
      fetchLostPersons();
      
    } catch (error) {
      console.error('Error adding lost person:', error);
      toast({
        title: "Error",
        description: "Failed to add lost person. Please try again.",
        variant: "destructive"
      });
    } finally {
      setUploading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getStatusColor = (found) => {
    return found ? 'text-green-600' : 'text-orange-600';
  };

  const getStatusIcon = (found) => {
    return found ? CheckCircle : AlertCircle;
  };

  if (loading) {
    return (
      <div className="page-container">
        <div className="flex items-center justify-center h-64">
          <div className="loading-spinner"></div>
          <span className="ml-3 text-gray-600">Loading lost persons data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <div className="mb-8">
        <h1 className="page-title">Lost & Found</h1>
        <p className="page-subtitle">Upload photos and track lost persons through our camera network</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Form */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Upload className="h-5 w-5 mr-2" />
                Report Lost Person
              </CardTitle>
              <CardDescription>
                Upload a photo and details of the lost person. Our system will monitor all camera feeds automatically.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Person's Name *</Label>
                  <Input
                    id="name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleInputChange}
                    placeholder="Enter the person's name"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description (Optional)</Label>
                  <Textarea
                    id="description"
                    name="description"
                    value={formData.description}
                    onChange={handleInputChange}
                    placeholder="Additional details (clothing, distinctive features, etc.)"
                    rows={3}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="photo-upload">Photo *</Label>
                  <div className="flex items-center justify-center w-full">
                    <label htmlFor="photo-upload" className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <Upload className="w-8 h-8 mb-2 text-gray-500" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span> a clear photo
                        </p>
                        <p className="text-xs text-gray-500">PNG, JPG or JPEG (MAX. 10MB)</p>
                        {formData.file && (
                          <p className="text-xs text-blue-600 mt-1">{formData.file.name}</p>
                        )}
                      </div>
                      <input
                        id="photo-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                        required
                      />
                    </label>
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full"
                  disabled={uploading}
                >
                  {uploading ? (
                    <>
                      <div className="loading-spinner mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <Search className="h-4 w-4 mr-2" />
                      Start Monitoring
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>

        {/* Lost Persons List */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <User className="h-5 w-5 mr-2" />
                Active Cases ({lostPersons.length})
              </CardTitle>
              <CardDescription>
                Currently monitoring these persons across all camera feeds
              </CardDescription>
            </CardHeader>
            <CardContent>
              {lostPersons.length === 0 ? (
                <div className="text-center py-8">
                  <Search className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">No active cases</p>
                  <p className="text-sm text-gray-500">Upload a photo to start monitoring</p>
                </div>
              ) : (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {lostPersons.map((person) => {
                    const StatusIcon = getStatusIcon(person.found);
                    
                    return (
                      <div key={person.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="font-semibold text-lg">{person.name}</h3>
                            <div className="flex items-center mt-1">
                              <StatusIcon className={`h-4 w-4 mr-1 ${getStatusColor(person.found)}`} />
                              <span className={`text-sm font-medium ${getStatusColor(person.found)}`}>
                                {person.found ? 'Found' : 'Searching'}
                              </span>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="flex items-center text-xs text-gray-500 mb-1">
                              <Clock className="h-3 w-3 mr-1" />
                              {formatTimestamp(person.timestamp)}
                            </div>
                          </div>
                        </div>

                        {person.description && (
                          <p className="text-sm text-gray-600 mb-3">{person.description}</p>
                        )}

                        {person.found && person.found_location && (
                          <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-3">
                            <div className="flex items-center text-green-800">
                              <MapPin className="h-4 w-4 mr-2" />
                              <span className="text-sm font-medium">
                                Found at {person.found_location}
                              </span>
                            </div>
                            {person.found_timestamp && (
                              <p className="text-xs text-green-600 mt-1">
                                {formatTimestamp(person.found_timestamp)}
                              </p>
                            )}
                          </div>
                        )}

                        {!person.found && (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mt-3">
                            <div className="flex items-center text-blue-800">
                              <Camera className="h-4 w-4 mr-2" />
                              <span className="text-sm font-medium">
                                Monitoring all 3 camera feeds
                              </span>
                            </div>
                            <p className="text-xs text-blue-600 mt-1">
                              System will alert when person is detected
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default LostFound;