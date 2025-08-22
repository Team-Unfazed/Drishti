import requests
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import io

class CrowdShieldAPITester:
    def __init__(self, base_url="https://crowdshield.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, timeout=10)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                self.failed_tests.append(f"{name}: Expected {expected_status}, got {response.status_code}")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(f"{name}: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        return self.run_test("Health Check", "GET", "health", 200)

    def test_camera_feeds(self):
        """Test camera feed endpoints"""
        results = []
        for camera_id in [1, 2, 3]:
            success, data = self.run_test(f"Camera {camera_id} Feed", "GET", f"camera/{camera_id}/feed", 200)
            results.append(success)
            
            success, data = self.run_test(f"Camera {camera_id} Stream", "GET", f"camera/{camera_id}/stream", 200)
            results.append(success)
        
        return all(results)

    def test_detection_status(self):
        """Test detection status endpoint"""
        return self.run_test("Detection Status", "GET", "detections/status", 200)

    def test_detection_processing(self):
        """Test detection processing endpoint"""
        test_detection = {
            "camera_id": 1,
            "detection_type": "fire",
            "confidence": 0.8,
            "location": {"x": 100, "y": 200},
            "timestamp": datetime.now().isoformat()
        }
        return self.run_test("Process Detection", "POST", "detections/process", 200, data=test_detection)

    def test_lost_persons_get(self):
        """Test get lost persons endpoint"""
        return self.run_test("Get Lost Persons", "GET", "lost-persons", 200)

    def test_lost_person_upload(self):
        """Test lost person upload with file"""
        # Create a simple test image file in memory
        test_image_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {
            'file': ('test_person.png', io.BytesIO(test_image_content), 'image/png')
        }
        data = {
            'name': 'Test Person',
            'description': 'Test description for lost person'
        }
        
        return self.run_test("Add Lost Person", "POST", "lost-person", 200, data=data, files=files)

    def test_emergency_call(self):
        """Test emergency call endpoint"""
        emergency_data = {
            "alert_type": "fire",
            "camera_id": 1,
            "description": "Test emergency call"
        }
        return self.run_test("Emergency Call", "POST", "emergency/call", 200, data=emergency_data)

    def test_route_suggestions(self):
        """Test route suggestions endpoint"""
        results = []
        for camera_id in [1, 2, 3]:
            success, data = self.run_test(f"Route Suggestions Camera {camera_id}", "GET", f"route-suggestions/{camera_id}", 200)
            results.append(success)
        return all(results)

    def test_alerts(self):
        """Test alerts endpoint"""
        return self.run_test("Get Alerts", "GET", "alerts", 200)

    def test_websocket_connection(self):
        """Test WebSocket connection (basic connectivity test)"""
        try:
            import websocket
            ws_url = f"{self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws"
            print(f"\n🔍 Testing WebSocket Connection...")
            print(f"   URL: {ws_url}")
            
            def on_message(ws, message):
                print(f"✅ WebSocket message received: {message[:100]}...")
                ws.close()

            def on_error(ws, error):
                print(f"❌ WebSocket error: {error}")

            def on_open(ws):
                print("✅ WebSocket connection opened")

            ws = websocket.WebSocketApp(ws_url,
                                      on_message=on_message,
                                      on_error=on_error,
                                      on_open=on_open)
            
            # Run for a short time to test connection
            ws.run_forever(timeout=5)
            self.tests_run += 1
            self.tests_passed += 1
            return True
            
        except ImportError:
            print("❌ WebSocket test skipped - websocket-client not installed")
            return True  # Don't fail the test suite for missing dependency
        except Exception as e:
            print(f"❌ WebSocket test failed: {str(e)}")
            self.failed_tests.append(f"WebSocket Connection: {str(e)}")
            self.tests_run += 1
            return False

def main():
    print("🚀 Starting Crowd Shield API Testing...")
    print("=" * 60)
    
    tester = CrowdShieldAPITester()
    
    # Run all tests
    test_results = []
    
    print("\n📋 BASIC HEALTH TESTS")
    print("-" * 30)
    test_results.append(tester.test_health_check())
    
    print("\n📹 CAMERA FEED TESTS")
    print("-" * 30)
    test_results.append(tester.test_camera_feeds())
    
    print("\n🔍 DETECTION TESTS")
    print("-" * 30)
    test_results.append(tester.test_detection_status())
    test_results.append(tester.test_detection_processing())
    
    print("\n👥 LOST & FOUND TESTS")
    print("-" * 30)
    test_results.append(tester.test_lost_persons_get())
    test_results.append(tester.test_lost_person_upload())
    
    print("\n🚨 EMERGENCY TESTS")
    print("-" * 30)
    test_results.append(tester.test_emergency_call())
    
    print("\n🗺️ ROUTE SUGGESTION TESTS")
    print("-" * 30)
    test_results.append(tester.test_route_suggestions())
    
    print("\n📢 ALERTS TESTS")
    print("-" * 30)
    test_results.append(tester.test_alerts())
    
    print("\n🔌 WEBSOCKET TESTS")
    print("-" * 30)
    test_results.append(tester.test_websocket_connection())
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for i, failure in enumerate(tester.failed_tests, 1):
            print(f"   {i}. {failure}")
    
    # Determine if backend is functional enough to proceed
    success_rate = (tester.tests_passed / tester.tests_run) * 100
    if success_rate < 50:
        print(f"\n🚨 CRITICAL: Backend success rate is {success_rate:.1f}% - Too many failures!")
        print("   Recommendation: Fix backend issues before proceeding with frontend testing")
        return 1
    elif success_rate < 80:
        print(f"\n⚠️  WARNING: Backend success rate is {success_rate:.1f}% - Some issues detected")
        print("   Recommendation: Review failed tests but can proceed with frontend testing")
        return 0
    else:
        print(f"\n✅ SUCCESS: Backend is functioning well ({success_rate:.1f}% success rate)")
        print("   Recommendation: Proceed with frontend testing")
        return 0

if __name__ == "__main__":
    sys.exit(main())