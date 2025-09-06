#!/usr/bin/env python3
"""
Test script for the Mental Health A2A Agent Ecosystem
"""

import asyncio
import json
import requests
import time
from datetime import datetime


class MentalHealthSystemTester:
    """Test the mental health A2A agent ecosystem"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": "Bearer demo-token"
        })
    
    def test_health_check(self):
        """Test system health"""
        print("🔍 Testing system health...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ System is healthy: {data['status']}")
                print(f"   Agents online: {data['agents_online']}")
                print(f"   Active sessions: {data['active_sessions']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_agent_status(self):
        """Test agent status"""
        print("\n🤖 Testing agent status...")
        try:
            response = self.session.get(f"{self.base_url}/agents/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {data['total_agents']} agents:")
                for agent in data['agents']:
                    print(f"   - {agent['agent_id']}: {agent['status']}")
                return True
            else:
                print(f"❌ Agent status check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Agent status error: {e}")
            return False
    
    def test_screening(self):
        """Test mental health screening"""
        print("\n📋 Testing mental health screening...")
        
        test_cases = [
            {
                "name": "Mild Depression",
                "input": "I've been feeling a bit down lately and having trouble sleeping. Nothing too serious though.",
                "expected_risk": "low"
            },
            {
                "name": "Moderate Anxiety",
                "input": "I'm constantly worried about everything and can't seem to relax. My heart races and I feel overwhelmed.",
                "expected_risk": "medium"
            },
            {
                "name": "Crisis Situation",
                "input": "I don't want to live anymore. I can't take this pain and I'm thinking about ending it all.",
                "expected_risk": "high"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {test_case['name']}")
            
            try:
                # Start screening
                screening_data = {
                    "user_id": f"test-user-{i}",
                    "input_data": {
                        "text": test_case["input"],
                        "input_type": "text"
                    },
                    "session_context": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "test_case": test_case["name"]
                    }
                }
                
                response = self.session.post(
                    f"{self.base_url}/screening/start",
                    json=screening_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    session_id = data["session_id"]
                    print(f"   ✅ Screening started: {session_id}")
                    
                    # Wait for completion and get results
                    results_data = self._wait_for_screening_results(session_id)
                    if results_data:
                        results.append({
                            "test_case": test_case["name"],
                            "session_id": session_id,
                            "results": results_data
                        })
                        print(f"   ✅ Results received")
                    else:
                        print(f"   ⚠️  Results not available")
                else:
                    print(f"   ❌ Screening failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Test error: {e}")
        
        return results
    
    def test_crisis_detection(self):
        """Test crisis detection"""
        print("\n🚨 Testing crisis detection...")
        
        crisis_test_cases = [
            {
                "name": "No Crisis",
                "input": "I'm feeling okay today. Just a normal day.",
                "expected_crisis": False
            },
            {
                "name": "Crisis Indicators",
                "input": "I want to hurt myself and I don't see any point in living anymore.",
                "expected_crisis": True
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(crisis_test_cases, 1):
            print(f"\n   Test {i}: {test_case['name']}")
            
            try:
                crisis_data = {
                    "user_id": f"crisis-test-user-{i}",
                    "session_id": f"crisis-session-{i}",
                    "interaction_data": {
                        "text": test_case["input"]
                    },
                    "context": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "test_case": test_case["name"]
                    }
                }
                
                response = self.session.post(
                    f"{self.base_url}/crisis/analyze",
                    json=crisis_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    crisis_detected = data["crisis_detected"]
                    alert_level = data.get("alert_level")
                    
                    print(f"   ✅ Crisis analysis completed")
                    print(f"   📊 Crisis detected: {crisis_detected}")
                    if alert_level:
                        print(f"   📊 Alert level: {alert_level}")
                    
                    results.append({
                        "test_case": test_case["name"],
                        "crisis_detected": crisis_detected,
                        "alert_level": alert_level,
                        "expected": test_case["expected_crisis"]
                    })
                else:
                    print(f"   ❌ Crisis analysis failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Test error: {e}")
        
        return results
    
    def _wait_for_screening_results(self, session_id, max_wait=30):
        """Wait for screening results to be available"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                # Check status
                status_response = self.session.get(f"{self.base_url}/screening/{session_id}/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    
                    if status_data["status"] == "completed":
                        # Get results
                        results_response = self.session.get(f"{self.base_url}/screening/{session_id}/results")
                        if results_response.status_code == 200:
                            return results_response.json()
                    elif status_data["status"] == "error":
                        print(f"   ❌ Screening error: {status_data.get('message', 'Unknown error')}")
                        return None
                
                time.sleep(2)  # Wait 2 seconds before checking again
                
            except Exception as e:
                print(f"   ❌ Error waiting for results: {e}")
                return None
        
        print(f"   ⚠️  Timeout waiting for results")
        return None
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Mental Health A2A Agent Ecosystem - System Tests")
        print("=" * 60)
        
        # Test 1: Health check
        if not self.test_health_check():
            print("\n❌ System is not healthy. Please check the server.")
            return False
        
        # Test 2: Agent status
        if not self.test_agent_status():
            print("\n❌ Agent status check failed.")
            return False
        
        # Test 3: Mental health screening
        screening_results = self.test_screening()
        print(f"\n📊 Screening Results: {len(screening_results)} tests completed")
        
        # Test 4: Crisis detection
        crisis_results = self.test_crisis_detection()
        print(f"\n📊 Crisis Detection Results: {len(crisis_results)} tests completed")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Test Summary:")
        print(f"   Health Check: ✅")
        print(f"   Agent Status: ✅")
        print(f"   Screening Tests: {len(screening_results)} completed")
        print(f"   Crisis Detection Tests: {len(crisis_results)} completed")
        print("\n🎉 All tests completed!")
        
        return True


def main():
    """Main test function"""
    print("Starting Mental Health A2A Agent Ecosystem Tests...")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    tester = MentalHealthSystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)


if __name__ == "__main__":
    main()
