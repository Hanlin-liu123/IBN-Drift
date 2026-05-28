# utils/intent_client.py
import requests
import json

class IntentClient:
    """Intent sent to the client"""
    
    def __init__(self, controller_url='http://127.0.0.1:8080'):
        self.controller_url = controller_url
    
    def install_intent(self, intent_config):
        """Install intent"""
        url = f"{self.controller_url}/intent"
        try:
            response = requests.post(url, json=intent_config, timeout=10)
            print(f"  [DEBUG] Response status: {response.status_code}")
            print(f"  [DEBUG] Response text: {response.text[:200] if response.text else 'empty'}")
            
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                return {'success': False, 'error': f'Status {response.status_code}: {response.text}'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Cannot connect to controller'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def remove_intent(self, intent_id):
        """Remove intent"""
        url = f"{self.controller_url}/intent/{intent_id}"
        try:
            response = requests.delete(url, timeout=10)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                return {'success': False, 'error': f'Status {response.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def check_intent(self, intent_id):
        """Check intent compliance"""
        url = f"{self.controller_url}/intent/{intent_id}/check"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                return {'compliant': False, 'error': f'Status {response.status_code}'}
        except Exception as e:
            return {'compliant': False, 'error': str(e)}
    
    def get_state(self):
        """Get network state"""
        url = f"{self.controller_url}/state"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                return {'error': f'Status {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def list_intents(self):
        """List all intents"""
        url = f"{self.controller_url}/intents"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                return {}
        except Exception as e:
            return {}
    
    def test_connection(self):
        """Test the connection to the controller"""
        try:
            response = requests.get(f"{self.controller_url}/state", timeout=5)
            return response.status_code == 200
        except:
            return False
