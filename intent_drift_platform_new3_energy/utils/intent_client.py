# utils/intent_client.py
import requests
import json

class IntentClient:
    """意图下发客户端"""
    
    def __init__(self, controller_url='http://127.0.0.1:8080'):
        self.controller_url = controller_url
    
    def install_intent(self, intent_config):
        """下发意图"""
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
        """移除意图"""
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
        """检查意图合规性"""
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
        """获取网络状态"""
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
        """列出所有意图"""
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
        """测试与控制器的连接"""
        try:
            response = requests.get(f"{self.controller_url}/state", timeout=5)
            return response.status_code == 200
        except:
            return False
