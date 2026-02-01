"""
Home Assistant notification module using REST API
"""
import logging
import requests
from datetime import datetime

import config

logger = logging.getLogger(__name__)


class HomeAssistantNotifier:
    def __init__(self, url=None, token=None, mobile_service=None):
        self.url = (url or config.HA_URL).rstrip("/")
        self.token = token or config.HA_TOKEN
        self.mobile_service = mobile_service or config.HA_MOBILE_NOTIFY_SERVICE
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def _call_service(self, domain, service, data):
        """Call a Home Assistant service"""
        endpoint = f"{self.url}/api/services/{domain}/{service}"
        try:
            response = requests.post(endpoint, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            logger.debug(f"Service {domain}.{service} called successfully")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call {domain}.{service}: {e}")
            return False
    
    def send_persistent_notification(self, message, title="Bird Detected! 🐦"):
        """Send a persistent notification to the HA dashboard"""
        data = {
            "message": message,
            "title": title,
            "notification_id": f"bird_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        return self._call_service("persistent_notification", "create", data)
    
    def send_mobile_notification(self, message, title="Bird Detected! 🐦", image_path=None):
        """Send a push notification to mobile app"""
        # Extract service name parts (e.g., "notify.mobile_app_phone" -> domain="notify", service="mobile_app_phone")
        parts = self.mobile_service.split(".", 1)
        if len(parts) != 2:
            logger.error(f"Invalid mobile service name: {self.mobile_service}")
            return False
        
        domain, service = parts
        
        data = {
            "message": message,
            "title": title,
        }
        
        # Add image if provided (needs to be accessible URL or local path on HA)
        if image_path:
            data["data"] = {
                "image": image_path
            }
        
        return self._call_service(domain, service, data)
    
    def notify_bird_detected(self, confidence, image_url=None):
        """Send both dashboard and mobile notifications for bird detection"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Bird detected at {timestamp} with {confidence:.0%} confidence"
        
        # Send to dashboard
        dashboard_ok = self.send_persistent_notification(message)
        
        # Send to mobile
        mobile_ok = self.send_mobile_notification(message, image_path=image_url)
        
        if dashboard_ok or mobile_ok:
            logger.info(f"Notification sent: {message}")
        
        return dashboard_ok and mobile_ok
    
    def test_connection(self):
        """Test the connection to Home Assistant"""
        try:
            response = requests.get(
                f"{self.url}/api/",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Home Assistant connection test successful")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Home Assistant connection test failed: {e}")
            return False
