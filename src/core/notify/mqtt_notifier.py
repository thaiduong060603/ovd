"""
MQTT notifier for IoT integration
"""
import json
from typing import Dict, Any

from src.core.notify.base_notifier import BaseNotifier, NotificationPayload


class MQTTNotifier(BaseNotifier):
    """
    Send notifications via MQTT (for IoT systems)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MQTT notifier
        
        Config format:
        {
            'enabled': True,
            'broker': 'mqtt.eclipse.org',
            'port': 1883,
            'topic': 'ovd/alerts',
            'username': 'user',  # Optional
            'password': 'pass',  # Optional
            'qos': 1,
            'retain': False
        }
        """
        super().__init__(config)
        
        self.broker = config.get('broker')
        self.port = config.get('port', 1883)
        self.topic = config.get('topic', 'ovd/alerts')
        self.username = config.get('username')
        self.password = config.get('password')
        self.qos = config.get('qos', 1)
        self.retain = config.get('retain', False)
        
        if not self.broker:
            raise ValueError("MQTT broker is required")
        
        # Initialize MQTT client
        try:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client()
            
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            print(f"✓ MQTT Notifier: {self.broker}:{self.port} → topic: {self.topic}")
            
        except ImportError:
            raise ImportError("paho-mqtt not installed. Install with: pip install paho-mqtt")
    
    def send(self, payload: NotificationPayload) -> bool:
        """
        Send MQTT notification
        
        Args:
            payload: NotificationPayload object
        
        Returns:
            True if successful
        """
        if not self.can_send():
            print(f"⚠️  MQTT: Rate limited, skipping notification")
            return False
        
        try:
            # Connect to broker
            self.client.connect(self.broker, self.port, keepalive=60)
            
            # Build message
            message = self._build_message(payload)
            
            # Publish
            result = self.client.publish(
                self.topic,
                json.dumps(message),
                qos=self.qos,
                retain=self.retain
            )
            
            # Wait for publish to complete
            result.wait_for_publish()
            
            # Disconnect
            self.client.disconnect()
            
            if result.rc == 0:
                self.mark_sent()
                print(f"✓ MQTT notification sent: {payload.incident_id}")
                return True
            else:
                print(f"✗ MQTT error: return code {result.rc}")
                return False
                
        except Exception as e:
            print(f"✗ MQTT notification failed: {e}")
            return False
    
    def _build_message(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Build MQTT message (JSON format)"""
        return {
            'event_type': 'security_alert',
            'incident_id': payload.incident_id,
            'rule_id': payload.rule_id,
            'rule_description': payload.rule_description,
            'track_id': payload.track_id,
            'timestamp': payload.confirmed_time,
            'camera_id': payload.camera_id,
            'location': payload.location,
            'confidence': payload.avg_confidence,
            'evidence': {
                'snapshot': payload.snapshot_path,
                'video': payload.video_clip_path
            }
        }