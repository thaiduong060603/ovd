"""
Notification manager - coordinates multiple notifiers
"""
from typing import List, Dict, Any
from pathlib import Path

from src.core.notify.base_notifier import BaseNotifier, NotificationPayload
from src.core.notify.console_notifier import ConsoleNotifier
from src.core.notify.slack_notifier import SlackNotifier
from src.core.notify.email_notifier import EmailNotifier
from src.core.notify.mqtt_notifier import MQTTNotifier
from src.models.rule import Incident, Rule


class NotificationManager:
    """
    Manages multiple notification channels
    """
    
    def __init__(self, config_path: str = "configs/notifications"):
        """
        Initialize notification manager
        
        Args:
            config_path: Path to notifications config directory
        """
        self.notifiers: Dict[str, BaseNotifier] = {}
        self.config_path = Path(config_path)
        
        print(f"\n[Notification Manager] Loading notifiers...")
        self._load_notifiers()
    
    def _load_notifiers(self):
        """Load all configured notifiers"""
        import yaml
        
        # Always load Console (for testing)
        console_config_path = self.config_path / "console.yaml"
        if console_config_path.exists():
            try:
                with open(console_config_path) as f:
                    config = yaml.safe_load(f)
                    if config.get('enabled', True):
                        self.notifiers['console'] = ConsoleNotifier(config)
            except Exception as e:
                print(f"⚠️  Could not load Console notifier: {e}")
        else:
            # Use default console notifier
            self.notifiers['console'] = ConsoleNotifier({'enabled': True})
        
        # Load Slack
        slack_config_path = self.config_path / "slack.yaml"
        if slack_config_path.exists():
            try:
                with open(slack_config_path) as f:
                    config = yaml.safe_load(f)
                    if config.get('enabled', False):
                        self.notifiers['slack'] = SlackNotifier(config)
            except Exception as e:
                print(f"⚠️  Could not load Slack notifier: {e}")
        
        # Load Email
        email_config_path = self.config_path / "email.yaml"
        if email_config_path.exists():
            try:
                with open(email_config_path) as f:
                    config = yaml.safe_load(f)
                    if config.get('enabled', False):
                        self.notifiers['email'] = EmailNotifier(config)
            except Exception as e:
                print(f"⚠️  Could not load Email notifier: {e}")
        
        # Load MQTT
        mqtt_config_path = self.config_path / "mqtt.yaml"
        if mqtt_config_path.exists():
            try:
                with open(mqtt_config_path) as f:
                    config = yaml.safe_load(f)
                    if config.get('enabled', False):
                        self.notifiers['mqtt'] = MQTTNotifier(config)
            except Exception as e:
                print(f"⚠️  Could not load MQTT notifier: {e}")
        
        if not self.notifiers:
            print("⚠️  No notifiers enabled")
        else:
            print(f"✓ Loaded {len(self.notifiers)} notifier(s): {list(self.notifiers.keys())}")
    
    def notify(self, incident: Incident, rule: Rule) -> Dict[str, bool]:
        """
        Send notifications for incident
        
        Args:
            incident: Incident object
            rule: Associated Rule object
        
        Returns:
            Dict of channel -> success status
        """
        # Build payload
        payload = NotificationPayload(
            incident_id=incident.incident_id,
            rule_id=rule.rule_id,
            rule_description=rule.description,
            track_id=incident.track_id,
            confirmed_time=incident.confirmed_time or incident.first_detected_time,
            snapshot_path=incident.snapshots[0] if incident.snapshots else None,
            video_clip_path=incident.video_clip_path,
            avg_confidence=sum(incident.confidence_scores) / len(incident.confidence_scores) if incident.confidence_scores else 0,
            camera_id=rule.area_id,
            location=rule.area_id
        )
        
        # Send to all enabled channels (not just rule-configured ones)
        results = {}
        
        # Send to rule-configured channels
        for channel_name in rule.actions.notify_channels:
            if channel_name in self.notifiers:
                try:
                    success = self.notifiers[channel_name].send(payload)
                    results[channel_name] = success
                except Exception as e:
                    print(f"✗ Error sending to {channel_name}: {e}")
                    results[channel_name] = False
            else:
                print(f"⚠️  Notifier '{channel_name}' not configured or disabled")
                results[channel_name] = False
        
        # Always send to console (if enabled)
        if 'console' in self.notifiers and 'console' not in rule.actions.notify_channels:
            try:
                self.notifiers['console'].send(payload)
            except Exception as e:
                print(f"⚠️  Console notification failed: {e}")
        
        return results