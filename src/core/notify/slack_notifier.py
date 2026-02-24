"""
Slack webhook notifier
"""
import requests
import json
from typing import Dict, Any
from pathlib import Path

from src.core.notify.base_notifier import BaseNotifier, NotificationPayload


class SlackNotifier(BaseNotifier):
    """
    Send notifications via Slack webhook
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack notifier
        
        Config format:
        {
            'enabled': True,
            'webhook_url': 'https://hooks.slack.com/services/...',
            'channel': '#alerts',
            'username': 'OVD Watchdog',
            'rate_limit_seconds': 60
        }
        """
        super().__init__(config)
        
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'OVD Watchdog')
        self.icon_emoji = config.get('icon_emoji', ':rotating_light:')
        
        if not self.webhook_url:
            raise ValueError("Slack webhook_url is required")
        
        print(f"✓ Slack Notifier: channel={self.channel}")
    
    def send(self, payload: NotificationPayload) -> bool:
        """
        Send Slack notification
        
        Args:
            payload: NotificationPayload object
        
        Returns:
            True if successful
        """
        if not self.can_send():
            print(f"⚠️  Slack: Rate limited, skipping notification")
            return False
        
        try:
            # Build Slack message
            message = self._build_message(payload)
            
            # Send to Slack
            response = requests.post(
                self.webhook_url,
                json=message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.mark_sent()
                print(f"✓ Slack notification sent: {payload.incident_id}")
                return True
            else:
                print(f"✗ Slack error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Slack notification failed: {e}")
            return False
    
    def _build_message(self, payload: NotificationPayload) -> Dict[str, Any]:
        """Build Slack message format"""
        
        # Format timestamp
        time_str = self.format_timestamp(payload.confirmed_time)
        
        # Build attachment color based on confidence
        if payload.avg_confidence >= 0.8:
            color = "danger"  # Red - high confidence
        elif payload.avg_confidence >= 0.6:
            color = "warning"  # Orange - medium confidence
        else:
            color = "#808080"  # Gray - low confidence
        
        # Build message blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Security Alert: {payload.rule_description}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Incident ID:*\n{payload.incident_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{time_str}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Camera:*\n{payload.camera_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Location:*\n{payload.location}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Track ID:*\n{payload.track_id}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Confidence:*\n{payload.avg_confidence:.1%}"
                    }
                ]
            }
        ]
        
        # Add evidence section
        if payload.snapshot_path or payload.video_clip_path:
            evidence_text = "*Evidence:*\n"
            if payload.snapshot_path:
                evidence_text += f"📸 Snapshot: `{payload.snapshot_path}`\n"
            if payload.video_clip_path:
                evidence_text += f"🎥 Video: `{payload.video_clip_path}`"
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": evidence_text
                }
            })
        
        # Add divider
        blocks.append({"type": "divider"})
        
        # Build final message
        message = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "text": f"🚨 Security Alert: {payload.rule_description}",
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "footer": "OVD Watchdog System",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                    "ts": int(payload.confirmed_time)
                }
            ]
        }
        
        return message