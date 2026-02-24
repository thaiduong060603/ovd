"""
Email notifier using SMTP
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any
from pathlib import Path

from src.core.notify.base_notifier import BaseNotifier, NotificationPayload


class EmailNotifier(BaseNotifier):
    """
    Send notifications via Email (SMTP)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Email notifier
        
        Config format:
        {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'from_email': 'ovd-watchdog@company.com',
            'to_emails': ['security@company.com', 'admin@company.com'],
            'rate_limit_seconds': 60
        }
        """
        super().__init__(config)
        
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
        
        if not all([self.smtp_server, self.username, self.password]):
            raise ValueError("Email: smtp_server, username, password are required")
        
        if not self.to_emails:
            raise ValueError("Email: to_emails list is required")
        
        print(f"✓ Email Notifier: {self.smtp_server}:{self.smtp_port} → {len(self.to_emails)} recipients")
    
    def send(self, payload: NotificationPayload) -> bool:
        """
        Send email notification
        
        Args:
            payload: NotificationPayload object
        
        Returns:
            True if successful
        """
        if not self.can_send():
            print(f"⚠️  Email: Rate limited, skipping notification")
            return False
        
        try:
            # Create message
            msg = self._build_message(payload)
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                
                # Send email
                server.send_message(msg)
            
            self.mark_sent()
            print(f"✓ Email notification sent: {payload.incident_id}")
            return True
            
        except Exception as e:
            print(f"✗ Email notification failed: {e}")
            return False
    
    def _build_message(self, payload: NotificationPayload) -> MIMEMultipart:
        """Build email message"""
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🚨 Security Alert: {payload.rule_description}"
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Format timestamp
        time_str = self.format_timestamp(payload.confirmed_time)
        
        # Build HTML body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #dc3545; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .info-table {{ width: 100%; border-collapse: collapse; }}
                .info-table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                .label {{ font-weight: bold; width: 150px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 Security Alert</h2>
                <p>{payload.rule_description}</p>
            </div>
            
            <div class="content">
                <h3>Incident Details</h3>
                <table class="info-table">
                    <tr>
                        <td class="label">Incident ID:</td>
                        <td>{payload.incident_id}</td>
                    </tr>
                    <tr>
                        <td class="label">Time:</td>
                        <td>{time_str}</td>
                    </tr>
                    <tr>
                        <td class="label">Camera:</td>
                        <td>{payload.camera_id}</td>
                    </tr>
                    <tr>
                        <td class="label">Location:</td>
                        <td>{payload.location}</td>
                    </tr>
                    <tr>
                        <td class="label">Track ID:</td>
                        <td>{payload.track_id}</td>
                    </tr>
                    <tr>
                        <td class="label">Confidence:</td>
                        <td>{payload.avg_confidence:.1%}</td>
                    </tr>
                </table>
                
                <h3>Evidence</h3>
                <p>
                    📸 Snapshot: <code>{payload.snapshot_path or 'N/A'}</code><br>
                    🎥 Video: <code>{payload.video_clip_path or 'N/A'}</code>
                </p>
            </div>
            
            <div class="footer">
                <p>OVD Watchdog System - Automated Security Monitoring</p>
            </div>
        </body>
        </html>
        """
        
        # Plain text alternative
        text_body = f"""
Security Alert: {payload.rule_description}

Incident Details:
- Incident ID: {payload.incident_id}
- Time: {time_str}
- Camera: {payload.camera_id}
- Location: {payload.location}
- Track ID: {payload.track_id}
- Confidence: {payload.avg_confidence:.1%}

Evidence:
- Snapshot: {payload.snapshot_path or 'N/A'}
- Video: {payload.video_clip_path or 'N/A'}

---
OVD Watchdog System
        """
        
        # Attach both versions
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach snapshot if exists
        if payload.snapshot_path and Path(payload.snapshot_path).exists():
            try:
                with open(payload.snapshot_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename=Path(payload.snapshot_path).name)
                    msg.attach(img)
            except Exception as e:
                print(f"Warning: Could not attach snapshot: {e}")
        
        return msg