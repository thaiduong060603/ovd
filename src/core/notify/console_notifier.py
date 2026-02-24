"""
Console notifier - prints alerts to terminal (for testing)
"""
from typing import Dict, Any
from datetime import datetime

from src.core.notify.base_notifier import BaseNotifier, NotificationPayload


class ConsoleNotifier(BaseNotifier):
    """
    Print notifications to console (useful for testing)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.verbose = config.get('verbose', True)
        self.color_output = config.get('color_output', True)
        
        print(f"✓ Console Notifier: verbose={self.verbose}")
    
    def send(self, payload: NotificationPayload) -> bool:
        """Print notification to console"""
        
        # ANSI color codes
        if self.color_output:
            RED = '\033[91m'
            YELLOW = '\033[93m'
            GREEN = '\033[92m'
            BLUE = '\033[94m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
        else:
            RED = YELLOW = GREEN = BLUE = BOLD = RESET = ''
        
        # Print header
        print(f"\n{RED}{BOLD}{'='*70}")
        print(f"🚨 SECURITY ALERT 🚨")
        print(f"{'='*70}{RESET}")
        
        # Print details
        time_str = datetime.fromtimestamp(payload.confirmed_time).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{BOLD}Rule:{RESET} {payload.rule_description}")
        print(f"{BOLD}Incident ID:{RESET} {payload.incident_id}")
        print(f"{BOLD}Time:{RESET} {time_str}")
        print(f"{BOLD}Camera:{RESET} {payload.camera_id}")
        print(f"{BOLD}Location:{RESET} {payload.location}")
        print(f"{BOLD}Track ID:{RESET} {payload.track_id}")
        
        # Confidence with color
        conf_pct = payload.avg_confidence * 100
        if conf_pct >= 80:
            conf_color = RED
        elif conf_pct >= 60:
            conf_color = YELLOW
        else:
            conf_color = GREEN
        
        print(f"{BOLD}Confidence:{RESET} {conf_color}{conf_pct:.1f}%{RESET}")
        
        # Evidence
        if self.verbose:
            print(f"\n{BLUE}{BOLD}Evidence:{RESET}")
            if payload.snapshot_path:
                print(f"  📸 Snapshot: {payload.snapshot_path}")
            if payload.video_clip_path:
                print(f"  🎥 Video: {payload.video_clip_path}")
        
        print(f"{RED}{'='*70}{RESET}\n")
        
        return True