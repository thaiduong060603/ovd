"""
Ring buffer for continuous video recording
"""
from collections import deque
import numpy as np
from typing import Optional, Tuple, List
import cv2
from pathlib import Path


class VideoRingBuffer:
    """
    Circular buffer to store recent video frames
    """
    
    def __init__(
        self,
        max_seconds: int = 30,
        fps: float = 30.0,
        max_memory_mb: int = 500
    ):
        """
        Initialize ring buffer
        
        Args:
            max_seconds: Maximum seconds of video to store
            fps: Frame rate
            max_memory_mb: Maximum memory usage (MB)
        """
        self.max_seconds = max_seconds
        self.fps = fps
        self.max_frames = int(max_seconds * fps)
        self.max_memory_mb = max_memory_mb
        
        # Circular buffer: (frame, timestamp, frame_id)
        self.buffer: deque = deque(maxlen=self.max_frames)
        
        # Track memory usage
        self.estimated_frame_size_mb = 0
        
        print(f"✓ Ring Buffer: {max_seconds}s @ {fps}fps = {self.max_frames} frames (max {max_memory_mb}MB)")
    
    def add_frame(self, frame: np.ndarray, timestamp: float, frame_id: int):
        """
        Add frame to ring buffer
        
        Args:
            frame: Video frame (numpy array)
            timestamp: Frame timestamp
            frame_id: Frame number
        """
        # Estimate memory usage on first frame
        if self.estimated_frame_size_mb == 0 and frame is not None:
            frame_bytes = frame.nbytes
            self.estimated_frame_size_mb = frame_bytes / (1024 * 1024)
            total_mb = self.estimated_frame_size_mb * self.max_frames
            
            if total_mb > self.max_memory_mb:
                # Reduce buffer size to fit memory constraint
                self.max_frames = int(self.max_memory_mb / self.estimated_frame_size_mb)
                self.buffer = deque(maxlen=self.max_frames)
                print(f"⚠️  Adjusted buffer size to {self.max_frames} frames (~{self.max_memory_mb}MB)")
        
        # Store copy of frame
        self.buffer.append((frame.copy(), timestamp, frame_id))
    
    def extract_clip(
        self,
        trigger_timestamp: float,
        pre_seconds: float,
        post_seconds: float,
        output_path: str,
        fps: Optional[float] = None
    ) -> Optional[str]:
        """
        Extract video clip around trigger time
        
        Args:
            trigger_timestamp: When incident occurred
            pre_seconds: Seconds before trigger to include
            post_seconds: Seconds after trigger to include
            output_path: Where to save video
            fps: Output FPS (default: same as buffer)
        
        Returns:
            Path to saved video or None if failed
        """
        if not self.buffer:
            print("Warning: Ring buffer is empty")
            return None
        
        fps = fps or self.fps
        
        # Calculate time range
        start_time = trigger_timestamp - pre_seconds
        end_time = trigger_timestamp + post_seconds
        
        # Filter frames in time range
        clip_frames = [
            (frame, ts, fid) for frame, ts, fid in self.buffer
            if start_time <= ts <= end_time
        ]
        
        if not clip_frames:
            print(f"Warning: No frames found in range [{start_time:.2f}, {end_time:.2f}]")
            return None
        
        print(f"Extracting clip: {len(clip_frames)} frames ({clip_frames[0][1]:.2f}s to {clip_frames[-1][1]:.2f}s)")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write video
        first_frame = clip_frames[0][0]
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return None
        
        for frame, ts, fid in clip_frames:
            writer.write(frame)
        
        writer.release()
        
        print(f"✓ Saved clip: {output_path}")
        return output_path
    
    def get_frame_at_time(self, timestamp: float) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get frame closest to specified timestamp
        
        Returns:
            (frame, frame_id) or None
        """
        if not self.buffer:
            return None
        
        # Find closest frame
        closest = min(self.buffer, key=lambda x: abs(x[1] - timestamp))
        return closest[0], closest[2]
    
    def save_snapshot(
        self,
        timestamp: float,
        output_path: str
    ) -> Optional[str]:
        """
        Save single frame at timestamp
        
        Args:
            timestamp: Timestamp to snapshot
            output_path: Where to save image
        
        Returns:
            Path to saved image or None
        """
        result = self.get_frame_at_time(timestamp)
        
        if result is None:
            print(f"Warning: No frame found at timestamp {timestamp:.2f}")
            return None
        
        frame, frame_id = result
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        success = cv2.imwrite(output_path, frame)
        
        if success:
            print(f"✓ Saved snapshot: {output_path} (frame {frame_id})")
            return output_path
        else:
            print(f"Error: Could not save snapshot to {output_path}")
            return None
    
    def get_buffer_info(self) -> dict:
        """Get buffer statistics"""
        if not self.buffer:
            return {
                'frames': 0,
                'duration_seconds': 0,
                'memory_mb': 0
            }
        
        first_ts = self.buffer[0][1]
        last_ts = self.buffer[-1][1]
        
        return {
            'frames': len(self.buffer),
            'duration_seconds': last_ts - first_ts,
            'memory_mb': self.estimated_frame_size_mb * len(self.buffer),
            'first_timestamp': first_ts,
            'last_timestamp': last_ts
        }