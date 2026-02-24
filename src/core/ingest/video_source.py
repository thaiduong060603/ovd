"""
Video input sources (file, RTSP, USB camera)
"""
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class VideoSource(ABC):
    """Abstract base class for video sources"""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame
        Returns:
            (success, frame) tuple
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release resources"""
        pass
    
    @property
    @abstractmethod
    def fps(self) -> float:
        """Get frames per second"""
        pass
    
    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Get total frame count (if available)"""
        pass


class FileVideoSource(VideoSource):
    """Video file source (MP4, AVI, etc.)"""
    
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0
        
        print(f"Opened video: {path}")
        print(f"  FPS: {self._fps}")
        print(f"  Total frames: {self._frame_count}")
        print(f"  Duration: {self._frame_count / self._fps:.2f}s")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        success, frame = self.cap.read()
        if success:
            self._current_frame += 1
        return success, frame
    
    def release(self):
        self.cap.release()
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def current_frame(self) -> int:
        return self._current_frame


class RTSPVideoSource(VideoSource):
    """RTSP stream source"""
    
    def __init__(self, url: str, buffer_size: int = 1):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        
        # Reduce buffer to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to connect to RTSP stream: {url}")
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if unknown
        
        print(f"Connected to RTSP: {url}")
        print(f"  Estimated FPS: {self._fps}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_count(self) -> int:
        return -1  # Unknown for streams


class USBCameraSource(VideoSource):
    """USB camera source"""
    
    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720):
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera: {device_id}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"Opened USB camera {device_id}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self._fps}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def frame_count(self) -> int:
        return -1  # Unknown for camera


def create_video_source(source: str) -> VideoSource:
    """
    Factory function to create appropriate video source
    
    Args:
        source: Video file path, RTSP URL, or 'usb:0' for camera
    
    Returns:
        VideoSource instance
    """
    if source.startswith('rtsp://'):
        return RTSPVideoSource(source)
    elif source.startswith('usb:'):
        device_id = int(source.split(':')[1])
        return USBCameraSource(device_id)
    else:
        # Assume it's a file path
        return FileVideoSource(source)