"""
Basic unit tests for video utilities module.
"""

import pytest
import os
import cv2
import numpy as np
from pathlib import Path

from src.video_utils import get_video_properties

@pytest.mark.unit
class TestVideoUtils:
    """Tests for video utility functions"""
    
    def test_get_video_properties(self):
        """Test getting video properties from a temporary video file"""
        # Create a temporary video file
        test_dir = Path("tests/test_data")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        video_path = test_dir / "test_video.avi"
        
        # Only create the video if it doesn't exist
        if not video_path.exists():
            # Create a small test video with specified properties
            width, height = 320, 240
            fps = 24
            num_frames = 10
            
            # Create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # Write some frames (random color frames)
            for _ in range(num_frames):
                # Create a random color frame
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                writer.write(frame)
            
            writer.release()
        
        # Now test the get_video_properties function
        cap = cv2.VideoCapture(str(video_path))
        fps, total_frames, width, height = get_video_properties(cap)
        
        # Assert basic properties
        assert width > 0, "Video width should be positive"
        assert height > 0, "Video height should be positive"
        assert fps > 0, "Video FPS should be positive"
        assert total_frames > 0, "Total frames should be positive"
        
        # Clean up
        cap.release()
