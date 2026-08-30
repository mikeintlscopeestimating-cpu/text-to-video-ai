"""Utility functions for Text-to-Video AI Generator"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_output_filename(prompt, extension="mp4"):
    """Generate a unique output filename based on prompt and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean prompt for filename
    clean_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_prompt = clean_prompt.replace(' ', '_')
    return f"{timestamp}_{clean_prompt}.{extension}"


def save_video(frames, output_path, fps=8):
    """Save a list of PIL images as a video file"""
    if not frames:
        logger.error("No frames to save")
        return None
    
    # Convert PIL images to numpy arrays
    frame_array = np.array([np.array(frame) for frame in frames])
    
    height, width = frame_array[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame in frame_array:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    logger.info(f"Video saved to: {output_path}")
    return str(output_path)


def load_image(image_path, size=(1024, 576)):
    """Load and resize an image"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def validate_prompt(prompt):
    """Validate text prompt"""
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    if len(prompt) > 500:
        logger.warning("Prompt is very long, may affect quality")
    return True


def setup_logging(log_level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
