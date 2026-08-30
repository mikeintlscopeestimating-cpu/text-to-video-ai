"""Configuration settings for Text-to-Video AI Generator"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Model settings
DEFAULT_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
DEVICE = "cuda"  # or "cpu" if GPU not available
TORCH_DTYPE = "float16"  # or "float32" for CPU

# Video generation settings
DEFAULT_NUM_FRAMES = 25
DEFAULT_HEIGHT = 576
DEFAULT_WIDTH = 1024
DEFAULT_NUM_INFERENCE_STEPS = 25

# Image-to-Video settings
IMAGE_TO_VIDEO_STEPS = 25
IMAGE_GENERATION_STEPS = 50

# Performance settings
ENABLE_ATTENTION_SLICING = True
ENABLE_MEMORY_EFFICIENT_ATTENTION = True

# Output settings
OUTPUT_FORMAT = "mp4"  # or "webm"
OUTPUT_FPS = 8
OUTPUT_QUALITY = "high"  # "low", "medium", "high"

# Logging
LOG_LEVEL = "INFO"
