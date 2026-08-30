#!/usr/bin/env python3
"""Basic example of text-to-video generation"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import TextToVideoGenerator
from src import config


def main():
    """Generate a sample video of a rising building"""
    
    print("\n" + "="*60)
    print("Text-to-Video AI Generator - Basic Example")
    print("="*60 + "\n")
    
    # Initialize generator
    print("Initializing generator...")
    generator = TextToVideoGenerator(
        device=config.DEVICE,
        torch_dtype="float16"
    )
    
    # Example 1: Rising Building
    print("\n[Example 1] Generating: Rising Building")
    print("-" * 60)
    
    prompt_1 = "A modern glass and steel skyscraper rising from the ground, time-lapse construction, cranes moving, clear blue sky, sunny day"
    
    try:
        video_path_1 = generator.generate(
            text_prompt=prompt_1,
            num_frames=25,
            height=576,
            width=1024,
            num_inference_steps=25
        )
        print(f"✓ Video saved: {video_path_1}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Multiple Videos
    print("\n[Example 2] Generating multiple videos")
    print("-" * 60)
    
    prompts = [
        "A waterfall flowing down a rocky cliff, misty spray, lush green vegetation",
        "Sunrise over a mountain range, golden clouds, birds flying",
        "Ocean waves crashing on a sandy beach, sunset, golden hour"
    ]
    
    try:
        video_paths = generator.generate_multiple(prompts)
        for i, path in enumerate(video_paths, 1):
            if path:
                print(f"✓ Video {i} saved: {path}")
            else:
                print(f"✗ Video {i} failed to generate")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Clean up
    print("\nCleaning up...")
    generator.clear_memory()
    
    print("\n" + "="*60)
    print("Generation complete!")
    print(f"Output files saved to: {config.OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
