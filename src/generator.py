"""Main Text-to-Video Generator Class"""

import torch
from PIL import Image
from pathlib import Path
import logging
from .models import ModelManager
from .utils import create_output_filename, save_video, validate_prompt, setup_logging
from . import config

logger = logging.getLogger(__name__)
setup_logging(config.LOG_LEVEL)


class TextToVideoGenerator:
    """Generate videos from text descriptions"""
    
    def __init__(
        self,
        model_id=config.DEFAULT_MODEL,
        device=config.DEVICE,
        torch_dtype="float16"
    ):
        """Initialize the generator
        
        Args:
            model_id: HuggingFace model ID
            device: 'cuda' or 'cpu'
            torch_dtype: 'float16' or 'float32'
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        
        self.model_manager = ModelManager(device=device, torch_dtype=self.torch_dtype)
        self.pipe = None
        
        logger.info(f"TextToVideoGenerator initialized with device: {device}")
    
    def _load_model(self):
        """Load the video diffusion model"""
        if self.pipe is None:
            self.pipe = self.model_manager.load_video_diffusion_model(self.model_id)
    
    def generate(
        self,
        text_prompt,
        num_frames=config.DEFAULT_NUM_FRAMES,
        height=config.DEFAULT_HEIGHT,
        width=config.DEFAULT_WIDTH,
        num_inference_steps=config.DEFAULT_NUM_INFERENCE_STEPS,
        output_path=None,
        image_path=None,
        seed=None
    ):
        """Generate a video from text prompt
        
        Args:
            text_prompt: Description of the video to generate
            num_frames: Number of frames in the output video
            height: Video height in pixels
            width: Video width in pixels
            num_inference_steps: Number of inference steps (higher = better quality, slower)
            output_path: Path to save the output video
            image_path: Path to initial image (optional, for image-to-video)
            seed: Random seed for reproducibility
        
        Returns:
            Path to generated video
        """
        # Validate inputs
        validate_prompt(text_prompt)
        
        # Load model if not already loaded
        self._load_model()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate output path if not provided
        if output_path is None:
            output_filename = create_output_filename(text_prompt)
            output_path = config.OUTPUT_DIR / output_filename
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating video for prompt: {text_prompt}")
        logger.info(f"Output will be saved to: {output_path}")
        
        try:
            # If image path is provided, use image-to-video
            if image_path:
                logger.info(f"Using image-to-video mode with initial image: {image_path}")
                initial_image = Image.open(image_path).convert('RGB')
                initial_image = initial_image.resize((width, height), Image.Resampling.LANCZOS)
                
                with torch.no_grad():
                    frames = self.pipe(
                        initial_image,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps
                    ).frames[0]
            else:
                # Text-to-video generation
                logger.info("Using text-to-video generation mode")
                with torch.no_grad():
                    frames = self.pipe(
                        text_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps
                    ).frames[0]
            
            # Save video
            video_path = save_video(frames, output_path, fps=config.OUTPUT_FPS)
            logger.info(f"Video generation completed successfully!")
            
            return video_path
        
        except Exception as e:
            logger.error(f"Error during video generation: {e}")
            raise
    
    def generate_multiple(
        self,
        prompts,
        num_frames=config.DEFAULT_NUM_FRAMES,
        height=config.DEFAULT_HEIGHT,
        width=config.DEFAULT_WIDTH
    ):
        """Generate multiple videos from a list of prompts
        
        Args:
            prompts: List of text prompts
            num_frames: Number of frames per video
            height: Video height
            width: Video width
        
        Returns:
            List of paths to generated videos
        """
        results = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating video {i}/{len(prompts)}: {prompt}")
            try:
                video_path = self.generate(
                    prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width
                )
                results.append(video_path)
            except Exception as e:
                logger.error(f"Failed to generate video for prompt: {prompt}")
                results.append(None)
        
        return results
    
    def clear_memory(self):
        """Clear GPU memory"""
        self.model_manager.clear_all_models()
        self.pipe = None
        torch.cuda.empty_cache()
        logger.info("Memory cleared")
