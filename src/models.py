"""Model management for Text-to-Video AI Generator"""

import torch
from diffusers import StableVideoDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of AI models"""
    
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.models = {}
    
    def load_video_diffusion_model(self, model_id="stabilityai/stable-video-diffusion-img2vid-xt"):
        """Load Stable Video Diffusion model"""
        if model_id not in self.models:
            logger.info(f"Loading model: {model_id}")
            try:
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    variant="fp16" if self.torch_dtype == torch.float16 else None
                )
                pipe = pipe.to(self.device)
                
                # Enable memory efficient attention
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                self.models[model_id] = pipe
                logger.info(f"Model loaded successfully: {model_id}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        return self.models[model_id]
    
    def unload_model(self, model_id):
        """Unload a model to free memory"""
        if model_id in self.models:
            del self.models[model_id]
            torch.cuda.empty_cache()
            logger.info(f"Model unloaded: {model_id}")
    
    def clear_all_models(self):
        """Clear all loaded models"""
        self.models.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("All models cleared")
