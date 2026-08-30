# Text-to-Video AI Generator

A Python-based text-to-video AI generation tool using state-of-the-art diffusion models. Generate stunning videos from simple text descriptions.

## Features

- 🎬 Generate videos from text prompts
- 🚀 Easy-to-use API
- 🤖 Powered by Stable Video Diffusion
- 💾 Multiple output formats (MP4, WebM)
- ⚡ GPU acceleration support
- 📦 Simple installation and setup

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support, optional but recommended)
- 8GB+ RAM (16GB+ recommended for faster generation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mikeintlscopeestimating-cpu/text-to-video-ai.git
cd text-to-video-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.generator import TextToVideoGenerator

# Initialize the generator
generator = TextToVideoGenerator()

# Generate a video
video_path = generator.generate(
    text_prompt="A modern glass building rising from the ground, time-lapse construction",
    num_frames=25,
    height=576,
    width=1024
)

print(f"Video saved to: {video_path}")
```

### Command Line Usage

```bash
python examples/basic_example.py --prompt "A rising building" --output video.mp4
```

## Examples

### Generate a Rising Building Video
```python
from src.generator import TextToVideoGenerator

generator = TextToVideoGenerator()
video = generator.generate(
    text_prompt="A modern skyscraper rising from the ground with cranes and construction workers, time-lapse, clear day",
    num_frames=25,
    output_path="rising_building.mp4"
)
```

### Generate Multiple Videos
```python
prompts = [
    "A building rising from the ground",
    "A sunrise over a cityscape",
    "A water fountain flowing"
]

for prompt in prompts:
    video = generator.generate(prompt)
    print(f"Generated: {video}")
```

## Project Structure

```
text-to-video-ai/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── generator.py
│   ├── models.py
│   └── utils.py
├── examples/
│   └── basic_example.py
├── tests/
│   └── test_generator.py
└── .gitignore
```

## Configuration

Edit `src/config.py` to customize:
- Default model selection
- Video output quality
- Frame dimensions
- Number of inference steps
- CUDA device selection

## Performance Tips

1. **First run**: Model download (~7GB) - may take several minutes
2. **GPU Usage**: Set `device="cuda"` for 10x faster generation
3. **Quality vs Speed**: Increase `num_inference_steps` for better quality (slower)
4. **Memory**: Reduce `height` and `width` if running out of memory

## Supported Models

- **Stable Video Diffusion** (default) - Fast, good quality
- **AnimateDiff** - Longer sequences
- **ModelScope** - Experimental

## Troubleshooting

### Out of Memory Error
```python
# Reduce resolution
generator = TextToVideoGenerator(height=480, width=768)
```

### Slow Generation
```python
# Use fewer inference steps (faster but lower quality)
video = generator.generate(prompt, num_inference_steps=25)
```

### CUDA Not Found
```python
# Fall back to CPU
generator = TextToVideoGenerator(device="cpu")
```

## API Reference

### TextToVideoGenerator

```python
class TextToVideoGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    )
    
    def generate(
        self,
        text_prompt: str,
        num_frames: int = 25,
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        output_path: str = None
    ) -> str
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Video Diffusion
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [OpenAI](https://openai.com/) for CLIP model

## Support

For issues and questions:
- Open an [Issue](https://github.com/mikeintlscopeestimating-cpu/text-to-video-ai/issues)
- Check [Discussions](https://github.com/mikeintlscopeestimating-cpu/text-to-video-ai/discussions)

## Disclaimer

This tool uses AI models for video generation. Generated content may not always match prompts perfectly. Always review generated content for accuracy and appropriateness.

---

**Made with ❤️ for the AI community**
