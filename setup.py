from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="text-to-video-ai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python-based text-to-video AI generation tool using state-of-the-art diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikeintlscopeestimating-cpu/text-to-video-ai",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "text-to-video=examples.basic_example:main",
        ],
    },
)
