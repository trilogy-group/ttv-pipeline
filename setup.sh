#!/bin/bash
# Setup script for the TTV Pipeline
# This script installs required dependencies and downloads necessary models

set -e  # Exit on error

echo "Setting up TTV Pipeline environment..."

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install uv for faster package installation
pip install uv

# Install dependencies with optimized settings
echo "Installing dependencies..."
MAX_JOBS=64 uv pip install flash-attn --no-build-isolation
uv pip install -r requirements.txt --no-build-isolation
uv pip install "huggingface_hub[cli]"

# Create directories
mkdir -p frameworks
mkdir -p models
mkdir -p output/frames
mkdir -p output/videos

# Download and organize repositories and models
echo "Setting up frameworks and downloading models (this may take some time)..."

# Clone the Wan2.1 framework repository
echo "Cloning Wan2.1 framework..."
git clone https://github.com/Wan-Ai/Wan2.1.git ./frameworks/Wan2.1

# Download the FLF2V model weights
echo "Downloading FLF2V model weights..."
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir ./models/Wan2.1-FLF2V-14B-720P

# Optional: Download additional models if needed
# Uncomment the lines below if you need these models
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P

# Setup FramePack (optional but recommended)
setup_framepack() {
  echo "Setting up FramePack framework..."
  git clone https://github.com/lllyasviel/FramePack.git ./frameworks/FramePack
  
  # Create FramePack download directory
  mkdir -p ./frameworks/FramePack/hf_download/hub
  
  # Download FramePack models
  echo "Downloading FramePack models (this will take some time)..."
  huggingface-cli download hunyuanvideo-community/HunyuanVideo --local-dir ./frameworks/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo
  huggingface-cli download lllyasviel/FramePackI2V_HY --local-dir ./frameworks/FramePack/hf_download/hub/models--lllyasviel--FramePackI2V_HY
  huggingface-cli download lllyasviel/flux_redux_bfl --local-dir ./frameworks/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl
  
  # Create version file
  echo "1" > ./frameworks/FramePack/hf_download/hub/version.txt
  
  echo "FramePack setup complete!"
}

# Uncomment the line below to also set up FramePack
# setup_framepack

echo "Setup complete! Please update pipeline_config.yaml with your API keys and model paths."
echo "Example usage: python pipeline.py --config pipeline_config.yaml"
echo "NOTE: To use FramePack, uncomment the setup_framepack function call in this script."
