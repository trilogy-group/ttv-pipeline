#!/bin/bash
# Setup script for the TTV Pipeline
# This script installs required dependencies and downloads necessary models
# It may not work automatically end-to-end, but serves as a reference for needed commands to set up the environment

set -e  # Exit on error

# Default configuration
SETUP_ENV=true
SETUP_WAN21=false
SETUP_FRAMEPACK=false
SETUP_DOCKER=false
SETUP_REMOTE_APIS=false

show_help() {
  echo "TTV Pipeline Setup Script"
  echo ""
  echo "Usage: ./setup.sh [options]"
  echo ""
  echo "Options:"
  echo "  --all                  Setup everything (Python env, all backends, Docker)"
  echo "  --env-only             Setup only Python environment (default)"
  echo "  --wan21                Setup Wan2.1 backend and models"
  echo "  --framepack            Setup FramePack backend and models"
  echo "  --docker               Build Docker images"
  echo "  --remote-apis          Setup configuration for remote APIs"
  echo "  --help                 Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./setup.sh --wan21 --docker   # Setup Wan2.1 and build Docker images"
  echo "  ./setup.sh --all              # Setup everything"
}

# Parse command line arguments
while [ "$1" != "" ]; do
  case $1 in
    --all )
      SETUP_WAN21=true
      SETUP_FRAMEPACK=true
      SETUP_DOCKER=true
      SETUP_REMOTE_APIS=true
      ;;
    --env-only )
      SETUP_ENV=true
      SETUP_WAN21=false
      SETUP_FRAMEPACK=false
      SETUP_DOCKER=false
      SETUP_REMOTE_APIS=false
      ;;
    --wan21 )
      SETUP_WAN21=true
      ;;
    --framepack )
      SETUP_FRAMEPACK=true
      ;;
    --docker )
      SETUP_DOCKER=true
      ;;
    --remote-apis )
      SETUP_REMOTE_APIS=true
      ;;
    --help )
      show_help
      exit
      ;;
    * )
      show_help
      exit 1
  esac
  shift
done

echo "Setting up TTV Pipeline environment..."

# Create base directories
mkdir -p frameworks
mkdir -p models
mkdir -p output/frames
mkdir -p output/videos

# Create configuration file if it doesn't exist
if [ ! -f pipeline_config.yaml ]; then
  cp pipeline_config.yaml.sample pipeline_config.yaml
  echo "Created pipeline_config.yaml from template"
fi

# Setup Python environment
if [ "$SETUP_ENV" = true ]; then
  echo "Setting up Python environment..."
  
  # Create virtual environment if it doesn't exist
  if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "Created Python virtual environment"
  fi
  
  # Activate virtual environment
  source .venv/bin/activate
  
  # Install uv for faster package installation
  pip install uv
  
  # Install dependencies with optimized settings
  echo "Installing dependencies..."
  MAX_JOBS=64 uv pip install flash-attn --no-build-isolation
  uv pip install -r requirements.txt --no-build-isolation
  uv pip install "huggingface_hub[cli]"
  
  echo "Python environment setup complete"
fi

# Setup Wan2.1 backend
if [ "$SETUP_WAN21" = true ]; then
  echo "Setting up Wan2.1 backend..."
  
  # Clone the Wan2.1 framework repository if it doesn't exist
  if [ ! -d "./frameworks/Wan2.1" ]; then
    echo "Cloning Wan2.1 framework..."
    git clone https://github.com/Wan-Video/Wan2.1.git ./frameworks/Wan2.1
  else
    echo "Wan2.1 framework already exists, skipping clone"
  fi
  
  # Activate environment for model downloads
  source .venv/bin/activate
  
  # Download the FLF2V model weights if they don't exist
  if [ ! -d "./models/Wan2.1-FLF2V-14B-720P" ]; then
    echo "Downloading FLF2V model weights..."
    huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir ./models/Wan2.1-FLF2V-14B-720P
  else
    echo "FLF2V model already exists, skipping download"
  fi
  
  # Download the I2V model weights for chaining mode if they don't exist
  if [ ! -d "./models/Wan2.1-I2V-14B-720P" ]; then
    echo "Downloading I2V model weights..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
  else
    echo "I2V model already exists, skipping download"
  fi
  
  # Update configuration to use Wan2.1 backend
  sed -i 's|# wan2_dir: ./frameworks/Wan2.1|wan2_dir: ./frameworks/Wan2.1|' pipeline_config.yaml
  
  echo "Wan2.1 backend setup complete"
fi

# Setup FramePack backend
if [ "$SETUP_FRAMEPACK" = true ]; then
  echo "Setting up FramePack framework..."
  
  # Clone the FramePack repository if it doesn't exist
  if [ ! -d "./frameworks/FramePack" ]; then
    git clone https://github.com/lllyasviel/FramePack.git ./frameworks/FramePack
  else
    echo "FramePack framework already exists, skipping clone"
  fi
  
  # Create FramePack download directory
  mkdir -p ./frameworks/FramePack/hf_download/hub
  
  # Activate environment for model downloads
  source .venv/bin/activate
  
  # Download FramePack models if they don't exist
  if [ ! -d "./frameworks/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo" ]; then
    echo "Downloading FramePack models (this will take some time)..."
    huggingface-cli download hunyuanvideo-community/HunyuanVideo --local-dir ./frameworks/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo
  else
    echo "HunyuanVideo model already exists, skipping download"
  fi
  
  if [ ! -d "./frameworks/FramePack/hf_download/hub/models--lllyasviel--FramePackI2V_HY" ]; then
    huggingface-cli download lllyasviel/FramePackI2V_HY --local-dir ./frameworks/FramePack/hf_download/hub/models--lllyasviel--FramePackI2V_HY
  else
    echo "FramePackI2V_HY model already exists, skipping download"
  fi
  
  if [ ! -d "./frameworks/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl" ]; then
    huggingface-cli download lllyasviel/flux_redux_bfl --local-dir ./frameworks/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl
  else
    echo "flux_redux_bfl model already exists, skipping download"
  fi
  
  # Create version file
  echo "1" > ./frameworks/FramePack/hf_download/hub/version.txt
  
  # Update configuration to use FramePack backend
  sed -i 's|# framepack_dir: ./frameworks/FramePack|framepack_dir: ./frameworks/FramePack|' pipeline_config.yaml
  
  echo "FramePack setup complete!"
fi

# Setup Remote API backends
if [ "$SETUP_REMOTE_APIS" = true ]; then
  echo "Setting up Remote API backend configuration..."
  
  # Update remote API settings in configuration
  echo "Configuring remote API settings in pipeline_config.yaml"
  echo ""
  echo "NOTE: You will need to manually add your API keys to pipeline_config.yaml:"
  echo "  - For Runway ML: runway_ml.api_key"
  echo "  - For Google Veo 3: google_veo.project_id and credentials_path"
  echo "  - For Minimax: minimax.api_key"
  echo ""
  echo "You can also set these as environment variables:"
  echo "  - export RUNWAY_API_KEY=your_key_here"
  echo "  - export MINIMAX_API_KEY=your_key_here"
  echo "  - export GOOGLE_APPLICATION_CREDENTIALS=path_to_credentials.json"
  
  echo "Remote API configuration setup complete"
fi

# Build Docker images
if [ "$SETUP_DOCKER" = true ]; then
  echo "Building Docker images..."
  
  # Build base image
  echo "Building base image (ttv-base)..."
  docker build -f Dockerfile.base -t ttv-base:latest .
  
  # Build core API image
  echo "Building core API image (ttv-pipeline:core)..."
  docker build -f Dockerfile.core -t ttv-pipeline:core .
  
  # Build backend-specific images if the backends are set up
  if [ "$SETUP_FRAMEPACK" = true ]; then
    echo "Building FramePack image (ttv-pipeline:framepack)..."
    docker build -f Dockerfile.framepack -t ttv-pipeline:framepack .
  fi
  
  if [ "$SETUP_WAN21" = true ]; then
    echo "Building Wan2.1 image (ttv-pipeline:wan21)..."
    docker build -f Dockerfile.wan21 -t ttv-pipeline:wan21 .
  fi
  
  echo "Docker images built successfully"
  docker images | grep ttv
  
  echo ""
  echo "Run container examples:"
  echo "  - Core API: docker run --gpus all -p 7860:7860 -e RUNWAY_API_KEY=your_key ttv-pipeline:core"
  echo "  - FramePack: docker run --gpus all -p 7860:7860 -v /path/to/models:/workspace/ttv-pipeline/models ttv-pipeline:framepack"
  echo "  - Wan2.1: docker run --gpus all -p 7860:7860 -v /path/to/models:/workspace/ttv-pipeline/models ttv-pipeline:wan21"
  echo ""
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Verify pipeline_config.yaml settings"
echo "2. Run the pipeline: python -m pipeline --config pipeline_config.yaml"

if [ "$SETUP_ENV" = true ]; then
  echo "Note: To activate the Python environment: source .venv/bin/activate"
fi
