# Deployment and Containers

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/5-deployment-and-containers)*

This document covers the containerized deployment of the TTV Pipeline system, focusing on Docker configuration, FramePack integration, and container orchestration. The deployment strategy enables GPU-accelerated video generation in an isolated, reproducible environment with web-based interfaces.

**Key Source Files:**
- [`Dockerfile`](../Dockerfile) - Container build configuration
- [`demo_gradio.py`](../demo_gradio.py) - Web interface application

## Purpose and Scope

This document covers the containerized deployment of the TTV Pipeline system, focusing on Docker configuration, FramePack integration, and container orchestration. The deployment strategy enables GPU-accelerated video generation in an isolated, reproducible environment with web-based interfaces.

For initial setup and installation procedures, see [Getting Started](02-getting-started.md). For configuration of video generation backends within containers, see [Video Generation Backends](04-video-generation-backends.md).

## Container Architecture Overview

The TTV Pipeline deployment uses a multi-stage containerized architecture built on NVIDIA CUDA containers, with specialized images for different use cases.

### Specialized Dockerfiles

The TTV Pipeline provides separate Dockerfiles for different deployment scenarios:

1. **`Dockerfile.base`**: Base image with common dependencies and TTV Pipeline core
2. **`Dockerfile.core`**: Minimal deployment for API-only backends (no local GPU models)
3. **`Dockerfile.framepack`**: Specialized image with FramePack backend
4. **`Dockerfile.wan21`**: Specialized image with Wan2.1 backend

This approach allows users to build only the specific image they need, without dependencies for unused backends.

**Key Architecture Components:**
- **Base Layer**: NVIDIA CUDA 12.1.1 with Ubuntu 22.04
- **Runtime Environment**: Python 3.10 with UV package management
- **Core Application**: TTV Pipeline with configurable backends
- **Specialized Backends**: Separate images for each backend (FramePack, Wan2.1)
- **Web Interface**: Pipeline-driven Gradio interface
- **GPU Integration**: CUDA 11.8-enabled PyTorch for hardware acceleration

### Container Architecture

The containerized deployment follows a layered architecture with specialized builds for different use cases:

#### Base Image (`ttv-base`)
1. **Infrastructure Layer**: NVIDIA CUDA base image with GPU support
2. **System Layer**: Ubuntu packages and development tools (including ffmpeg for video processing)
3. **Python Environment**: UV-managed virtual environment with CUDA-enabled PyTorch
4. **Core Application**: TTV Pipeline core functionality

#### Core Image (`ttv-core`)
Builds on `ttv-base` and adds:
1. **Configuration**: Minimal pipeline configuration for API-based backends
2. **Interface Layer**: Pipeline-driven Gradio web interface for user interaction

#### FramePack Image (`ttv-framepack`)
Builds on `ttv-base` and adds:
1. **FramePack Backend**: FramePack framework with its dependencies
2. **Enhanced Configuration**: Pipeline configuration with enabled FramePack backend
3. **Interface Layer**: Pipeline-driven Gradio web interface for user interaction

#### Wan2.1 Image (`ttv-wan21`)
Builds on `ttv-base` and adds:
1. **Wan2.1 Backend**: Wan2.1 framework with its dependencies
2. **Enhanced Configuration**: Pipeline configuration with enabled Wan2.1 backend
3. **Interface Layer**: Pipeline-driven Gradio web interface for user interaction

*Source: [`Dockerfile`](../Dockerfile)*

### Deployment Components Mapping

**File System Layout:**
```
/workspace/
└── ttv-pipeline/              # Cloned TTV Pipeline repository
    ├── .venv/                # UV-managed virtual environment
    ├── frameworks/           # External frameworks directory
    │   ├── FramePack/        # FramePack as a dependency (in ttv-framepack image)
    │   └── Wan2.1/           # Wan2.1 as a dependency (in ttv-wan21 image)
    ├── models/               # Model checkpoints directory
    ├── outputs/              # Generated outputs directory
    ├── pipeline_config.yaml  # Pipeline configuration
    └── requirements.txt      # Python dependencies
```

*Source: [`Dockerfile`](../Dockerfile)*

## Docker Configuration

### Base Image and Environment

The container deployment uses NVIDIA CUDA as the foundation to support GPU-accelerated video processing:

**Base Configuration:**
```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"
```

**Key Environment Features:**
- **CUDA Version**: 12.1.1 development environment
- **OS**: Ubuntu 22.04 LTS
- **Python Version**: 3.10 (stable release)
- **Non-interactive**: Automated installation without prompts

The base environment setup includes essential system packages and development tools:
- **`htop`**: System process monitoring
- **`tree`**: Directory structure visualization
- **`nvtop`**: GPU monitoring and management
- **`git`**: Version control for repository cloning
- **`curl`**: HTTP client for downloads
- **`ca-certificates`**: SSL/TLS certificate management
- **`ffmpeg`**: Video processing utilities
- **`libsm6`/`libxext6`**: Required X11 libraries for OpenCV

*Source: [`Dockerfile`](../Dockerfile)*

### Package Management Strategy

The deployment uses Astral UV CLI for efficient Python package management:

**UV Installation:**
```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh -o /uv-installer.sh && \
    sh /uv-installer.sh && \
    rm /uv-installer.sh
```

**Environment Setup:**
```dockerfile
RUN uv venv --python 3.10 && \
    uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install -r requirements.txt && \
    uv pip install gradio
```

**UV Benefits:**
- **Speed**: 10-100x faster than pip for dependency resolution
- **Reliability**: Consistent dependency resolution across environments
- **Isolation**: Clean virtual environment management
- **Reproducibility**: Exact dependency versions for consistent builds

*Sources: [`Dockerfile`](../Dockerfile) (lines 11-13, 20-22)*

## FramePack Integration

### Repository Setup

The container automatically clones and configures the FramePack repository for video generation capabilities:

**Repository Configuration:**
```dockerfile
WORKDIR /workspace
RUN git clone https://github.com/lllyasviel/FramePack.git
WORKDIR /workspace/FramePack
```

**FramePack Features:**
- **Video Generation**: Advanced I2V generation capabilities
- **Web Interface**: Gradio-based user interface
- **Model Support**: Multiple video generation models
- **GPU Acceleration**: CUDA-optimized processing

*Source: [`Dockerfile`](../Dockerfile)*

### Dependencies and Environment

The FramePack environment includes:

**Core Components:**
- **PyTorch Stack**: Core deep learning framework with CUDA support
- **Application Requirements**: Dependencies specified in `requirements.txt`
- **Virtual Environment**: Isolated Python environment using UV

**Dependency Management:**
- **Gradio**: Web interface framework
- **FFmpeg**: Video processing and concatenation
- **NumPy/PIL/OpenCV**: Image and video processing libraries

*Source: [`Dockerfile`](../Dockerfile)*

## TTV Pipeline Configuration

The Dockerfile sets up the pipeline configuration with proper paths for the FramePack integration:

```dockerfile
# Set up pipeline configuration
COPY pipeline_config.yaml.sample pipeline_config.yaml
RUN sed -i 's|# framepack_dir: ./frameworks/FramePack|framepack_dir: ./frameworks/FramePack|' pipeline_config.yaml
```

This configuration enables the TTV Pipeline to properly locate and utilize FramePack for video generation.

## Container Deployment Process

### Build and Runtime Configuration

The container exposes a web interface for video generation through Gradio:

**Network Configuration:**
```dockerfile
EXPOSE 7860

CMD ["bash", "-lc", "source .venv/bin/activate && python -m pipeline --config pipeline_config.yaml"]
```

**Runtime Parameters:**
- **Port**: `7860` - Standard Gradio port
- **Server**: `0.0.0.0` - Bind to all network interfaces
- **Share Mode**: `--share` - Enable external access
- **Application**: `demo_gradio.py` - Web interface entry point

### Command Execution Flow

The startup command chain performs the following sequence:

1. **Environment Activation**: Source UV environment variables
2. **Virtual Environment**: Activate Python virtual environment
3. **Application Launch**: Start Gradio web server
4. **Network Binding**: Listen on all interfaces
5. **Share Enable**: Create public access URL

**Complete Startup Command:**
```bash
bash -lc "source $HOME/.local/bin/env && source .venv/bin/activate && python demo_gradio.py --server 0.0.0.0 --port 7860 --share"
```

*Source: [`Dockerfile`](../Dockerfile) (lines 26)*

## Network and Interface Configuration

### Port Exposure and Access

The TTV Pipeline with Gradio integration provides a simple, interactive way to interact with the video generation system:

**Network Configuration:**
- **Internal Port**: `7860` - Gradio application port
- **Binding**: `0.0.0.0` - Listen on all network interfaces  
- **Share Mode**: Enabled for external access

**Docker Run Example:**
```bash
docker run -p 7860:7860 --gpus all ttv-pipeline:latest
```

**Access Methods:**
- **Local Access**: `http://localhost:7860`
- **Network Access**: `http://<container-ip>:7860`
- **Share URL**: Automatically generated public URL

### Service Architecture

The Gradio server provides both local and shared access to the video generation interface, enabling remote usage and collaboration.

**Service Features:**
- **Web Interface**: Browser-based video generation
- **Real-time Processing**: Live generation monitoring
- **File Upload**: Direct image upload capability
- **Result Download**: Generated video download
- **Parameter Control**: Interactive generation controls

*Source: [`Dockerfile`](../Dockerfile)*

## GPU and Resource Requirements

### CUDA Environment

The deployment requires NVIDIA GPU support with CUDA compatibility:

**GPU Requirements:**
- **Base Image**: NVIDIA CUDA 12.9.0 development environment
- **GPU Access**: Container must be run with `--gpus all` flag
- **CUDA Libraries**: Development libraries included for compilation
- **PyTorch**: CUDA-enabled PyTorch installation

**Docker GPU Command:**
```bash
docker run --gpus all -p 7860:7860 ttv-pipeline:latest
```

**CUDA Features:**
- **Memory Management**: Efficient GPU memory utilization
- **Multi-GPU Support**: Automatic GPU detection and usage
- **CUDA Kernels**: Optimized computation kernels
- **Development Tools**: Full CUDA toolkit for compilation

### System Requirements

**Minimum Requirements:**
- **GPU**: NVIDIA GPU with CUDA 12.0+ support
- **Memory**: 8GB GPU VRAM (16GB+ recommended)
- **Storage**: 20GB disk space for models and cache
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB system memory

**Recommended Specifications:**
- **GPU**: RTX 4080/4090 or A100/H100
- **Memory**: 24GB+ GPU VRAM
- **Storage**: SSD with 100GB+ free space
- **CPU**: 16+ core processor
- **RAM**: 32GB+ system memory

*Source: [`Dockerfile`](../Dockerfile)*

## Deployment Commands

### Building the Container

The multi-stage Dockerfile allows building different images for specific use cases:

```bash
# First build the base image
docker build -f Dockerfile.base -t ttv-base:latest .

# Build the minimal core image (API backends only)
docker build -f Dockerfile.core -t ttv-pipeline:core .

# Build the FramePack image
docker build -f Dockerfile.framepack -t ttv-pipeline:framepack .

# Build the Wan2.1 image
docker build -f Dockerfile.wan21 -t ttv-pipeline:wan21 .
```

### Running the Core Container (API-only)

```bash
# Run the core image (small, no local backends)
docker run --gpus all -p 7860:7860 \
  -e RUNWAY_API_KEY=your_api_key \
  -e MINIMAX_API_KEY=your_api_key \
  ttv-pipeline:core
```

### Running the FramePack Container

```bash
# Run the FramePack image with model volume mounts
docker run --gpus all -p 7860:7860 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /path/to/models:/workspace/ttv-pipeline/models \
  -v /path/to/outputs:/workspace/ttv-pipeline/outputs \
  ttv-pipeline:framepack
```

### Running the Wan2.1 Container

```bash
# Run the Wan2.1 image with model volume mounts
docker run --gpus all -p 7860:7860 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /path/to/models:/workspace/ttv-pipeline/models \
  -v /path/to/outputs:/workspace/ttv-pipeline/outputs \
  ttv-pipeline:wan21
```

### Volume Mounts and Environment Variables

**Essential Volume Mounts:**
- **Models**: `-v /path/to/models:/workspace/ttv-pipeline/models` - Store large model files outside container
- **Outputs**: `-v /path/to/outputs:/workspace/ttv-pipeline/outputs` - Persist generated videos

**Common Environment Variables:**
- **GPU Selection**: `-e CUDA_VISIBLE_DEVICES=0,1` - Control which GPUs are used
- **API Keys**: `-e RUNWAY_API_KEY=xxx -e MINIMAX_API_KEY=xxx` - Remote API credentials

### Development Mode

```bash
# Run with source code mounting for development
docker run --gpus all -p 7860:7860 \
  -v /local/ttv-pipeline:/workspace/ttv \
  -it ttv-pipeline:latest bash
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - **Solution**: Ensure `--gpus all` flag is used
   - **Check**: Verify NVIDIA Docker runtime installation

2. **Port Already in Use**
   - **Solution**: Use different port mapping `-p 8080:7860`
   - **Check**: Identify processes using port 7860

3. **Out of Memory Errors**
   - **Solution**: Reduce batch size or model parameters
   - **Monitor**: Use `nvtop` to monitor GPU memory

4. **Container Build Failures**
   - **Solution**: Check internet connectivity for package downloads
   - **Retry**: Clean build with `--no-cache` flag

### Performance Optimization

**Memory Optimization:**
- Use smaller batch sizes for limited VRAM
- Enable gradient checkpointing for memory efficiency
- Monitor memory usage with container metrics

**Network Optimization:**
- Use host networking for better performance
- Configure proper DNS resolution
- Optimize container image size

---

## Next Steps

- **Local Setup**: See [Getting Started](02-getting-started.md) for non-containerized installation
- **Configuration**: See [Video Generation Backends](04-video-generation-backends.md) for backend setup
- **GPU Setup**: See [Local Generators](06-local-generators.md) for GPU optimization
- **API Integration**: See [Remote API Generators](07-remote-api-generators.md) for cloud alternatives
