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

The TTV Pipeline deployment uses a containerized architecture built on NVIDIA CUDA containers with FramePack integration for web-based video generation interfaces.

**Key Architecture Components:**
- **Base Layer**: NVIDIA CUDA 12.9.0 with Ubuntu 24.04
- **Runtime Environment**: Python 3.12 with UV package management
- **Core Application**: FramePack video generation framework
- **Web Interface**: Gradio-based web application
- **GPU Integration**: CUDA-enabled PyTorch for hardware acceleration

### Container Architecture

The containerized deployment follows a layered architecture:

1. **Infrastructure Layer**: NVIDIA CUDA base image with GPU support
2. **System Layer**: Ubuntu packages and development tools
3. **Python Environment**: UV-managed virtual environment with dependencies
4. **Application Layer**: FramePack framework and TTV Pipeline integration
5. **Interface Layer**: Gradio web interface for user interaction

*Source: [`Dockerfile`](../Dockerfile) (lines 1-26)*

### Deployment Components Mapping

**File System Layout:**
```
/workspace/
├── FramePack/           # Cloned FramePack repository
│   ├── .venv/          # UV-managed virtual environment
│   ├── requirements.txt # Python dependencies
│   └── demo_gradio.py  # Web interface application
└── ttv-pipeline/       # TTV Pipeline integration
```

*Source: [`Dockerfile`](../Dockerfile) (lines 1-26)*

## Docker Configuration

### Base Image and Environment

The container deployment uses NVIDIA CUDA as the foundation to support GPU-accelerated video processing:

**Base Configuration:**
```dockerfile
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"
```

**Key Environment Features:**
- **CUDA Version**: 12.9.0 development environment
- **OS**: Ubuntu 24.04 LTS
- **Python Version**: 3.12 (latest stable)
- **Non-interactive**: Automated installation without prompts

The base environment setup includes essential system packages and development tools:
- **`htop`**: System process monitoring
- **`tree`**: Directory structure visualization
- **`nvtop`**: GPU monitoring and management
- **`git`**: Version control for repository cloning
- **`curl`**: HTTP client for downloads
- **`ca-certificates`**: SSL/TLS certificate management

*Source: [`Dockerfile`](../Dockerfile) (lines 6-8)*

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
RUN uv venv --python 3.12 && \
    uv pip install torch torchvision torchaudio && \
    uv pip install -r requirements.txt
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

*Source: [`Dockerfile`](../Dockerfile) (lines 17-18)*

### Dependencies and Environment

The FramePack environment includes:

**Core Components:**
- **PyTorch Stack**: Core deep learning framework with CUDA support
- **Application Requirements**: Dependencies specified in `requirements.txt`
- **Virtual Environment**: Isolated Python environment using UV

**Dependency Management:**
```dockerfile
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
```

**Key Dependencies:**
- **PyTorch**: GPU-accelerated tensor computing
- **TorchVision**: Computer vision utilities
- **TorchAudio**: Audio processing capabilities
- **Gradio**: Web interface framework
- **NumPy/PIL**: Image processing libraries

*Source: [`Dockerfile`](../Dockerfile) (lines 15-22)*

## Container Deployment Process

### Build and Runtime Configuration

The container exposes a web interface for video generation through Gradio:

**Network Configuration:**
```dockerfile
EXPOSE 7860

CMD ["bash", "-lc", "source $HOME/.local/bin/env && source .venv/bin/activate && python demo_gradio.py --server 0.0.0.0 --port 7860 --share"]
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

The container configuration enables web-based access to the video generation system:

**Network Configuration:**
- **Internal Port**: `7860` - Gradio application port
- **Binding**: `0.0.0.0` - Listen on all network interfaces  
- **Share Mode**: Enabled for external access
- **Protocol**: HTTP web interface

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

*Source: [`Dockerfile`](../Dockerfile) (lines 24-26)*

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

*Source: [`Dockerfile`](../Dockerfile) (lines 1-26)*

## Deployment Commands

### Building the Container

```bash
# Build the Docker image
docker build -t ttv-pipeline:latest .

# Build with custom tag
docker build -t ttv-pipeline:v1.0 .
```

### Running the Container

```bash
# Basic run with GPU support
docker run --gpus all -p 7860:7860 ttv-pipeline:latest

# Run with volume mounting for persistent storage
docker run --gpus all -p 7860:7860 \
  -v /local/models:/workspace/models \
  ttv-pipeline:latest

# Run with environment variables
docker run --gpus all -p 7860:7860 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  ttv-pipeline:latest
```

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
