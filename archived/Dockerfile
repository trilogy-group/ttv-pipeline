# Base image with common dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS ttv-base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y htop tree nvtop git curl ca-certificates \
    python3.10 python3.10-venv python3-pip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Astral UV CLI via installer
RUN curl -LsSf https://astral.sh/uv/install.sh -o /uv-installer.sh && \
    sh /uv-installer.sh && \
    rm /uv-installer.sh

# Set up workspace directory
WORKDIR /workspace

# Clone TTV Pipeline repository
RUN git clone https://github.com/trilogy-group/ttv-pipeline.git
WORKDIR /workspace/ttv-pipeline

# Create frameworks directory for dependencies
RUN mkdir -p frameworks

# Set up Python environment with base requirements
RUN uv venv --python 3.10 && \
    uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install -r requirements.txt

# Install Gradio for web interface
RUN uv pip install gradio

# Expose port for web interface
EXPOSE 7860

# Create directories for models and outputs
RUN mkdir -p models outputs

# Set up minimal pipeline configuration
COPY pipeline_config.yaml.sample pipeline_config.yaml

# Base image with only core pipeline (for API backends)
FROM ttv-base AS ttv-core

CMD ["bash", "-lc", "source .venv/bin/activate && python -m pipeline --config pipeline_config.yaml"]

# Image with FramePack backend
FROM ttv-base AS ttv-framepack

# Clone FramePack into frameworks directory
RUN git clone https://github.com/lllyasviel/FramePack.git frameworks/FramePack

# Install FramePack requirements
RUN source .venv/bin/activate && \
    cd frameworks/FramePack && \
    uv pip install -r requirements.txt

# Enable FramePack in pipeline configuration
RUN sed -i 's|# framepack_dir: ./frameworks/FramePack|framepack_dir: ./frameworks/FramePack|' pipeline_config.yaml

CMD ["bash", "-lc", "source .venv/bin/activate && python -m pipeline --config pipeline_config.yaml"]

# Image with Wan2.1 backend
FROM ttv-base AS ttv-wan21

# Clone Wan2.1 into frameworks directory
RUN git clone https://github.com/Wan-Video/Wan2.1.git frameworks/Wan2.1

# Install Wan2.1 requirements
RUN source .venv/bin/activate && \
    cd frameworks/Wan2.1 && \
    uv pip install -r requirements.txt

# Enable Wan2.1 in pipeline configuration
RUN sed -i 's|# wan2_dir: ./frameworks/Wan2.1|wan2_dir: ./frameworks/Wan2.1|' pipeline_config.yaml

CMD ["bash", "-lc", "source .venv/bin/activate && python -m pipeline --config pipeline_config.yaml"]