FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y htop tree nvtop git curl ca-certificates python3.12 python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Astral UV CLI via installer
RUN curl -LsSf https://astral.sh/uv/install.sh -o /uv-installer.sh && \
    sh /uv-installer.sh && \
    rm /uv-installer.sh

# Clone and set up FramePack
WORKDIR /workspace
RUN git clone https://github.com/lllyasviel/FramePack.git
WORKDIR /workspace/FramePack

RUN uv venv --python 3.12 && \
    uv pip install torch torchvision torchaudio && \
    uv pip install -r requirements.txt

EXPOSE 7860

CMD ["bash", "-lc", "source $HOME/.local/bin/env && source .venv/bin/activate && python demo_gradio.py --server 0.0.0.0 --port 7860 --share"]