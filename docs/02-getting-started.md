# Getting Started

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/2-getting-started)*

This document covers the initial setup, installation, and basic configuration required to run the TTV (Text-to-Video) Pipeline.

**Related Files:**
- [`frame_extractor.py`](../frame_extractor.py) - Frame extraction utilities
- [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample) - Configuration template
- [`requirements.txt`](../requirements.txt) - Python dependencies
- [`setup.sh`](../setup.sh) - Automated setup script

## Prerequisites

Before beginning setup, ensure you have:
- Python 3.8 or higher
- CUDA-compatible GPU (for local generation)
- Git
- 50+ GB of disk space (for model downloads)

## Installation Process

The TTV Pipeline provides an automated setup script that handles dependency installation and model downloads.

### Setup Flow

*Source: [`setup.sh`](../setup.sh)*

### Running Setup

Execute the setup script from the repository root:

```bash
chmod +x setup.sh
./setup.sh
```

The script performs these operations:

1. **Creates a Python virtual environment** in `.venv`
2. **Installs flash-attn** with optimized compilation settings
3. **Installs all dependencies** from `requirements.txt`
4. **Downloads the Wan2.1 framework** to `./frameworks/Wan2.1`
5. **Downloads FLF2V model weights** (14B parameters, 720P resolution)
6. **Downloads I2V model weights** for chaining mode

*Source: [`setup.sh`](../setup.sh)*

### Directory Structure After Setup

After successful setup, your directory structure will look like:

```
ttv-pipeline/
├── .venv/                           # Python virtual environment
├── frameworks/Wan2.1/               # Wan2.1 framework code
├── models/Wan2.1-FLF2V-14B-720P/    # FLF2V model weights
├── models/Wan2.1-I2V-14B-720P/      # I2V model weights
├── output/frames/                   # Generated keyframes
├── output/videos/                   # Generated video segments
└── pipeline_config.yaml             # Your configuration file
```

*Source: [`setup.sh`](../setup.sh)*

## Configuration

The pipeline is configured through `pipeline_config.yaml`, which you must create from the provided sample.

### Configuration Structure

*Source: [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample)*

### Initial Configuration

1. **Copy the sample configuration:**
   ```bash
   cp pipeline_config.yaml.sample pipeline_config.yaml
   ```

2. **Set your backend preference** by editing `default_backend`:
   - `wan2.1` - Local GPU generation using downloaded models
   - `veo3` - Google Veo 3 API (requires GCP setup)
   - `runway` - Runway ML API (requires API key)
   - `minimax` - Minimax API (requires API key)
   - `auto` - Automatic backend selection with fallback

3. **Configure API keys** (for remote backends):
   - **OpenAI**: Set `openai_api_key`
   - **Runway ML**: Set `runway_ml.api_key`
   - **Google Veo**: Set `google_veo.project_id` and `credentials_path`
   - **Minimax**: Set `minimax.api_key` or environment variable `MINIMAX_API_KEY`
   - **Stability AI**: Set `stability_api_key`

*Source: [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample)*

### Key Configuration Options

**Generation Mode:**
- `generation_mode`: Choose `"chaining"` or `"keyframe"`

**GPU Settings:**
- `total_gpus`: Number of GPUs available
- `parallel_segments`: Enable parallel segment processing

**Generation Parameters:**
- `segment_duration_seconds`: Duration of each video segment
- `text_to_image_model`: Choose between `"openai/gpt-image-1"` or `"stabilityai/sd3:stable"`

*Source: [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample)*

## Dependencies

### Core Dependencies

The pipeline requires several key Python packages installed via `requirements.txt`:

- **Video Generation**: PyTorch, transformers, accelerate
- **API Integration**: OpenAI, Google Cloud, requests
- **Media Processing**: FFmpeg-python, PIL, opencv-python
- **Configuration**: PyYAML, instructor

### System Dependencies

- **FFmpeg**: Required for video concatenation and processing
- **CUDA Toolkit**: Required for local GPU generation
- **Git LFS**: Required for downloading large model files

## First Run

### Basic Usage Pattern

*Source: [`setup.sh`](../setup.sh)*

Activate the virtual environment and run the pipeline:

```bash
source .venv/bin/activate
python pipeline.py
```

### Command Execution

The pipeline will:

1. **Load your configuration** from `pipeline_config.yaml`
2. **Initialize the selected backend** (local or remote)
3. **Process your prompt** through OpenAI enhancement ([`pipeline_config.yaml.sample:118`](../pipeline_config.yaml.sample))
4. **Generate video segments** using the configured backend
5. **Concatenate segments** using FFmpeg into the final video
6. **Save output** to the `output_dir` specified in configuration ([`pipeline_config.yaml.sample:141`](../pipeline_config.yaml.sample))

*Sources: [`pipeline_config.yaml.sample`](../pipeline_config.yaml.sample) (lines 7-17, 118, 141)*

## Troubleshooting Setup

### Common Issues

1. **CUDA Out of Memory**: Reduce `parallel_segments` or use fewer GPUs
2. **Model Download Failures**: Check internet connection and disk space
3. **API Authentication**: Verify API keys are correctly set in configuration
4. **FFmpeg Not Found**: Install FFmpeg system package

### FramePack Optional Setup

For advanced users, FramePack integration is available:

1. Edit `setup.sh` and uncomment the `setup_framepack` line
2. Re-run the setup script to download FramePack models
3. Configure FramePack-specific settings in your configuration

## Next Steps

- **Pipeline Details**: See [Core Pipeline](03-core-pipeline.md) for orchestration details
- **Backend Configuration**: See [Video Generation Backends](04-video-generation-backends.md) for backend setup
- **Deployment**: See [Deployment and Containers](08-deployment-and-containers.md) for production setup

---

**Quick Start Checklist:**
- [ ] Prerequisites installed
- [ ] Setup script executed successfully
- [ ] Configuration file created and API keys set
- [ ] First test run completed
- [ ] Output directory contains generated video
