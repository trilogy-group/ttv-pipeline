# TTV Pipeline

An end-to-end pipeline for generating long-form videos from text prompts using Stability AI or OpenAI for keyframe generation and Wan2.1 FLF2V model for video segments, with support for GPU parallelization.

## Overview

This project implements a multi-stage pipeline that:
1. Breaks down a long text prompt into segments
2. Enhances prompts using OpenAI/Instructor with colorized output
3. Generates keyframes for each segment using Stability AI or OpenAI gpt-image-1
4. Uses Wan2.1 FLF2V model to generate video segments between keyframes (with parallel processing)
5. Stitches segments together into a final video

## Project Structure

### Core Files
- `pipeline.py` - Main orchestration script that runs the end-to-end video generation pipeline
- `keyframe_generator.py` - Handles generation of keyframes using Stability AI or OpenAI API
- `pipeline_config.yaml.sample` - Sample configuration file (copy to pipeline_config.yaml and add your API keys)
- `setup.sh` - Script to set up the environment and download necessary models
- `requirements.txt` - Python dependencies for the project

### Directory Structure
After running the setup script, your project will have this structure:

```
/
├── frameworks/           # Framework code repositories
│   ├── Wan2.1/           # Wan2.1 framework code
│   └── FramePack/        # FramePack framework (optional)
│
├── models/               # Model weights
│   └── Wan2.1-FLF2V-14B-720P/  # FLF2V model weights
│
├── output/               # Generated outputs
│   ├── frames/           # Generated keyframes
│   └── videos/           # Generated video segments
│
├── pipeline.py           # Main pipeline script
├── keyframe_generator.py # Keyframe generation module
└── pipeline_config.yaml  # Your configuration
```

### Configuration

#### Pipeline Configuration
`pipeline_config.yaml` includes:
- API keys for OpenAI and Stability AI
- Model paths and directories
- GPU settings for distributed and parallel processing
- Input/output path settings
- Model selection options for keyframe generation (OpenAI or Stability AI)

#### GPU Parallelization
The pipeline supports both distributed processing and parallel segment generation with a simplified configuration:

```yaml
# GPU settings
total_gpus: 8        # Total number of GPUs available in the system
parallel_segments: 2  # Number of segments to generate in parallel
```

The system automatically calculates how many GPUs to use per segment: `gpus_per_segment = total_gpus / parallel_segments`

Optimization strategies:
- Use `parallel_segments: 1` to dedicate all GPUs to a single segment for maximum quality and speed
- Use `parallel_segments` equal to the number of segments for maximum throughput (1 GPU per segment)
- Use a balanced approach (e.g., `parallel_segments: 2` with 8 GPUs gives 4 GPUs per segment)
- The system will automatically adjust settings if you request more parallel segments than available GPUs

## Setup and Requirements

### Dependencies
- Python 3.8+
- Stability AI API (for keyframe generation)
- OpenAI API (for prompt enhancement)
- Wan2.1 FLF2V model
- FramePack (optional, for additional video generation capabilities)
- Pydantic & Instructor (for structured output)

### Environment Setup

1. Clone this repository
   ```bash
   git clone https://github.com/trilogy-group/ttv-pipeline.git
   cd ttv-pipeline
   ```

2. Copy the sample configuration and add your API keys
   ```bash
   cp pipeline_config.yaml.sample pipeline_config.yaml
   # Edit pipeline_config.yaml to add your API keys
   ```

3. Run the setup script to download required frameworks and models
   ```bash
   # Make the script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh
   ```
   
   *Note: To also set up FramePack, edit the setup.sh script and uncomment the `setup_framepack` function call.*

## Usage

```bash
python pipeline.py --config pipeline_config.yaml
```

## Pipeline Stages

1. **Prompt Enhancement**
   - Uses OpenAI API with Instructor to structure and expand prompts
   - Colored terminal output for better visualization of prompt segments
   - Outputs enhanced_prompt.json with segment details

2. **Keyframe Generation**
   - Supports multiple image generation models:
     - Stability AI API for consistent character generation (1024x1024)
     - OpenAI's gpt-image-1 model for high-quality keyframes (1536x1024)
   - Supports sequential generation where each keyframe influences the next
   - Outputs keyframes to `output/frames/`

3. **Video Segment Generation**
   - Multiple options for video generation:
     - Wan2.1 FLF2V model (default) to generate video segments between keyframes
     - FramePack integration (optional) for alternative video generation capabilities
   - Parallel processing for increased throughput
   - Distributed GPU usage for faster generation
   - Saves video segments to `output/videos/`

4. **Video Concatenation**
   - Stitches all generated video segments together
   - Produces a final continuous video

## Directory Structure

```
/output
  /frames  - Contains generated keyframe images
  /videos  - Contains generated video segments
  config.yaml - Copy of used configuration
  enhanced_prompt.json - Enhanced prompt data
  final_video.mp4 - Final stitched video (when multiple segments are generated)
```

## Image Generation Options

### Supported Models

The pipeline supports multiple image generation models:

1. **Stability AI** - `stabilityai/sd3:stable`
   - Dimensions: Fixed 1024x1024 square format
   - Image-to-image capability with high consistency
   - Good for maintaining character consistency across keyframes

2. **OpenAI GPT-Image-1** - `openai/gpt-image-1`
   - Dimensions: 1536x1024 (landscape) or 1024x1536 (portrait)
   - Supports both text-to-image and image-to-image generation
   - Advanced masking capabilities:
     - Auto-mask generation for preserving parts of the input image
     - Manual mask support for fine-grained control

### Image Dimension Handling

The pipeline handles dimensions in the following ways:

- **Starting Image**: Can be any dimensions - up to 1280x720 is supported for Wan2.1 FLF2V
- **Generated Keyframes**: 
  - Stability AI: Fixed 1024x1024 square format
  - OpenAI: 1536x1024 (landscape) or 1024x1536 (portrait)

Note: Some visual inconsistency may occur between the initial image and generated keyframes due to dimension differences.

## FramePack Integration

The pipeline includes optional support for FramePack, a powerful frame-to-video model that can be used as an alternative to Wan2.1 FLF2V.

### Setting up FramePack

1. Edit the `setup.sh` script and uncomment the line that calls `setup_framepack`
2. Run the setup script again to download FramePack and its models:
   ```bash
   ./setup.sh
   ```

3. In your `pipeline_config.yaml`, uncomment the FramePack section and ensure the paths are correct:
   ```yaml
   framepack_dir: ./frameworks/FramePack    # Path to the FramePack repository
   ```

### Using FramePack

Once integrated, you can use FramePack for video generation by modifying the pipeline.py script to use FramePack's API instead of the Wan2.1 FLF2V model. This integration is a work in progress and will be enhanced in future updates.

## Performance Notes

### GPU Utilization

- Requires significant GPU resources (configurable via parallelization settings)
- Tested on systems with H200 GPUs
- Supports both distributed processing (multiple GPUs per segment) and parallel processing (multiple segments simultaneously)

### Parallelization Options

Two configuration values control GPU parallelization:

```yaml
total_gpus: 8        # Total available GPUs in the system
parallel_segments: 2  # Number of segments to process in parallel
```

The system automatically distributes available GPUs among parallel segments. This provides flexibility in choosing between speed and throughput:

- **Speed optimized**: Lower `parallel_segments` value allocates more GPUs per segment
- **Throughput optimized**: Higher `parallel_segments` value processes more segments simultaneously

### Model Size Considerations

- The FLF2V-14B model requires substantial GPU memory
- Input images are automatically resized to match required dimensions for each API
- For OpenAI integration, 1536x1024 is the default image size
- For Stability AI API, images are resized to 1024x1024