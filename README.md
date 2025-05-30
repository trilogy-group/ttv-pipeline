# TTV Pipeline

An end-to-end pipeline for generating high-quality long-form videos from text prompts.

## Overview

Generating high-quality long-form videos from text is challenging due to limitations in current AI video models:

- Most models can only generate short 5-10 second clips
- Quality degrades in longer videos due to "drifting"
- Character consistency is difficult to maintain

This pipeline solves these problems by:

1. Breaking your long prompt into optimal segments
2. Generating videos using either **Keyframe Mode** (with intermediate keyframes) or **Chaining Mode** (frame-to-frame continuity)
3. Scaling video generation through flexible parallelization:
   - **GPU Distribution**: Distribute individual segment processing across multiple GPUs (available in all modes)
   - **Parallel Segments**: Process multiple segments simultaneously (keyframe mode only, since chaining mode requires sequential processing)
   - **Cloud Scaling**: Leverage remote APIs (Runway ML, Veo3, Minimax) for on-demand processing without local GPU requirements
4. Combining everything into a seamless final video

### New: Remote API Support Now Available! 

The pipeline now supports remote video generation APIs, making high-quality video generation accessible without expensive GPU hardware:

- **Runway ML**: Successfully integrated Gen-4 Turbo models via API with full support for image-to-video generation
- **Google Veo 3**: State-of-the-art video generation via Google Cloud (initial implementation ready for final testing once allowlisted)
- **Minimax**: Cost-effective I2V-01-Director model with advanced camera movement controls and 6-second generation capability
- **Seamless Integration**: Switch between local (Wan2.1) and cloud generation, or set a preferred default and fallback, all through `pipeline_config.yaml`
- **Cost Effective**: Pay-per-use model for cloud APIs can be more economical than GPU rental for some use cases
- **Environment Variable Security**: API keys are securely managed through environment variables
- **Comprehensive Generator Architecture**: Extensible factory pattern with automatic fallback support

See the "Configuration" section for details on setting up different backends.

## System Architecture

### 1. Segment Planning with OpenAI
   - Use OpenAI's API with Instructor library to create structured JSON outputs
   - Parse a single long prompt into multiple segment prompts and keyframe descriptions
   - Output includes segmentation logic, keyframe prompts, and video segment prompts
   - Enhanced terminal UI with color-coded prompts and segments for better visualization

### 2. Keyframe Generation (Used in Keyframe Mode)
   - Generate keyframe images using a text-to-image model
   - Multiple model support:
     - Stability AI (SD3) for 1024x1024 image generation
     - OpenAI gpt-image-1 for 1536x1024 high-quality images
   - Support for image-to-image generation with masking capabilities to maintain character and setting consistency
   - Robust error handling with automatic retry mechanism for API failures
   - Content moderation handling with prompt rewording capability
   - Colored terminal output for keyframe prompts for better tracking
   - These keyframes serve as visual anchors between segments

### 3. Video Generation Modes
   
   **Keyframe Mode: First-Last-Frame-to-Video (FLF2V) Generation**
   - Use specialized Wan2.1 FLF2V model to generate video between keyframes
   - Each segment interpolates between keyframes, using the previous segment's ending keyframe as its first frame
   - Automatic generation of initial frame (segment_00.png) when no starting image is provided
   - Multiple fallback mechanisms to ensure segment_00.png always exists
   - Parallel processing capabilities for multi-segment generation
   - Intelligent GPU resource allocation across segments
   
   **Chaining Mode: Image-to-Video Generation**
   - Use image-to-video models that take a reference image and prompt to create video
   - Automatically extract the last frame of each generated segment to use as reference for the next segment
   - Maintains visual continuity while allowing for narrative progression
   - Used by remote APIs (Runway ML, Veo3, Minimax) and local Wan2.1 I2V models
   - Sequential processing required to maintain frame continuity

### 4. Video Concatenation
   - Stitch all generated video segments together using ffmpeg
   - Create a seamless final video from individually generated segments

## Project Structure

### Core Files
- `pipeline.py` - Main orchestration script that runs the end-to-end video generation pipeline
- `keyframe_generator.py` - Handles generation of keyframes using Stability AI or OpenAI API
- `pipeline_config.yaml.sample` - Comprehensive sample configuration file. Copy to `pipeline_config.yaml` and customize for your API keys, preferred backends (local Wan2.1, Runway, Google Veo, Minimax), generation parameters, and other settings.
- `setup.sh` - Script to set up the environment and download necessary models for local generation.
- `requirements.txt` - Python dependencies for the project.
- `generators/` - Directory containing the video generator abstraction layer, including interfaces and specific backend implementations (local and remote).

### Directory Structure
After running the setup script (for local Wan2.1 usage), your project will have this structure:

```
/
├── frameworks/           # Framework code repositories (for local Wan2.1)
│   ├── Wan2.1/           # Wan2.1 framework code
│   └── FramePack/        # FramePack framework (optional, local)
│
├── models/               # Model weights (for local Wan2.1)
│   └── Wan2.1-FLF2V-14B-720P/  # FLF2V model weights
│   └── Wan2.1-I2V-14B-720P/    # I2V model weights
│
├── generators/           # Video Generator Abstraction Layer
│   ├── local/            # Local generator implementations (e.g., Wan2.1)
│   ├── remote/           # Remote API generator implementations (e.g., Runway, Veo3, Minimax)
│   ├── factory.py        # Factory for creating generator instances
│   └── video_generator_interface.py # Base interface for video generators
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

All pipeline behavior is controlled through `pipeline_config.yaml`. Copy `pipeline_config.yaml.sample` to `pipeline_config.yaml` and edit it to suit your needs.

#### Key Configuration Sections in `pipeline_config.yaml`:

1.  **Base Configuration**:
    *   `task`, `size`, `prompt`: Core details for your video.

2.  **Backend Selection**:
    *   `default_backend`: Specify your preferred video generation backend (e.g., "wan2.1", "hunyuan", "runway", "veo3", "minimax"). The system will attempt to use this backend first.

3.  **Local Backend Configuration (Wan2.1)**:
    *   `wan2_dir`, `flf2v_model_dir`, `i2v_model_dir`: Paths for the local Wan2.1 setup.
    *   `total_gpus`, `parallel_segments`, `gpu_count`: GPU settings for local generation. `parallel_segments` is only supported in keyframe mode (chaining mode requires sequential processing). See "GPU Parallelization (for Local Wan2.1 Backend)" below.
    *   `chaining_max_retries`, `chaining_use_fsdp_flags`: Specific to Wan2.1 chaining mode.
    *   `hunyuan_video`: Paths for the HunyuanVideo generator (`hunyuan_dir`, `config_file`, `ckpt_path`).

4.  **Remote Backend Configuration**:
    *   `runway_ml`: Settings for Runway ML, including `api_key`, `model_version`, etc.
    *   `google_veo`: Settings for Google Veo, including `project_id`, `credentials_path`, etc.
    *   `minimax`: Settings for Minimax API, including `api_key`, `model_version`, etc.

5.  **Remote API Settings (Common to all remote backends)**:
    *   `max_retries`, `timeout`: General settings for remote API calls.
    *   `fallback_backend`: (Optional) Specify a backend (e.g., "wan2.1", "runway", "minimax") to use if the `default_backend` (if it's a remote API) fails. If not set, or if the specified fallback also fails, the system may try other available registered backends.

6.  **Cost Optimization**:
    *   `max_cost_per_video`, `prefer_local_when_available`.
    *   `api_priority_order`: (Currently a placeholder for future advanced fallback strategies) Defines a preferred order if multiple remote APIs are configured. The current fallback logic is simpler (tries `fallback_backend` then iterates others).

7.  **Image Generation Configuration**:
    *   `text_to_image_model`, `image_size`, API keys for image services (`image_router_api_key`, `stability_api_key`, `openai_api_key`).

8.  **Generation Parameters (Applies to all video backends)**:
    *   `segment_duration_seconds`: Desired duration for each video segment in seconds (e.g., 5.0). Crucial for chaining mode.
    *   `frame_num`, `sample_steps`, `guide_scale`, `base_seed`, etc.

9.  **Output and Logging Configuration**.

#### GPU Parallelization (for Local Wan2.1 Backend)
When using the local Wan2.1 backend, the pipeline supports distributed processing and parallel segment generation:

```yaml
# GPU settings for local generation (in pipeline_config.yaml)
total_gpus: 1        # Total number of GPUs available in the system
parallel_segments: 1 # Number of segments to generate in parallel
# gpu_count: 1       # Legacy, can often be derived or ignored if using total_gpus and parallel_segments
```

The system automatically calculates how many GPUs to use per segment: `gpus_per_segment = total_gpus / parallel_segments`.

Optimization strategies for local Wan2.1:
- Use `parallel_segments: 1` to dedicate all GPUs to a single segment for maximum quality and speed per segment.
- Use `parallel_segments` equal to the number of segments for maximum throughput (if `total_gpus` allows for at least 1 GPU per segment).
- Use a balanced approach (e.g., `parallel_segments: 2` with 8 `total_gpus` gives 4 GPUs per segment).

Remote backends (Runway, Veo3, Minimax) manage their own scaling and do not use these local GPU settings.

## Setup and Requirements

### Dependencies
- Python 3.10+ (Not yet tested on Python 3.12 & Cuda 12.9 for Wan2.1 - coming soon!)
- For Keyframe Generation: Stability AI API or OpenAI gpt-image-1 API key.
- For Prompt Enhancement: OpenAI API key.
- For Local Video Generation (Wan2.1): Wan2.1 I2V model (for chaining mode) or FLF2V model (for keyframe mode). Download via `setup.sh`.
- For Remote Video Generation: API keys for Runway ML, Google Cloud Project with Veo 3 API enabled, and Minimax API.
- Pydantic & Instructor (for structured output).

### Environment Setup

1. Clone this repository
   ```bash
   git clone https://github.com/trilogy-group/ttv-pipeline.git
   cd ttv-pipeline
   ```

2. Copy the sample configuration and add your API keys and preferences:
   ```bash
   cp pipeline_config.yaml.sample pipeline_config.yaml
   # Edit pipeline_config.yaml to add your API keys for OpenAI, Stability, Runway, Google Cloud, Minimax, etc.
   # Configure your default_backend, model paths (if using local), and other parameters.
   ```

3. If using local Wan2.1 generation, run the setup script to download required frameworks and models:
   ```bash
   # Make the script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh
   ```
   *Note: To also set up FramePack (optional local generator), edit the `setup.sh` script and uncomment the `setup_framepack` function call before running.* 
   If you are only using remote APIs for video generation, you might not need to run the full `setup.sh` script, but ensure Python dependencies from `requirements.txt` are installed.

## Usage

```bash
python pipeline.py --config pipeline_config.yaml
```

## How It Works

### Pipeline Stages

1. **Prompt Enhancement**: Transforms your single prompt into optimized segment prompts using AI.

2. **Keyframe Generation**: Creates key visual anchors for your video using either:
   - Stability AI API for consistent character generation (1024x1024).
   - OpenAI's gpt-image-1 model for high-quality images (1536x1024).

3. **Video Generation**: The pipeline supports two generation modes:
   - **Chaining Mode**: Uses image-to-video models that sequentially process segments, with each segment's final frame becoming the starting frame for the next segment. Available with local Wan2.1 (I2V model) or remote APIs (Runway, Veo3, Minimax).
   - **Keyframe Mode**: Uses the specialized Wan2.1 FLF2V model to generate video between pre-defined keyframes, enabling parallel processing for faster generation.
   - Includes fallback mechanisms: if the `default_backend` fails, it can switch to a `fallback_backend` or other available generators as configured.

4. **Video Concatenation**: Stitches all generated video segments together into your final video.

### Key Features

- **Flexible Backend Support**: Choose between local Wan2.1, Runway ML API, Google Veo 3 API, or Minimax API for video generation.
- **Fallback System**: Configure fallback backends in case your primary choice fails.
- **Robust Error Handling**: Automatic retries with prompt rewording for API failures.
- **Resource Optimization**: Intelligent GPU allocation for parallel processing (local Wan2.1).
- **User-Friendly Output**: Color-coded terminal outputs for tracking progress.
- **Comprehensive Configuration**: Customize all aspects via a single YAML file (`pipeline_config.yaml`).

## Directory Structure (Output)

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

The pipeline supports multiple image generation models for keyframes:

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

- **Starting Image**: Can be any dimensions - up to 1280x720 is supported for Wan2.1 FLF2V (local keyframe mode).
- **Generated Keyframes**: 
  - Stability AI: Fixed 1024x1024 square format.
  - OpenAI: 1536x1024 (landscape) or 1024x1536 (portrait).


Note: Some visual inconsistency may occur between the initial image and generated keyframes due to dimension differences.

## HunyuanVideo Integration

You can also use Tencent's [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) model for local generation. Clone the repository and set the `hunyuan_video` paths in `pipeline_config.yaml`.

## FramePack Integration (Optional Local Generator)

The pipeline includes optional support for FramePack, a powerful frame-to-video model that can be used as an alternative local generator to Wan2.1 FLF2V.

### Setting up FramePack

1. Edit the `setup.sh` script and uncomment the line that calls `setup_framepack`.
2. Run the setup script again to download FramePack and its models:
   ```bash
   ./setup.sh
   ```

### Using FramePack

To use FramePack for video generation (typically in keyframe mode):

1. Ensure FramePack is set up correctly (models downloaded, environment configured).
2. In your `pipeline_config.yaml`, you would typically set `default_backend: "wan2.1"` and `generation_mode: "keyframe"`, then ensure your Wan2.1 configuration points to FramePack models if it's adapted to use them, or select FramePack via a more specific (future) configuration if the abstraction layer supports it directly.
   *(Developer Note: The direct selection of 'framepack' as a distinct backend in `default_backend` would require registering it in `generators/factory.py` and creating a corresponding `FramePackGenerator` class. The current README implies it's used via Wan2.1's FLF2V mode, which might need clarification or code changes for direct FramePack backend selection.)*

*Note: FramePack integration details may evolve.*

## Future Enhancements

- Advanced prompt engineering techniques
- Improved character consistency across segments, especially with remote APIs
- UI for easier interaction with the pipeline
- More sophisticated fallback strategies and cost controls
- Support for additional video generation models and APIs
- FramePack direct backend integration (currently used via Wan2.1)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
