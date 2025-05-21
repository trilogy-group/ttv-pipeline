# Project Requirements Document: Long Video Generation Pipeline

## Product Vision

Create a scalable, automated pipeline for generating high-quality long-form videos by breaking down complex prompts into optimally-sized segments, managing their generation with sophisticated error handling, and efficiently utilizing available compute resources. This document outlines the technical architecture, implementation decisions, and development roadmap.

## Technical Background

### Video Generation Challenges

Video generation with text-to-video models faces several significant challenges:

1. **Complex Inference Requirements**: Video generation needs to maintain both spatial and temporal consistency, realistic physics, and faces the usual challenges of text-to-image models (like difficulty with text rendering).

2. **Limited Training Data**: Despite being a harder inference challenge, video generation has access to much less training data than text-to-image models, as there are far fewer text-video paired samples.

3. **Quality Degradation in Longer Videos**: Longer videos suffer from "drifting" or error accumulation, causing quality to degrade as the video progresses.

4. **Character Consistency**: Maintaining consistent character appearance throughout a video is particularly challenging, as characters need to preserve their visual attributes while changing posture and movement.

5. **Resource Constraints**: Most text-to-video models can only generate short clips (5-10 seconds) due to computational limitations, processing time, and consistency challenges.

## Architecture and Implementation

### System Architecture

1. **Segment Planning with OpenAI**
   - Use OpenAI's API with Instructor library to create structured JSON outputs
   - Parse a single long prompt into multiple segment prompts and keyframe descriptions
   - Output includes segmentation logic, keyframe prompts, and video segment prompts
   - Enhanced terminal UI with color-coded prompts and segments for better visualization

2. **Keyframe Generation** (Used in Keyframe Mode)
   - Generate keyframe images using a text-to-image model
   - Multiple model support:
     - Stability AI (SD3) for 1024x1024 image generation
     - OpenAI gpt-image-1 for 1536x1024 high-quality images
   - Support for image-to-image generation with masking capabilities to maintain character and setting consistency
   - Robust error handling with automatic retry mechanism for API failures
   - Content moderation handling with prompt rewording capability
   - Colored terminal output for keyframe prompts for better tracking
   - These keyframes serve as visual anchors between segments

3. **Video Generation Modes**
   
   **Keyframe Mode: First-Last-Frame-to-Video (FLF2V) Generation**
   - Use specialized Wan2.1 FLF2V model to generate video between keyframes
   - Each segment interpolates between keyframes, using the previous segment's ending keyframe as its first frame
   - Automatic generation of initial frame (segment_00.png) when no starting image is provided
   - Multiple fallback mechanisms to ensure segment_00.png always exists
   - Parallel processing capabilities for multi-segment generation
   - Intelligent GPU resource allocation across segments
   
   **Chaining Mode: Image-to-Video Generation** (Planned)
   - Use image-to-video models that take a reference image and prompt to create video
   - Automatically extract the last frame of each generated segment to use as reference for the next segment
   - Maintains visual continuity while allowing for narrative progression

4. **Video Concatenation**
   - Stitch all generated video segments together using ffmpeg
   - Create a seamless final video from individually generated segments

### Technical Requirements

- **Models Required**:
  - Text-to-Image models for keyframe generation:
    - Stability AI's SD3 model for 1024x1024 images
    - OpenAI's gpt-image-1 model for 1536x1024 high-quality images (also used in image-to-image mode for character consistency)
  - Wan2.1 FLF2V model for video segment generation in keyframe mode
  - Future support for image-to-video models in chaining mode
  - OpenAI API for prompt enhancement and segmentation

- **Compute Requirements**:
  - The FLF2V-14B model requires multiple GPUs (optimally 8 H200s)
  - Support for various parallelization strategies:
    - Distributed processing (multiple GPUs per segment)
    - Parallel processing (multiple segments simultaneously)
  - Configurable GPU allocation based on throughput vs quality priorities

## Current Solutions Implemented

1. **Wrapper Architecture**:
   - Successfully implemented a wrapper script around the original Wan2.1 generate.py
   - Using command-line arguments for proper integration with the original code
   - Complete pipeline orchestration handled by our wrapper scripts

2. **Component Isolation**:
   - Prompt enhancement logic is completely separated from Wan2.1 code
   - Keyframe generation supports multiple models (Stability AI and OpenAI)
   - Smart API selection based on configured text-to-image model
   - Video segments can be generated in parallel processes

3. **Resource Management**:
   - Implemented flexible GPU allocation strategies
   - Configurable parallelization for optimizing throughput vs quality
   - Clear terminal output with color coding for monitoring progress
   - Improved logging with reduced redundancy and better error reporting

## Remaining Challenges

1. **Compute Resource Requirements**:
   - The FLF2V-14B model still requires significant GPU resources
   - Generation times can be long for complex scenes

2. **Image Dimension Handling**:
   - Different models produce different image dimensions, which can affect consistency
   - Need for automatic resizing/cropping to maintain aspect ratios

3. **Quality Consistency**:
   - Ensuring visual consistency across segment boundaries
   - Balancing between parallelization and visual quality

## Implementation Strategy Rationale

### Why Segmentation Works

Segmentation addresses fundamental video generation challenges by:

1. **Limiting Error Accumulation**: By constraining each segment to a short duration, we prevent the quality degradation that occurs with longer videos

2. **Enabling Narrative Control**: With separate prompts for each segment, we can precisely control the narrative flow and scene transitions

3. **Optimizing Resource Usage**: Short segments can be processed in parallel, making efficient use of available GPU resources

4. **Improving Character Consistency**: Using image-to-image techniques between segments helps maintain consistent character appearance

### Keyframe Mode vs. Chaining Mode

- **Keyframe Mode**: Offers greater creative control by explicitly defining the start and end points of each segment. Particularly useful for complex narratives with specific visual milestones.

- **Chaining Mode**: Provides a more streamlined workflow when the exact visual endpoints aren't critical. More efficient for simpler narratives or when rapid generation is prioritized.

## Future Improvements

1. **Advanced Quality Enhancements**:
   - Implement automatic frame blending at segment boundaries
   - Add post-processing options for smoothing transitions
   - Explore anti-drifting techniques similar to FramePack's approach

2. **More Efficient Resource Utilization**:
   - Explore model quantization for reduced memory requirements
   - Investigate streaming generation options to reduce latency
   - Implement adaptive resource allocation based on segment complexity

3. **Extended Model Support**:
   - Add support for additional image and video generation providers
   - Implement a plugin system for easier integration of new models

## Current Implementation Status

### Completed Features

1. **Complete Pipeline Implementation (Keyframe Mode)**:
   - End-to-end pipeline for long video generation using first-last-frame approach
   - Support for multiple image generation providers for keyframes
   - Parallel processing capabilities for improved performance
   - Robust error handling with automatic retry mechanism and prompt rewording for API safety requirements
   - Automatic initial frame generation when no starting image is provided

2. **Enhanced User Experience**:
   - Color-coded terminal output for better visibility
   - Clear progress indicators and error messages
   - Detailed logging for debugging and monitoring

3. **Flexible Configuration**:
   - All settings controlled via YAML configuration
   - Support for different GPU parallelization strategies
   - Multiple image generation model options

## Next Steps

1. **Add Chaining Mode Functionality**:
   - Create separate prompts, one for each segment:
     - A video clip prompt to generate that segment's video with an image-to-video model
     - Use the previous clip's last frame (automatically extracted) as the reference image for the next segment
     - Use the provided initial image as the reference for the first segment
   - Implement automatic frame extraction using ffmpeg
   - Only sequential segment generation is supported in this mode, but can still do distributed processing for each segment

2. **Add FramePack integration**:
   - Find a way to use FramePack programmatically instead of via Gradio
   - Add FramePack support for Wan2.1 in addition to the existing Hunyuan model

3. **Further Robustness Enhancements**:
   - Improve keyframe prompt output display format for better user experience
   - Add more sophisticated fault tolerance for distributed processing
   - Implement additional fallback mechanisms for API failures
   - Add comprehensive logging and monitoring for long-running generations

4. **Documentation and User Interface**:
   - Create detailed documentation for all features
   - Develop a simple web UI for pipeline configuration
   - Add visualization tools for keyframe and segment planning

5. **Quality Improvements**:
   - Improve prompt engineering
   - Test and enhance masking capabilities for better image-to-image results

6. **Performance Optimization**:
   - Explore model quantization for reduced memory usage
   - Implement more sophisticated GPU allocation strategies

