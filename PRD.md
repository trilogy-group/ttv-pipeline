# Project Requirements Document: Long Video Generation Pipeline

## Project Overview

This project aims to create an end-to-end pipeline for generating high-quality long-form videos by breaking them down into segments, generating each segment separately, and then stitching them together into a seamless final product.

## Current Approach & Challenges

### Architecture Overview

1. **Segment Planning with OpenAI**
   - Use OpenAI's API with Instructor library to create structured JSON outputs
   - Parse a single long prompt into multiple segment prompts and keyframe descriptions
   - Output includes segmentation logic, keyframe prompts, and video segment prompts
   - Enhanced terminal UI with color-coded prompts and segments for better visualization

2. **Keyframe Generation**
   - Generate keyframe images using a text-to-image model
   - Multiple model support:
     - Stability AI (SD3) for 1024x1024 image generation
     - OpenAI gpt-image-1 for 1536x1024 high-quality images
   - Support for image-to-image generation with masking capabilities
   - These keyframes serve as visual anchors between segments

3. **First-Last-Frame-to-Video (FLF2V) Generation**
   - Use specialized Wan2.1 FLF2V model to generate video between keyframes
   - Each segment uses the last frame of previous segment as its first frame
   - Parallel processing capabilities for multi-segment generation
   - Intelligent GPU resource allocation across segments

4. **Video Concatenation**
   - Stitch all generated video segments together using ffmpeg

### Technical Requirements

- **Models Required**:
  - Text-to-Image models for keyframe generation:
    - Stability AI's SD3 model for 1024x1024 images
    - OpenAI's gpt-image-1 model for 1536x1024 high-quality images
  - Wan2.1 FLF2V model for video segment generation 
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
   - Video segments can be generated in parallel processes

3. **Resource Management**:
   - Implemented flexible GPU allocation strategies
   - Configurable parallelization for optimizing throughput vs quality
   - Clear terminal output with color coding for monitoring progress

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

## Future Improvements

1. **Advanced Quality Enhancements**:
   - Implement automatic frame blending at segment boundaries
   - Add post-processing options for smoothing transitions

2. **More Efficient Resource Utilization**:
   - Explore model quantization for reduced memory requirements
   - Investigate streaming generation options to reduce latency

3. **Extended API Support**:
   - Add support for additional image generation providers
   - Implement a plugin system for easier integration of new models

## Current Implementation Status

### Completed Features

1. **Complete Pipeline Implementation**:
   - End-to-end pipeline for long video generation
   - Support for multiple image generation providers
   - Parallel processing capabilities for improved performance

2. **Enhanced User Experience**:
   - Color-coded terminal output for better visibility
   - Clear progress indicators and error messages
   - Detailed logging for debugging and monitoring

3. **Flexible Configuration**:
   - All settings controlled via YAML configuration
   - Support for different GPU parallelization strategies
   - Multiple image generation model options

## Next Steps

1. **Add Support for FramePack**:
   - Find a way to use FramePack programmatically instead of via Gradio
   - Add support for FramePack in the pipeline

2. **Documentation and User Interface**:
   - Create detailed documentation for all features
   - Develop a simple web UI for pipeline configuration
   - Add visualization tools for keyframe and segment planning

3. **Quality Improvements**:
   - Implement frame blending at segment boundaries
   - Add post-processing options for smoother transitions
   - Enhance masking capabilities for better image-to-image results

4. **Performance Optimization**:
   - Explore model quantization for reduced memory usage
   - Implement more sophisticated GPU allocation strategies
   - Investigate streaming generation options

