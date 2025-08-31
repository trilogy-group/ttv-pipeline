# Image-to-Image Keyframe Generation with Gemini API

## Overview

The TTV Pipeline now supports Image-to-Image (I2I) keyframe generation using Google's Gemini API, enabling character and setting consistency across video segments. This feature is particularly useful for creating cutscenes and narrative-driven content where visual consistency is critical.

## Key Features

- **Character Consistency**: Maintain consistent character appearance across multiple video segments
- **Setting Preservation**: Keep environments and backgrounds consistent throughout the video
- **Gemini API Integration**: Leverage Google's latest Gemini models for high-quality image generation
- **Single-Keyframe Mode**: Compatible with Veo 3's image-to-video generation (no FLF support)
- **Reference Image Support**: Use multiple reference images to guide generation
- **Flexible Configuration**: Easily enable/disable and configure through YAML settings

## Architecture

### Components

1. **Keyframe Generator** (`keyframe_generator.py`)
   - `generate_keyframe_with_gemini()`: Core Gemini API integration
   - Supports input images for I2I transformation
   - Handles reference images for consistency
   - Implements retry logic with exponential backoff

2. **Pipeline Integration** (`pipeline.py`)
   - Extended `generate_keyframes()` function with Gemini support
   - New `generate_video_segments_single_keyframe()` for Veo 3 compatibility
   - Automatic detection of I2I mode from configuration

3. **Configuration** (`pipeline_config.yaml`)
   - New `i2i_mode` section for all I2I settings
   - `gemini_api_key` configuration option
   - Single-keyframe mode settings for Veo 3

## Configuration

### Basic Setup

```yaml
# Enable I2I mode
i2i_mode:
  enabled: true
  backend: "gemini"
  model: "gemini-2.0-flash-exp"
  
  # Reference images for consistency
  reference_images:
    character1: "path/to/character1.png"
    character2: "path/to/character2.png"
    setting: "path/to/setting.png"
  
  # Single keyframe mode for Veo 3
  single_keyframe_mode: true
  keyframe_position: "first"

# Gemini API key
gemini_api_key: "YOUR_GEMINI_API_KEY"

# Use Veo 3 for video generation
default_video_generation_backend: "veo3"
```

### Advanced Options

```yaml
i2i_mode:
  # Custom prompt template
  prompt_template: "Generate an image maintaining the visual consistency of {character_name} and {setting_description}. Scene: {prompt}"
  
  # Keyframe position options: "first", "middle", "last"
  keyframe_position: "first"
  
  # Model selection
  model: "gemini-2.0-flash-exp"  # or "gemini-1.5-pro", "gemini-1.5-flash"
```

## Usage

### Basic Usage

1. **Set up environment variables**:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GCP_PROJECT_ID="your-gcp-project"  # For Veo 3
```

2. **Configure pipeline**:
```yaml
# pipeline_config.yaml
i2i_mode:
  enabled: true
  backend: "gemini"
  reference_images:
    hero: "assets/hero.png"
    castle: "assets/castle.png"
```

3. **Run the pipeline**:
```bash
python pipeline.py --config pipeline_config.yaml
```

### Demo Script

Use the provided demo script to test I2I functionality:

```bash
# Single keyframe generation
python examples/i2i_keyframe_demo.py --mode single

# Character consistency demo
python examples/i2i_keyframe_demo.py --mode consistency

# Full pipeline demo
python examples/i2i_keyframe_demo.py --mode pipeline
```

### Programmatic Usage

```python
from keyframe_generator import generate_keyframe_with_gemini

# Generate a single I2I keyframe
result = generate_keyframe_with_gemini(
    prompt="A knight standing at castle gates",
    output_path="output/keyframe.png",
    gemini_api_key=api_key,
    reference_images={
        'knight': 'refs/knight.png',
        'castle': 'refs/castle.png'
    },
    input_image_path="previous_frame.png"  # Optional
)
```

## Workflow

### 1. Keyframe Generation Flow

```
Story Prompt → Segment Prompts → Gemini I2I Generation → Keyframes
                                        ↑
                                Reference Images
```

### 2. Video Generation Flow

```
Keyframes → Veo 3 I2V → Video Segments → Final Video
```

### 3. Character Consistency Flow

```
Initial Frame → I2I Transform → Frame 2
     ↓              ↓              ↓
Reference ←─────Reference ←────Reference
   Images         Images        Images
```

## Best Practices

### Reference Image Guidelines

1. **Quality**: Use high-resolution reference images (1024x1024 or higher)
2. **Clarity**: Ensure characters/settings are clearly visible
3. **Consistency**: Use similar lighting and angles across references
4. **Format**: PNG or JPEG, properly named for easy identification

### Prompt Engineering

1. **Be Specific**: Include detailed descriptions of actions and settings
2. **Reference Consistency**: Explicitly mention when to maintain character appearance
3. **Scene Transitions**: Plan smooth transitions between segments
4. **Lighting**: Specify lighting conditions for visual continuity

### Performance Optimization

1. **Batch Processing**: Generate multiple keyframes in parallel when possible
2. **Caching**: Reuse reference images across segments
3. **Error Handling**: Implement fallbacks for API failures
4. **Rate Limiting**: Respect Gemini API rate limits

## API Integration Details

### Gemini API Configuration

```python
import google.generativeai as genai

# Configure API
genai.configure(api_key=gemini_api_key)

# Model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
```

### Error Handling

The implementation includes comprehensive error handling:

- **Retry Logic**: Automatic retries with exponential backoff
- **Fallback Support**: Falls back to alternative generators if configured
- **Validation**: Input validation before API calls
- **Logging**: Detailed logging for debugging

## Limitations

1. **Veo 3 Compatibility**: Currently only supports single-keyframe mode (no FLF)
2. **API Costs**: Gemini API usage incurs costs based on token consumption
3. **Processing Time**: I2I generation adds latency to the pipeline
4. **Image Size**: Veo 3 requires 1024x1024 square images

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `GEMINI_API_KEY` is set in environment
   - Check key validity in Google Cloud Console

2. **Reference Images Not Loading**
   - Verify file paths are correct
   - Check image format compatibility (PNG/JPEG)

3. **Veo 3 Generation Fails**
   - Ensure GCP project has Veo 3 access enabled
   - Verify image dimensions are 1024x1024

4. **Character Inconsistency**
   - Provide clearer reference images
   - Use more specific prompts
   - Adjust prompt template

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Support for multiple Gemini models simultaneously
- [ ] Advanced prompt templating system
- [ ] Automatic reference image extraction from videos
- [ ] Support for FLF mode when available in Veo
- [ ] Integration with other I2I backends (Stability, DALL-E)
- [ ] Caching system for generated keyframes
- [ ] Web UI for configuration and monitoring

## Related Documentation

- [Core Pipeline Overview](03-core-pipeline.md)
- [Video Generation Backends](04-video-generation-backends.md)
- [Remote API Generators](07-remote-api-generators.md)
- [Veo 3 Integration](08-veo3-integration.md)

## Support

For issues or questions about I2I keyframe generation:

1. Check this documentation
2. Review the demo script examples
3. Enable debug logging for detailed error messages
4. Consult the Gemini API documentation
5. File an issue in the project repository
