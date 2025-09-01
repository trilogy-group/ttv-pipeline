# Image-to-Image Keyframe Generation Specification

## Executive Summary

This specification defines a new image-to-image (I2I) keyframe generation system for the TTV Pipeline that leverages advanced image editing APIs to create character-consistent cutscenes for long-format videos. The system will use Google's Gemini-2.5-Flash-Image-Preview model (NanoBanana) for image editing and integrate with Veo 3 for video generation.

## Problem Statement

Current keyframe mode in the TTV Pipeline relies on text-to-image generation, which lacks character and setting consistency across segments. This limitation prevents the creation of coherent narrative videos where the same characters appear throughout different scenes.

## Solution Overview

Extend the keyframe generation system to support image-to-image workflows where:
1. Initial reference images establish characters and settings
2. Image editing APIs (Gemini-2.5-Flash-Image-Preview) modify these references for different scenes
3. Generated keyframes maintain visual consistency while depicting different actions
4. Veo 3 uses these keyframes for image-to-video generation (not first-last-frame mode)

## Key Components

### 1. Gemini API Integration

**Requirements:**
- Add support for `gemini-2.5-flash-image-preview` model in `keyframe_generator.py`
- Implement Gemini API client using Google's generative AI SDK
- Support both text-to-image and image-to-image workflows
- Handle API authentication via `GEMINI_API_KEY` environment variable
- Consistent with existing API key patterns in the codebase

**Implementation Details:**
```python
def generate_keyframe_with_gemini(
    prompt: str,
    output_path: str,
    gemini_api_key: str,
    input_image_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image-preview"
) -> str:
    """Generate keyframe using Gemini API"""
    # Implementation details
```

### 2. Enhanced Keyframe Mode

**Current State:**
- Keyframe mode uses text-to-image for generating segment boundaries
- Sequential generation with previous frame as reference (when available)
- Supports OpenAI, Stability AI, and ImageRouter APIs

**Proposed Changes:**
- Add explicit I2I mode flag in pipeline configuration
- Support reference image library for character/setting consistency
- Implement prompt templates for consistent character references
- Enable keyframe-only mode for Veo 3 (disable first-last-frame requirements)

**Configuration Schema:**
```yaml
keyframe_settings:
  mode: "image_to_image"  # Options: "text_to_image", "image_to_image"
  image_model: "gemini-2.5-flash-image-preview"
  
  # Reference images for I2I mode
  reference_images:
    characters:
      - path: "assets/character1.png"
        name: "main_character"
        description: "Young woman with red hair, business attire"
    settings:
      - path: "assets/conference_room.png"
        name: "meeting_room"
        description: "Modern conference room with glass walls"
  
  # Prompt templates for consistency
  prompt_templates:
    character_action: "Image of {character_name}: {action_description}"
    scene_transition: "Same setting as reference, but {change_description}"
```

### 3. Veo 3 Integration Updates

**Current Limitations:**
- Veo 3 generator exists but only supports image-to-video
- No support for first-last-frame generation
- Keyframe mode currently expects first-last-frame capability

**Required Changes:**
- Modify `generate_video_keyframe_mode()` in `pipeline.py` to support I2V-only backends
- Add backend capability flags for keyframe compatibility
- Implement fallback logic for backends without first-last-frame support
- Update video segment generation to use single keyframes when needed

**Backend Capability Matrix:**
```python
BACKEND_CAPABILITIES = {
    "veo3": {
        "supports_i2v": True,
        "supports_flf": False,  # First-last-frame
        "keyframe_mode": "single",  # Use single keyframe per segment
        "max_duration": 5.0
    },
    "wan2.1": {
        "supports_i2v": True,
        "supports_flf": True,
        "keyframe_mode": "paired",  # Use keyframe pairs
        "max_duration": 10.0
    }
}
```

### 4. Workflow Pipeline

**Image-to-Image Keyframe Generation Flow:**

1. **Initialization Phase**
   - Load reference images from configuration
   - Initialize Gemini API client
   - Validate all reference images exist and are valid

2. **Prompt Enhancement Phase**
   - Original prompt describes the full narrative
   - System generates segment-specific prompts
   - Each prompt references characters/settings from library

3. **Keyframe Generation Phase**
   - For each segment:
     - Select appropriate reference image(s)
     - Apply Gemini I2I transformation with segment prompt
     - Save generated keyframe to frames directory
     - Use generated frame as reference for next segment (optional)

4. **Video Generation Phase**
   - For Veo 3 (single keyframe mode):
     - Use each keyframe as starting image
     - Generate video segment with motion prompt
   - For backends with FLF support:
     - Use traditional keyframe pairs

**Example Workflow:**
```python
# Phase 1: Setup
reference_images = load_reference_images(config)
gemini_client = initialize_gemini(api_key)

# Phase 2: Generate keyframes
keyframes = []
for segment in segments:
    # Select reference based on segment needs
    ref_image = select_reference(segment, reference_images)
    
    # Generate I2I keyframe
    keyframe = gemini_client.edit_image(
        image=ref_image,
        prompt=segment.prompt,
        model="gemini-2.5-flash-image-preview"
    )
    keyframes.append(keyframe)

# Phase 3: Generate videos
for i, keyframe in enumerate(keyframes):
    if backend == "veo3":
        # Single keyframe I2V
        video = veo3.generate_video(
            image=keyframe,
            prompt=segments[i].motion_prompt,
            duration=segment_duration
        )
    else:
        # Traditional FLF approach
        video = generate_flf_video(
            first_frame=keyframes[i],
            last_frame=keyframes[i+1],
            prompt=segments[i].prompt
        )
```

## Implementation Plan

### Phase 1: Gemini API Integration (Priority: High)

**Tasks:**
1. Add Gemini API client to `keyframe_generator.py`
2. Implement authentication and error handling
3. Add configuration parameters for Gemini
4. Test with sample images and prompts

**Files to Modify:**
- `keyframe_generator.py`: Add `generate_keyframe_with_gemini()` function
- `pipeline_config.yaml.sample`: Add Gemini configuration section
- `requirements.txt`: Add Google generative AI SDK

**Code Changes Required:**
```python
# In keyframe_generator.py
def generate_keyframe_with_gemini(
    prompt: str,
    output_path: str, 
    gemini_api_key: str,
    input_image_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image-preview",
    **kwargs
) -> str:
    """
    Generate or edit an image using Google Gemini API
    
    Args:
        prompt: Text prompt for generation/editing
        output_path: Where to save the result
        gemini_api_key: API key for Gemini
        input_image_path: Optional reference image for I2I
        model_name: Gemini model to use
        
    Returns:
        Path to generated image
    """
    import google.generativeai as genai
    
    # Configure API
    genai.configure(api_key=gemini_api_key)
    
    if input_image_path:
        # Image-to-image editing
        image = genai.Image.load(input_image_path)
        model = genai.GenerativeModel(model_name)
        
        response = model.edit_image(
            image=image,
            prompt=prompt,
            safety_settings=kwargs.get('safety_settings', None)
        )
    else:
        # Text-to-image generation
        model = genai.GenerativeModel(model_name)
        response = model.generate_image(
            prompt=prompt,
            safety_settings=kwargs.get('safety_settings', None)
        )
    
    # Save the result
    response.save(output_path)
    return output_path
```

### Phase 2: Keyframe Mode Enhancement (Priority: High)

**Tasks:**
1. Add I2I mode flag to pipeline configuration
2. Implement reference image management system
3. Modify `generate_keyframes_from_json()` to support I2I workflow
4. Add prompt template system for consistency

**Files to Modify:**
- `keyframe_generator.py`: Enhance batch generation logic
- `pipeline.py`: Update keyframe mode to support I2I
- `pipeline_config.yaml.sample`: Add I2I configuration options

### Phase 3: Veo 3 Keyframe Integration (Priority: Medium)

**Tasks:**
1. Update `generate_video_keyframe_mode()` to detect backend capabilities
2. Implement single-keyframe video generation for Veo 3
3. Add fallback logic for backends without FLF support
4. Test end-to-end workflow with Veo 3

**Files to Modify:**
- `pipeline.py`: Modify keyframe video generation logic
- `generators/factory.py`: Add capability detection
- `generators/remote/veo3_generator.py`: Ensure I2V mode works correctly

**Code Changes Required:**
```python
# In pipeline.py
def generate_video_keyframe_mode(config, video_prompts, output_dir):
    """Generate videos using keyframe mode"""
    
    # Get backend capabilities
    backend = config.get("default_backend", "wan2.1")
    generator = create_video_generator(backend, config)
    capabilities = generator.get_capabilities()
    
    if capabilities.get("supports_flf", False):
        # Traditional first-last-frame mode
        return generate_flf_videos(generator, video_prompts, output_dir)
    elif capabilities.get("supports_i2v", False):
        # Single keyframe I2V mode (for Veo3)
        return generate_i2v_videos(generator, video_prompts, output_dir)
    else:
        raise ValueError(f"Backend {backend} doesn't support keyframe mode")
```

## Configuration Examples

### Basic I2I Keyframe Configuration
```yaml
# pipeline_config.yaml
pipeline_mode: "keyframe"
default_backend: "veo3"

# Gemini API settings
gemini_api_key: "${GEMINI_API_KEY}"
gemini_model: "gemini-2.5-flash-image-preview"

keyframe_settings:
  mode: "image_to_image"
  image_model: "gemini"
  
  # Reference images
  reference_images:
    - path: "assets/speaker.png"
      id: "speaker"
      description: "Professional speaker at podium"
  
  # Generation settings
  sequential_generation: true  # Use previous keyframe as next reference
  maintain_consistency: true   # Apply consistency checks
```

### Advanced Multi-Character Configuration
```yaml
keyframe_settings:
  mode: "image_to_image"
  
  reference_library:
    characters:
      protagonist:
        image: "assets/protagonist.png"
        description: "Young woman, red hair, business suit"
        variations:
          - "standing"
          - "sitting"
          - "walking"
      
      antagonist:
        image: "assets/antagonist.png"
        description: "Middle-aged man, gray suit"
    
    settings:
      office: "assets/office.png"
      street: "assets/street.png"
  
  prompt_templates:
    scene_with_character: |
      {character_description} in {setting}, {action}.
      Maintain exact appearance from reference image.
    
    character_interaction: |
      {character1_description} and {character2_description}
      engaged in {interaction_type}. Same appearances as references.
```

## Testing Strategy

### Unit Tests
1. Test Gemini API integration with mock responses
2. Verify reference image loading and validation
3. Test prompt template rendering
4. Validate keyframe-to-video mapping logic

### Integration Tests
1. End-to-end I2I keyframe generation
2. Veo 3 single-keyframe video generation
3. Fallback behavior for missing references
4. Multi-backend compatibility

### Performance Tests
1. API rate limiting compliance
2. Image upload/download efficiency
3. Parallel keyframe generation
4. Memory usage with large reference libraries

## Error Handling

### API Errors
- Retry logic with exponential backoff
- Fallback to text-to-image if I2I fails
- Clear error messages for quota issues

### Image Validation
- Check reference image dimensions and formats
- Validate generated keyframes before video generation
- Handle missing or corrupted images gracefully

### Configuration Errors
- Validate all required fields at startup
- Check API key availability
- Verify reference image paths exist

## Security Considerations

1. **API Key Management**
   - Use environment variables for sensitive keys
   - Never commit API keys to repository
   - Implement key rotation support

2. **Image Privacy**
   - Don't upload reference images to public storage
   - Use temporary URLs with expiration
   - Clean up generated images after processing

3. **Rate Limiting**
   - Implement request throttling
   - Track API usage per key
   - Queue system for batch processing

## Performance Optimization

1. **Caching**
   - Cache generated keyframes for reuse
   - Store processed reference images
   - Implement smart cache invalidation

2. **Parallel Processing**
   - Generate multiple keyframes concurrently
   - Batch API requests where possible
   - Optimize image preprocessing

3. **Resource Management**
   - Limit concurrent API connections
   - Implement memory limits for image processing
   - Clean up temporary files automatically

## Migration Path

### From Current System
1. Existing text-to-image workflows continue to work
2. I2I mode is opt-in via configuration
3. Gradual migration of existing projects
4. Backward compatibility maintained

### Future Enhancements
1. Support for multiple image editing APIs
2. Advanced character pose control
3. Style transfer capabilities
4. Real-time preview generation

## Success Metrics

1. **Quality Metrics**
   - Character consistency score (visual similarity)
   - Scene coherence rating
   - User satisfaction surveys

2. **Performance Metrics**
   - Keyframe generation time
   - API success rate
   - Cost per video minute

3. **Adoption Metrics**
   - Projects using I2I mode
   - API usage statistics
   - Feature request tracking

## Dependencies

### Required Libraries
```txt
google-generativeai>=0.4.0
google-cloud-storage>=2.10.0
google-cloud-aiplatform>=1.38.0
Pillow>=10.0.0
numpy>=1.24.0
```

### API Requirements
- Gemini API key with image generation quota
- Google Cloud project with Vertex AI enabled
- Storage bucket permissions for Veo 3

## Rollout Plan

### Week 1-2: Foundation
- Implement Gemini API integration
- Basic I2I keyframe generation
- Unit tests and documentation

### Week 3-4: Integration  
- Modify pipeline for I2I mode
- Veo 3 single-keyframe support
- Integration testing

### Week 5-6: Polish
- Performance optimization
- Error handling improvements
- User documentation and examples

### Week 7-8: Release
- Beta testing with select users
- Bug fixes and refinements
- Production deployment

## Appendix A: API Response Formats

### Gemini I2I Response
```json
{
  "image": {
    "url": "https://...",
    "mime_type": "image/png",
    "dimensions": {"width": 1024, "height": 1024}
  },
  "metadata": {
    "model": "gemini-2.5-flash-image-preview",
    "prompt_tokens": 150,
    "generation_time_ms": 3500
  }
}
```

### Veo 3 Generation Request
```json
{
  "model": "veo-3.0-generate-preview",
  "image": {"gcs_uri": "gs://bucket/image.png"},
  "prompt": "Camera slowly pans right as subject walks forward",
  "config": {
    "aspect_ratio": "16:9",
    "duration": 5.0
  }
}
```

## Appendix B: Prompt Engineering Guidelines

### Character Consistency Prompts
- Always reference the original character description
- Specify "exact same appearance as reference"
- Include distinctive features in every prompt
- Use consistent naming for characters

### Scene Transition Prompts
- Maintain lighting and color palette
- Reference previous scene elements
- Gradual changes for smooth transitions
- Clear action descriptions

### Camera Movement for Veo 3
- Use cinematic terminology
- Specify speed and direction
- Match movement to narrative pace
- Consider aspect ratio in framing

---

*This specification provides a comprehensive blueprint for implementing image-to-image keyframe generation with character consistency in the TTV Pipeline. The system will enable creation of professional-quality narrative videos with consistent characters and settings throughout.*