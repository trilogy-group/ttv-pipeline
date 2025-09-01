# I2I Keyframes Implementation Reference

## Overview
This document provides a comprehensive technical reference for the Image-to-Image (I2I) keyframe generation implementation in the TTV Pipeline, using Google's Gemini-2.5-Flash-Image-Preview model for character and setting consistency.

## Core Components

### 1. Keyframe Generation Module
**File:** `/Users/magos/dev/trilogy/ttv/keyframe_generator.py`

#### Main Functions

##### `generate_keyframe_with_gemini()`
**Location:** Lines 408-520  
**Signature:**
```python
def generate_keyframe_with_gemini(
    prompt: str,
    output_path: str,
    gemini_api_key: str,
    input_image_path: Optional[str] = None,
    reference_images_dir: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image-preview",
    max_retries: int = 2
) -> str
```
**Purpose:** Generates keyframes using Gemini API with I2I capabilities
**Features:**
- Loads all reference images from specified directory
- Supports input image for I2I transformation
- Implements exponential backoff retry mechanism
- Returns absolute path to generated image

##### `generate_keyframe()`
**Location:** Lines 522-643  
**Signature:**
```python
def generate_keyframe(
    prompt, output_path, model_name,
    imageRouter_api_key=None,
    stability_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    input_image_path=None,
    mask_path=None,
    size=None,
    create_mask=False,
    reference_images_dir=None
)
```
**Purpose:** Wrapper function that routes to appropriate API based on model name
**Key Logic:**
- Checks for "gemini" in model_name to use Gemini API
- Falls back to other APIs (OpenAI, Stability, ImageRouter) as configured

##### `generate_keyframes_from_json()`
**Location:** Lines 645-745  
**Signature:**
```python
def generate_keyframes_from_json(
    json_file, output_dir, model_name=None,
    imageRouter_api_key=None,
    stability_api_key=None,
    openai_api_key=None,
    gemini_api_key=None,
    initial_image_path=None,
    image_size=None,
    reference_images_dir=None
)
```
**Purpose:** Batch generation of keyframes from JSON prompts
**Features:**
- Sequential generation for character consistency
- Uses previous keyframe as input for next generation
- Supports reference images directory for consistency

### 2. Pipeline Integration
**File:** `/Users/magos/dev/trilogy/ttv/pipeline.py`

#### Key Functions

##### `generate_keyframes()`
**Location:** Lines 223-264  
**Purpose:** Main keyframe generation orchestrator
**Key Parameters:**
- `reference_images_dir`: Directory containing reference images
- Passes through to `generate_keyframes_from_json()`

##### `generate_video_segments()`
**Location:** Lines 435-564  
**Purpose:** Generates video segments from keyframes
**I2I Integration:**
- Lines 559-593: Auto-generates initial reference image if needed
- Uses existing prompting system without modification
- Creates reference_images directory if not provided

##### `generate_video_segments_single_keyframe()`
**Location:** Lines 566-654  
**Purpose:** Single-keyframe video generation for Veo3 compatibility
**Features:**
- Selects keyframe based on `keyframe_position` config (first/middle/last)
- Uses VideoGeneratorInterface for backend abstraction
- Supports fallback generators

### 3. Configuration
**File:** `/Users/magos/dev/trilogy/ttv/pipeline_config.yaml.sample`

#### I2I Mode Configuration
**Location:** Lines 132-146
```yaml
i2i_mode:
  enabled: false                          # Enable I2I keyframe generation mode
  backend: "gemini"                       # Backend for I2I generation
  model: "gemini-2.5-flash-image-preview" # Gemini model for I2I
  reference_images_dir: "path/to/reference/images"  # Reference images directory
  auto_generate_initial: true              # Auto-generate if no references
  single_keyframe_mode: true              # For Veo3 (no FLF support)
  keyframe_position: "first"              # Position: first/middle/last

gemini_api_key: "YOUR_GEMINI_API_KEY"    # Required for Gemini
```

### 4. Video Generator Support
**File:** `/Users/magos/dev/trilogy/ttv/generators/remote/veo3_generator.py`

#### Veo3Generator Class
**Capabilities:**
- `supports_image_to_video = True`
- `supports_first_last_frame = False`
- Requires 1024x1024 square images
- Uses `veo-3.0-generate-preview` model

## Data Flow

### 1. Initial Setup
```
User Config → pipeline.py → Check i2i_mode.enabled
                          → Check reference_images_dir
                          → Auto-generate initial if needed
```

### 2. Keyframe Generation Flow
```
Pipeline → generate_keyframes() → generate_keyframes_from_json()
                                → generate_keyframe() (per prompt)
                                → generate_keyframe_with_gemini()
                                → Gemini API with reference images
```

### 3. Video Generation Flow
```
Keyframes → generate_video_segments() → Check single_keyframe_mode
                                      → generate_video_segments_single_keyframe()
                                      → Veo3Generator.generate()
                                      → Final video segments
```

## Key Design Decisions

### 1. Reference Images Directory
- **Flexible loading:** All images in directory are automatically loaded
- **Naming convention:** Filename (without extension) used as reference name
- **Supported formats:** .png, .jpg, .jpeg, .webp, .bmp

### 2. Auto-Generation of Initial Image
- **Trigger:** When `auto_generate_initial=true` and no reference images exist
- **Location:** Creates `reference_images/` subdirectory in output
- **Source:** Uses first prompt from video segments
- **API:** Uses configured keyframe_prompt_model (defaults to gpt-image-1)

### 3. Single-Keyframe Mode
- **Purpose:** Veo3 compatibility (no FLF support)
- **Selection:** Based on `keyframe_position` setting
- **Fallback:** Automatic fallback to configured backup generators

## API Integration

### Gemini API
- **Model:** gemini-2.5-flash-image-preview
- **Package:** google-generativeai>=0.8.0
- **Authentication:** Via GEMINI_API_KEY environment variable
- **Retry:** Exponential backoff with max_retries=2

### Veo3 API
- **Model:** veo-3.0-generate-preview
- **Requirements:** GCP_PROJECT_ID, proper OAuth scopes
- **Image format:** 1024x1024 square images
- **Video output:** Configurable aspect ratio (e.g., 16:9)

## Environment Variables
```bash
export GEMINI_API_KEY="your-gemini-key"
export GCP_PROJECT_ID="your-gcp-project"
```

## Usage Examples

### Basic I2I Pipeline Run
```python
config = {
    'image_generation_model': 'gemini-2.5-flash-image-preview',
    'reference_images_dir': './references',
    'image_size': '1024x1024',
    'auto_generate_initial': True,
    'single_keyframe_mode': True,
    'keyframe_position': 'first',
    'gemini_api_key': 'YOUR_KEY',
    'video_aspect_ratio': '16:9'
}
```

### Manual Reference Images
1. Create reference directory
2. Add character/setting images
3. Run pipeline with `auto_generate_initial: false`

### Automatic Initial Generation
1. Leave reference_images_dir empty or unset
2. Set `auto_generate_initial: true`
3. Pipeline generates from first prompt

## Error Handling

### Missing API Keys
- Checks for gemini_api_key in config
- Falls back to environment variable
- Raises ValueError if not found

### Reference Image Loading
- Logs each loaded image
- Skips non-image files
- Continues if directory empty (with auto-generation)

### Generation Failures
- Exponential backoff retry
- Fallback generator support
- Comprehensive error logging

## Testing

### Unit Test Coverage
- `test_gemini_keyframe_generation()`
- `test_reference_image_loading()`
- `test_auto_initial_generation()`
- `test_single_keyframe_selection()`

### Integration Testing
```bash
# Test with demo script
python examples/i2i_keyframe_demo.py --mode single
python examples/i2i_keyframe_demo.py --mode consistency
python examples/i2i_keyframe_demo.py --mode pipeline
```

## Performance Considerations

### Gemini API
- Rate limits: Check Gemini documentation
- Cost: Per-image generation pricing
- Latency: ~2-5 seconds per image

### Veo3 API
- Queue time: Variable based on load
- Cost: ~$0.08 per second of video
- Max duration: 10 seconds per segment

## Future Enhancements

1. **Multi-model support:** Add DALL-E 3, Stability AI I2I
2. **Caching:** Cache generated keyframes for reuse
3. **Batch processing:** Parallel keyframe generation
4. **UI integration:** Web interface for configuration
5. **Advanced consistency:** Style transfer techniques
