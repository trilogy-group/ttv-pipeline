# Remote API Generators

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/4.3-remote-api-generators)*

This document covers the cloud-based video generation backends that interface with external APIs for video generation. These generators provide access to commercial video generation services without requiring local GPU resources.

**Key Source Files:**
- [`generators/remote/__init__.py`](../generators/remote/__init__.py) - Remote generators package
- [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) - Runway ML implementation
- [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) - Google Veo 3 implementation
- [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) - Minimax I2V-01-Director implementation

## Overview

The remote API generators implement the `VideoGeneratorInterface` to provide video generation through external cloud services. Currently supported APIs include Runway ML, Google Veo 3, and Minimax I2V-01-Director. These generators handle authentication, request/response processing, polling for completion, and cost estimation while maintaining the same interface as local generators.

**Key Features:**
- **Unified Interface**: Same API as local generators via `VideoGeneratorInterface`
- **Cloud Scalability**: No local GPU requirements
- **Cost Management**: Built-in cost estimation and tracking
- **Asynchronous Processing**: Polling-based completion monitoring
- **Fallback Support**: Automatic switching between API providers

*Sources: [`generators/remote/__init__.py`](../generators/remote/__init__.py) (lines 1-9), [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 1-30), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 1-56), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 1-56)*

## Architecture Overview

The remote API generators follow a consistent architectural pattern:

1. **Configuration Loading**: Extract API credentials and settings
2. **Client Initialization**: Set up authenticated API clients
3. **Input Validation**: Verify inputs meet API requirements
4. **Request Submission**: Submit generation requests asynchronously
5. **Polling Loop**: Monitor completion status with configurable intervals
6. **Result Download**: Retrieve and save generated videos
7. **Cost Tracking**: Calculate and report generation costs

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 28-66), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 55-88), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 55-88), [`video_generator_interface.py`](../video_generator_interface.py) (lines 12-19)*

## RunwayML Generator Implementation

### Core Class Structure

The `RunwayMLGenerator` class provides video generation through Runway ML's API, specifically supporting image-to-video generation using their Gen-3 and Gen-4 models.

**Key Components:**
- **API Client**: Runway ML SDK integration
- **Authentication**: API key management
- **Model Selection**: Support for multiple generation models
- **Progress Monitoring**: Real-time generation tracking

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 28-66, 138-206)*

### API Workflow and Polling

The Runway ML generation process involves creating a task, polling for completion, and downloading the result:

**Generation Workflow:**
1. **`validate_inputs()`**: Verify image format and parameters
2. **`client.image_to_video.create()`**: Submit generation request
3. **`_poll_task()`**: Monitor completion status with exponential backoff
4. **`download_file()`**: Retrieve generated video file

**Polling Strategy:**
- **Initial Interval**: 5 seconds
- **Backoff Strategy**: Exponential with jitter
- **Maximum Timeout**: Configurable (default: 10 minutes)
- **Progress Updates**: Real-time status reporting

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 112-136, 207-241)*

### Supported Models and Capabilities

The generator supports multiple Runway ML models with different pricing tiers:

**Available Models:**
- **Gen-3 Alpha**: Fast generation, lower cost
- **Gen-3 Alpha Turbo**: Fastest generation, minimal cost
- **Gen-4**: Highest quality, premium pricing

**Supported Resolutions:**
- `1280x720` (16:9 landscape)
- `720x1280` (9:16 portrait)  
- `1104x832` (4:3 landscape)
- `832x1104` (3:4 portrait)
- `960x960` (1:1 square)
- `1584x672` (21:9 ultrawide)

**Generation Parameters:**
- **Duration**: 5-10 seconds
- **Frame Rate**: 24 FPS
- **Input**: Image + text prompt
- **Output**: MP4 video file

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 31-37, 67-92)*

## Google Veo 3 Generator Implementation

### Architecture and GCS Integration

The `Veo3Generator` integrates with Google's generative AI platform and requires Google Cloud Storage for input/output handling:

**Integration Components:**
- **Vertex AI**: Core video generation service
- **Google Cloud Storage**: Input image and output video storage
- **Service Account**: Authentication and permissions
- **Regional Processing**: Location-specific API endpoints

**GCS Workflow:**
1. **Input Upload**: Upload source images to GCS bucket
2. **API Request**: Submit generation request with GCS paths
3. **Processing**: Veo 3 processes request in specified region
4. **Output Download**: Retrieve generated video from GCS
5. **Cleanup**: Remove temporary files from GCS

*Sources: [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 194-304, 323-404)*

### Authentication and Client Initialization

The Veo3 generator requires Google Cloud authentication and multiple client initialization:

**Required Clients:**
- **`genai.Client()`**: Generative AI client for video generation
- **`storage.Client()`**: GCS client for file operations

**Environment Variables:**
- **`GOOGLE_CLOUD_PROJECT`**: GCP project identifier
- **`GOOGLE_APPLICATION_CREDENTIALS`**: Path to service account key
- **`GOOGLE_CLOUD_LOCATION`**: Processing region

**Authentication Options:**
1. **Service Account File**: Explicit credentials file path
2. **Application Default Credentials**: Automatic credential discovery
3. **Environment Variables**: Direct credential specification

The initialization process sets environment variables and loads credentials from either a service account file or application default credentials.

*Sources: [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 89-117, 66-88)*

### Veo 3 Capabilities and Constraints

The Veo 3 generator has specific limitations and features:

**Capabilities:**
- **Maximum Duration**: 10 seconds
- **Resolution**: 1280x720 (16:9)
- **Input Aspects**: 1:1 square images recommended
- **Quality**: High-fidelity video generation
- **Consistency**: Strong temporal coherence

**Constraints:**
- **Prompt Length**: Maximum 1000 characters
- **Image Requirements**: Specific aspect ratio requirements
- **Processing Time**: 2-5 minutes typical
- **Regional Availability**: Limited to specific GCP regions
- **Access Control**: Requires allowlisting for image-to-video

**Input Validation:**
- Image aspect ratio verification
- Prompt length checking
- Format compatibility validation
- GCS bucket accessibility

Input validation ensures images meet aspect ratio requirements and prompts don't exceed 1000 characters.

*Sources: [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 119-138, 155-192)*

## Minimax I2V-01-Director Generator Implementation

### Overview

The Minimax generator utilizes the I2V-01-Director model for image-to-video generation with sophisticated camera movement control. This backend excels at creating dynamic videos with specific camera motions and effects.

### Key Features

- **Image-to-Video Generation**: Convert static images to dynamic video content
- **Camera Movement Control**: Support for various camera movements including truck, pan, tilt, zoom, and dolly operations
- **Prompt Enhancement**: Automatic enhancement of prompts with camera movement specifications
- **Cost-Effective**: Competitive pricing for high-quality video generation
- **Real-time Processing**: Fast generation times with synchronous API responses

### Configuration

```yaml
minimax:
  api_key: "YOUR_MINIMAX_API_KEY"
  model: "I2V-01-Director"
  max_duration: 6
  base_url: "https://api.minimaxi.chat/v1"
```

#### Configuration Parameters

- **`api_key`**: Minimax API authentication key (required)
- **`model`**: Model version, currently "I2V-01-Director" (default)
- **`max_duration`**: Maximum video duration in seconds (default: 6)
- **`base_url`**: API endpoint base URL (default: "https://api.minimaxi.chat/v1")

### Camera Movement Support

The Minimax backend supports various camera movements that can be specified in prompts:

#### Supported Movements
- **Truck left/right**: Lateral camera movement
- **Pan left/right**: Horizontal camera rotation
- **Tilt up/down**: Vertical camera rotation
- **Zoom in/out**: Camera zoom operations
- **Dolly in/out**: Forward/backward camera movement
- **Orbital**: Circular camera movement around subject
- **Static**: No camera movement

#### Usage Examples

```python
# Explicit camera movement in prompt
prompt = "[Truck left,Pan right]A woman is drinking coffee."

# Multiple movements
prompt = "[Zoom in,Tilt up]A sunset over mountains."

# Static shot
prompt = "[Static]A peaceful garden scene."
```

### API Workflow

#### 1. Authentication
```python
headers = {
    'authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
```

#### 2. Image Encoding
Images are encoded to base64 format with proper MIME type specification:
```python
# JPEG format
f"data:image/jpeg;base64,{base64_data}"

# PNG format  
f"data:image/png;base64,{base64_data}"
```

#### 3. Request Submission
```python
payload = {
    "model": "I2V-01-Director",
    "prompt": "[Truck left,Pan right]A woman is drinking coffee.",
    "first_frame_image": f"data:image/jpeg;base64,{base64_data}"
}

response = requests.post(
    "https://api.minimaxi.chat/v1/video_generation",
    headers=headers,
    json=payload
)
```

#### 4. Response Processing
The API typically returns either:
- **Synchronous response**: Direct video URL in response
- **Asynchronous response**: Task ID for polling completion status

### Capabilities and Limitations

#### Capabilities
- **Maximum Duration**: 6 seconds per generation
- **Supported Formats**: MP4 output
- **Resolution Options**: 720x1280, 1280x720, 1024x1024
- **Image Input**: JPEG, PNG formats up to 10MB
- **Prompt Length**: Up to 500 characters
- **Camera Movements**: 12+ supported movement types

#### Limitations
- **Text-to-Video**: Not supported (image-to-video only)
- **Duration Constraints**: Limited to shorter clips compared to some competitors
- **Regional Availability**: May have geographic restrictions
- **Rate Limits**: Subject to API rate limiting

### Error Handling

The Minimax generator implements comprehensive error handling:

#### Common Error Types
- **401 Unauthorized**: Invalid API key or authentication failure
- **403 Forbidden**: Quota exceeded or access denied
- **400 Bad Request**: Invalid input parameters or format
- **429 Rate Limited**: Too many requests
- **500 Server Error**: Minimax service issues

#### Error Recovery
```python
try:
    video_path = generator.generate_video(prompt, image_path, output_path)
except QuotaExceededError:
    logger.error("API quota exceeded")
except InvalidInputError as e:
    logger.error(f"Invalid input: {e}")
except APIError as e:
    logger.error(f"API error: {e}")
```

### Cost Management

#### Pricing Structure
- **Estimated Cost**: ~$0.02 per second of generated video
- **Transparent Pricing**: Cost estimation before generation
- **No Hidden Fees**: Pay only for successful generations

#### Cost Optimization
```python
# Pre-generate cost estimation
estimated_cost = generator.estimate_cost(duration=5.0)
logger.info(f"Estimated cost: ${estimated_cost:.2f}")

# Optimize duration for cost
max_affordable_duration = budget / 0.02  # $0.02 per second
```

### Integration Example

```python
# Configuration
config = {
    "api_key": "your_minimax_api_key",
    "model": "I2V-01-Director",
    "max_duration": 6,
    "max_retries": 3,
    "timeout": 300
}

# Initialize generator
from generators.remote.minimax_generator import MinimaxGenerator
generator = MinimaxGenerator(config)

# Generate video with camera movement
prompt = "[Pan right,Zoom in]A bustling city street at sunset."
try:
    video_path = generator.generate_video(
        prompt=prompt,
        input_image_path="input.jpg",
        output_path="output.mp4",
        duration=5.0
    )
    print(f"Video generated: {video_path}")
except Exception as e:
    print(f"Generation failed: {e}")
```

### Best Practices

#### Prompt Optimization
1. **Be Specific**: Include detailed scene descriptions
2. **Camera Movements**: Specify desired camera motions explicitly
3. **Scene Continuity**: Ensure prompt matches input image content
4. **Length Management**: Keep prompts under 500 characters

#### Performance Optimization
1. **Image Quality**: Use high-quality input images (1024x1024 recommended)
2. **Appropriate Duration**: Match duration to content complexity
3. **Network Stability**: Ensure stable internet connection for uploads
4. **Retry Logic**: Implement proper retry mechanisms for transient failures

#### Troubleshooting

**Common Issues:**
1. **Authentication Failures**: Verify API key validity and format
2. **Image Format Errors**: Ensure proper JPEG/PNG format and size limits
3. **Timeout Issues**: Increase timeout for complex generations
4. **Quality Issues**: Optimize input image quality and prompt specificity

**Debug Steps:**
1. Test API connectivity with simple requests
2. Validate image encoding and format
3. Check prompt length and format
4. Monitor API response codes and messages

## Configuration and Authentication

### Configuration Structure

Both generators require specific configuration sections in `pipeline_config.yaml`:

**Runway ML Configuration:**
```yaml
runway_ml:
  api_key: "${RUNWAY_API_KEY}"
  model: "gen3a_turbo"
  timeout: 600
  polling_interval: 5
  max_retries: 3
```

**Google Veo 3 Configuration:**
```yaml
google_veo:
  project_id: "${GCP_PROJECT_ID}"
  credentials_path: "./gcp-credentials.json"
  region: "us-central1"
  bucket_name: "ttv-pipeline-storage"
  timeout: 900
  polling_interval: 10
```

**Minimax Configuration:**
```yaml
minimax:
  api_key: "YOUR_MINIMAX_API_KEY"
  model: "I2V-01-Director"
  max_duration: 6
  base_url: "https://api.minimaxi.chat/v1"
```

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 39-56), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 66-82), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 66-82)*

### Environment Variable Handling

Both generators support environment variable authentication as fallbacks:

**Runway ML Environment Variables:**
- **`RUNWAYML_API_SECRET`**: Primary API key
- **`RUNWAY_API_KEY`**: Alternative API key

**Google Veo 3 Environment Variables:**
- **`GOOGLE_CLOUD_PROJECT`**: GCP project identifier
- **`GOOGLE_APPLICATION_CREDENTIALS`**: Path to service account key
- **`GOOGLE_CLOUD_LOCATION`**: Processing region

**Minimax Environment Variables:**
- **`MINIMAX_API_KEY`**: Minimax API authentication key

**Environment Variable Priority:**
1. Configuration file values
2. Environment variables
3. Default values (where applicable)

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 50-56), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 92-96), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 92-96)*

## Error Handling and Monitoring

### Retry Logic and Progress Monitoring

Both generators implement comprehensive error handling with retry capabilities:

**Retry Strategies:**
- **Network Errors**: Automatic retry with exponential backoff
- **Rate Limiting**: Respect API rate limits with intelligent delays
- **Transient Failures**: Distinguish between permanent and temporary errors
- **Quota Errors**: Graceful handling of quota exceeded scenarios

**Progress Monitoring:**
- **Status Updates**: Real-time generation progress
- **Time Estimation**: Predicted completion times
- **Error Reporting**: Detailed error context and suggestions
- **Cost Tracking**: Running cost calculations

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 64-65), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 83-84), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 83-84), [`generators/base.py`](../generators/base.py)*

### Polling and Timeout Management

Both generators implement polling mechanisms with configurable timeouts:

**Polling Configuration:**
- **`polling_interval`**: Time between status checks (default: 5-10 seconds)
- **`timeout`**: Maximum wait time (default: 10-15 minutes)
- **`max_retries`**: Maximum retry attempts (default: 3)

**Timeout Strategies:**
- **Progressive Backoff**: Increase intervals on repeated failures
- **Adaptive Timing**: Adjust based on typical generation times
- **Early Termination**: Cancel on clear failure indicators
- **Resource Cleanup**: Ensure cleanup on timeout

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 47-48), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 72-73), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 72-73)*

## Cost Estimation

### Pricing Models

Both generators provide cost estimation based on duration and model selection:

**Runway ML Pricing:**
- **Gen-3 Alpha Turbo**: $0.05 per second
- **Gen-3 Alpha**: $0.075 per second  
- **Gen-4**: $0.125 per second

**Google Veo 3 Pricing:**
- **Standard Generation**: $0.20 per second
- **Regional Variations**: Pricing may vary by GCP region
- **Volume Discounts**: Available for high-usage scenarios

**Minimax Pricing:**
- **Estimated Cost**: ~$0.02 per second of generated video
- **Transparent Pricing**: Cost estimation before generation
- **No Hidden Fees**: Pay only for successful generations

**Cost Calculation Features:**
- **Pre-Generation Estimates**: Calculate costs before starting
- **Real-Time Tracking**: Monitor costs during generation
- **Historical Analysis**: Track spending over time
- **Budget Warnings**: Alert when approaching spending limits

The `estimate_cost()` method provides upfront cost estimates before generation begins, helping users understand the financial impact of their requests.

**Example Cost Estimation:**
```python
def estimate_cost(self, duration: float, model: str = "gen3a_turbo") -> float:
    model_costs = {
        "gen3a_turbo": 0.05,
        "gen3a": 0.075,
        "gen4": 0.125
    }
    return duration * model_costs.get(model, 0.075)
```

*Sources: [`generators/remote/runway_generator.py`](../generators/remote/runway_generator.py) (lines 94-110), [`generators/remote/veo3_generator.py`](../generators/remote/veo3_generator.py) (lines 140-153), [`generators/remote/minimax_generator.py`](../generators/remote/minimax_generator.py) (lines 140-153)*

## Usage Examples

### Runway ML Generation

```python
# Configuration
config = {
    "api_key": "runway_api_key_here",
    "model": "gen3a_turbo",
    "timeout": 600
}

# Initialize generator
generator = RunwayMLGenerator(**config)

# Generate video
output_path = generator.generate_video(
    prompt="A serene lake at sunset",
    image_path="input.jpg",
    duration=5.0
)
```

### Google Veo 3 Generation

```python
# Configuration  
config = {
    "project_id": "my-gcp-project",
    "credentials_path": "./gcp-key.json",
    "region": "us-central1"
}

# Initialize generator
generator = Veo3Generator(**config)

# Generate video
output_path = generator.generate_video(
    prompt="Ocean waves crashing on rocks",
    image_path="beach.jpg",
    duration=8.0
)
```

### Minimax Generation

```python
# Configuration
config = {
    "api_key": "your_minimax_api_key",
    "model": "I2V-01-Director",
    "max_duration": 6,
    "max_retries": 3,
    "timeout": 300
}

# Initialize generator
from generators.remote.minimax_generator import MinimaxGenerator
generator = MinimaxGenerator(config)

# Generate video with camera movement
prompt = "[Pan right,Zoom in]A bustling city street at sunset."
try:
    video_path = generator.generate_video(
        prompt=prompt,
        input_image_path="input.jpg",
        output_path="output.mp4",
        duration=5.0
    )
    print(f"Video generated: {video_path}")
except Exception as e:
    print(f"Generation failed: {e}")
```

---

## Next Steps

- **Local Alternative**: See [Local Generators](06-local-generators.md) for GPU-based generation
- **Configuration**: See [Getting Started](02-getting-started.md) for API setup
- **Architecture**: See [Architecture and Interface](05-architecture-and-interface.md) for implementation details
- **Deployment**: See [Deployment and Containers](08-deployment-and-containers.md) for production setup
