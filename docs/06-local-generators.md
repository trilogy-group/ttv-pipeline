# Local Generators

*Source: [DeepWiki Analysis](https://deepwiki.com/trilogy-group/ttv-pipeline/4.2-local-generators)*

Local Generators handle GPU-based video generation that runs directly on the user's hardware, as opposed to remote API-based generation. This subsystem provides high-performance video generation using locally installed models without incurring API costs or external dependencies.

**Key Source Files:**
- [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py) - Main local generator implementation

## Purpose and Scope

Local Generators handle GPU-based video generation that runs directly on the user's hardware, as opposed to remote API-based generation. This subsystem provides high-performance video generation using locally installed models without incurring API costs or external dependencies.

The local generation implementation centers around the Wan2.1 framework, which provides state-of-the-art Image-to-Video (I2V) generation capabilities. For information about remote API-based generators, see [Remote API Generators](07-remote-api-generators.md). For the overall architecture and interface design, see [Architecture and Interface](05-architecture-and-interface.md).

## Wan21Generator Implementation

The `Wan21Generator` class serves as the primary local video generation backend, implementing the `VideoGeneratorInterface` to provide a consistent API for the pipeline system.

### Wan21Generator Component Architecture

The generator handles the complete lifecycle from configuration validation through video output generation, providing robust error handling and GPU resource management.

**Architecture Flow:**
1. **Configuration Validation**: Verify all required parameters and paths
2. **GPU Resource Check**: Validate available CUDA devices
3. **Model Availability**: Ensure required model files are accessible
4. **Command Generation**: Build appropriate execution commands
5. **Process Management**: Execute and monitor generation process
6. **Output Validation**: Verify successful video generation

*Source: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)*

## Configuration Parameters

The `Wan21Generator` accepts comprehensive configuration through the pipeline configuration system:

### Required Parameters

- **`wan2_dir`**: Path to Wan2.1 framework directory
- **`i2v_model_dir`**: Path to Image-to-Video model checkpoint directory
- **`flf2v_model_dir`**: Path to First-Last-Frame-to-Video model directory

### Optional Parameters

- **`gpu_count`**: Number of GPUs to use for generation (default: auto-detect)
- **`size`**: Output video resolution (default: 1280x720)
- **`guide_scale`**: Guidance scale for generation quality (default: 7.5)
- **`sample_steps`**: Number of sampling steps (default: 50)
- **`sample_shift`**: Sampling shift parameter (default: 1.0)
- **`frame_num`**: Number of frames to generate (default: 81)

### Advanced Parameters

- **`chaining_max_retries`**: Maximum retry attempts for chaining mode (default: 3)
- **`chaining_use_fsdp_flags`**: Enable FSDP for distributed processing (default: true)

The generator validates all configuration parameters during initialization and provides detailed error messages for missing or invalid configurations.

*Source: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)*

## Command Generation and Execution

The video generation process involves building and executing commands that interface with the Wan2.1 framework:

### Command Building and Execution Flow

The command builder dynamically constructs execution commands based on GPU configuration and model parameters, handling both single-GPU and distributed multi-GPU scenarios.

**Base Command Structure:**
- **Task specification**: Uses `i2v-14B` task for Image-to-Video generation
- **Model paths**: References I2V model checkpoint directory
- **Generation parameters**: Includes guidance scale, sampling steps, and frame count
- **Output configuration**: Specifies save path and video dimensions

**Single-GPU Command:**
```bash
python generate.py \
    --save_file output.mp4 \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./models/Wan2.1-I2V-14B-720P \
    --image input.jpg \
    --prompt "A beautiful landscape" \
    --sample_guide_scale 7.5 \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --frame_num 81
```

**Multi-GPU Command:**
For multi-GPU setups, the generator automatically configures `torchrun` with appropriate process counts and adds FSDP flags for memory-efficient distributed processing:

```bash
torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29500 generate.py \
    --save_file output.mp4 \
    --task i2v-14B \
    --size 1280*720 \
    --ckpt_dir ./models/Wan2.1-I2V-14B-720P \
    --image input.jpg \
    --prompt "A beautiful landscape" \
    --sample_guide_scale 7.5 \
    --sample_steps 50 \
    --sample_shift 5.0 \
    --frame_num 81 \
    --ulysses_size 2 \
    --dit_fsdp --t5_fsdp
```

*Source: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)*

## Multi-GPU Support and FSDP

The `Wan21Generator` provides sophisticated multi-GPU support through PyTorch's distributed training capabilities:

### Multi-GPU Processing Architecture

**FSDP (Fully Sharded Data Parallel)** enables memory-efficient processing of large models by sharding parameters across multiple GPUs, allowing generation of high-quality videos that would exceed single-GPU memory limits.

**Key Features:**
- **Automatic GPU Detection**: Detects available CUDA devices
- **Resource Validation**: Ensures requested GPU count matches available hardware
- **Memory Optimization**: Uses FSDP to reduce per-GPU memory requirements
- **Process Coordination**: Manages distributed processes through torchrun

**FSDP Benefits:**
- **Reduced Memory Usage**: Model parameters are sharded across GPUs
- **Larger Model Support**: Enables processing of models that don't fit on single GPU
- **Scalable Performance**: Performance scales with additional GPUs
- **Fault Tolerance**: Handles GPU failures gracefully

The generator automatically detects available GPU resources and validates that the requested GPU count matches available hardware before attempting distributed execution.

*Source: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)*

## Error Handling and Retry Logic

The local generator implements comprehensive error handling with intelligent retry mechanisms:

### Error Handling and Retry Flow

The retry mechanism progressively reduces generation parameters to increase the likelihood of successful execution on resource-constrained systems. This approach balances quality with reliability.

**Key Error Handling Features:**

1. **Parameter Reduction**: Automatically reduces `sample_steps` and `guide_scale` on retry attempts
   - First retry: Reduce sample_steps by 25%
   - Second retry: Reduce guide_scale to 5.0
   - Final retry: Use minimal parameters for maximum compatibility

2. **Output Validation**: Verifies generated files exist and meet minimum size requirements
   - File existence check
   - Minimum file size validation
   - Video format verification

3. **Detailed Logging**: Provides comprehensive error reporting for debugging
   - Command execution logs
   - GPU memory status
   - Model loading diagnostics

4. **Resource Checking**: Validates GPU availability and model accessibility before execution
   - CUDA device availability
   - Model file accessibility
   - Sufficient disk space

**Retry Strategy:**
```python
def _retry_with_reduced_params(self, original_params):
    retries = [
        {"sample_steps": original_params["sample_steps"] * 0.75},
        {"guide_scale": 5.0},
        {"sample_steps": 25, "guide_scale": 3.0}
    ]
    
    for retry_params in retries:
        try:
            return self._execute_generation(retry_params)
        except Exception as e:
            log_retry_attempt(e, retry_params)
            continue
```

*Source: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py)*

## Input Validation and Capabilities

The generator provides comprehensive input validation and capability reporting:

### Input Validation and Capability System

The validation system ensures all inputs meet Wan2.1 model requirements before attempting generation, preventing resource waste on invalid requests.

### Supported Capabilities

- **Maximum duration**: 10 seconds (approximately 81 frames)
- **Resolutions**: 1280×720, 1024×576, 720×480
- **Input modes**: Image-to-Video generation
- **GPU requirement**: CUDA-capable hardware required
- **Cost model**: Zero cost for local processing

### Input Validation Process

The `ImageValidator` performs detailed image format and dimension validation to ensure compatibility with the underlying Wan2.1 models:

**Validation Checks:**
1. **Image Format**: Supports JPEG, PNG, WebP formats
2. **Dimensions**: Validates resolution matches supported options
3. **Aspect Ratio**: Ensures proper aspect ratio for model input
4. **File Size**: Checks for reasonable file size limits
5. **Color Space**: Validates RGB color space compatibility

**Capability Response:**
```python
def get_capabilities(self) -> Dict[str, Any]:
    return {
        "max_duration": 10.0,
        "supported_resolutions": [
            "1280x720", "1024x576", "720x480"
        ],
        "supported_formats": ["mp4"],
        "input_types": ["image_to_video"],
        "requires_gpu": True,
        "cost_per_second": 0.0,
        "max_concurrent": self.gpu_count
    }
```

*Sources: [`generators/local/wan21_generator.py`](../generators/local/wan21_generator.py) (lines 38-51, 57-84, 195-236)*

## HunyuanVideoGenerator

The `HunyuanVideoGenerator` provides local video generation using Tencent's open-source HunyuanVideo framework. It mirrors the `VideoGeneratorInterface` and runs the repository's `inference.py` script under the hood.

### Configuration Parameters

- **`hunyuan_dir`** – Directory containing the HunyuanVideo code
- **`config_file`** – Path to the model configuration YAML
- **`ckpt_path`** – Path to the pre-trained weights
- **`sample_steps`** – Sampling steps (default: 50)
- **`max_duration`** – Maximum generation duration

*Source: [`generators/local/hunyuan_video_generator.py`](../generators/local/hunyuan_video_generator.py)*

## Performance Optimization

### GPU Memory Management

The generator implements several optimization strategies for efficient GPU utilization:

**Memory Optimization Techniques:**
- **FSDP Sharding**: Distributes model parameters across multiple GPUs
- **Gradient Checkpointing**: Trades compute for memory by recomputing gradients
- **Mixed Precision**: Uses FP16 precision to reduce memory usage
- **Dynamic Batching**: Adjusts batch size based on available memory

### Performance Monitoring

**Key Metrics Tracked:**
- GPU memory utilization
- Generation time per frame
- Model loading time
- Disk I/O performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - **Solution**: Reduce `sample_steps` or use fewer GPUs
   - **Prevention**: Monitor GPU memory before generation

2. **Model Not Found**
   - **Solution**: Verify model paths in configuration
   - **Check**: Ensure setup.sh completed successfully

3. **Generation Timeout**
   - **Solution**: Increase timeout values or reduce quality parameters
   - **Monitor**: Check GPU utilization during generation

4. **Invalid Input Image**
   - **Solution**: Verify image format and dimensions
   - **Tools**: Use validation methods before generation

---

## Next Steps

- **Remote APIs**: See [Remote API Generators](07-remote-api-generators.md) for cloud alternatives
- **Setup Guide**: See [Getting Started](02-getting-started.md) for installation instructions
- **Configuration**: See [Video Generation Backends](04-video-generation-backends.md) for backend selection
- **Interface Details**: See [Architecture and Interface](05-architecture-and-interface.md) for implementation specifics
