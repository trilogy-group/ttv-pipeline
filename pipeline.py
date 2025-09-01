#!/usr/bin/env python3
"""
Long Video Generation Pipeline

This script orchestrates the generation of long-form videos by:
1. Breaking a long prompt into segments using OpenAI
2. Generating keyframe images using a T2I model
3. Creating video segments using Wan2.1 FLF2V model
4. Stitching the segments together

Instead of modifying the Wan2.1 codebase, this acts as a wrapper
that calls the original scripts with appropriate parameters.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict
import time
import yaml
from frame_extractor import extract_last_frame
from generators.factory import create_video_generator, get_fallback_generator
from video_generator_interface import VideoGenerationError, APIError, GenerationTimeoutError, InvalidInputError, QuotaExceededError
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from instructor import from_openai, Mode
from api.config_merger import ConfigMerger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_path}: {e}")
        raise

# Terminal colors for pretty output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'  # Reset to default

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

# ============================================================================
# Pydantic Models for Structured Prompt Enhancement
# ============================================================================

class SegmentationLogic(BaseModel):
    total_duration_seconds: int
    number_of_segments: int
    reasoning: str

class KeyframePrompt(BaseModel):
    segment: int
    prompt: str

class VideoPrompt(BaseModel):
    segment: int
    prompt: str
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None

class PromptEnhancementResult(BaseModel):
    segmentation_logic: SegmentationLogic
    keyframe_prompts: List[KeyframePrompt]
    video_prompts: List[VideoPrompt]

# Instructions for prompt splitting and enhancement
PROMPT_ENHANCEMENT_INSTRUCTIONS = """
Split and enhance an input text-to-video prompt and starting reference image into detailed, standalone prompts for 5-second video segments. This requires narrative pacing, descriptive clarity, cinematic knowledge, and continuity skills.

Analyze the Input Prompt and Starting Reference Image
- Prompt Analysis: Identify the setting, characters, actions, camera movements, and style (e.g., mood, lighting).
- Image Analysis: Examine the starting reference image (`provided_start_image.png`) for visual context, such as character positions or environmental details.
- Continuity: Ensure each prompt is standalone, with full descriptions (e.g., "A woman in a red dress, tall and athletic") and uses reference images to link segments visually.

Determine Duration and Segmentation
- Estimate the total duration based on the complexity and pacing of actions.
- Divide into 5-second segments, each with distinct start and end actions.
- Balance the narrative flow across segments for a cohesive story.

Enhance Prompt Specificity
- Include complete character descriptions in every segment.
- Define spatial and temporal details (e.g., "on the left side of the street").
- Use precise camera terms (e.g., "tracking shot," "slow pan").
- Maintain consistent style (e.g., "cinematic lighting").
- Use reference images for continuity rather than relying on text alone.

Prompting Guidelines: Effective Prompting Techniques
- Lead with Action, Not Just Description: Use verbs and motion to guide animation. Example: "A woman in a red dress running across a rainy street at night, cinematic lighting, camera tracking behind her."
- Specify Camera Movement: Use terms like "tracking shot," "dolly zoom," or "slow push-in." Example: "Close-up shot of a black panther crouching in tall grass at night, magical blue eyes glowing, slow push-in from camera."
- Establish Lighting & Mood: Include descriptive terms like "cinematic lighting," "dramatic shadows," or "volumetric fog." Example: "A scientist in a laboratory, blue-tinted lighting from computer screens, dramatic shadows, shallow depth of field."

Generate Keyframe Prompts
- Write text-to-image prompts for the last frame of each segment.
- Focus on the static scene: characters, objects, setting, and style.
- Note that each segment is prompted separate, with no knowledge of prior segments other than the reference image. Therefore, if you refer to characters by name, be sure to explain who they are in the prompt.
- Exclude camera movement; emphasize visual composition.
- Each keyframe prompt generates an image (e.g., `segment_01.png`) that becomes the last frame of its segment and the first frame of the next segment.

Generate Video Prompts
- Write text-to-video prompts for each 5-second segment.
- Detail actions, camera movements, and style.
- Reference the first frame (`provided_start_image.png` for segment 1, `segment_XX.png` for others) and last frame (`segment_YY.png`).
- Ensure each prompt is standalone with full context.

Reference Image Naming
- The starting image is `provided_start_image.png` (used as the first frame of segment 1).
- Each segment's last frame is named sequentially (e.g., `segment_01.png`, `segment_02.png`).
- The last frame of one segment (e.g., `segment_01.png`) becomes the first frame of the next segment, ensuring visual continuity.

Output Format
- Use JSON with:
  - 'segmentation_logic': Includes 'total_duration_seconds', 'number_of_segments', and 'reasoning'.
  - 'keyframe_prompts': List of segment-specific prompts for last frames.
  - 'video_prompts': List of segment-specific prompts with reference image file names.

1-Shot Example
Input Prompt: "A woman in a red dress running across a rainy street at night, cinematic lighting, camera tracking behind her."

Expected Output (JSON):
{
  "segmentation_logic": {
    "total_duration_seconds": 10,
    "number_of_segments": 2,
    "reasoning": "The action of running across the street is split into starting the run and completing the crossing."
  },
  "keyframe_prompts": [
    {
      "segment": 1,
      "prompt": "A woman in a red dress, mid-stride, running on a rainy street at night, cinematic lighting, viewed from behind, streetlights reflecting on wet pavement."
    },
    {
      "segment": 2,
      "prompt": "A woman in a red dress, reaching the other side of a rainy street at night, cinematic lighting, viewed from behind, streetlights reflecting on wet pavement."
    }
  ],
  "video_prompts": [
    {
      "segment": 1,
      "prompt": "The camera tracks behind a woman in a red dress as she begins running across a rainy street at night, cinematic lighting, streetlights reflecting on wet pavement, rendered in a realistic style.",
      "first_frame": "provided_start_image.png",
      "last_frame": "segment_01.png"
    },
    {
      "segment": 2,
      "prompt": "The camera continues tracking behind the woman in a red dress as she completes her run across the rainy street at night, cinematic lighting, streetlights reflecting on wet pavement, rendered in a realistic style.",
      "first_frame": "segment_01.png",
      "last_frame": "segment_02.png"
    }
  ]
}

Respond ONLY with the JSON response formatted as above, with NO wrapper or commentary.
"""

class PromptEnhancer:
    """Uses OpenAI to enhance prompts with structured output validation via instructor"""

    def __init__(self, api_key: str, base_url: str, model: str):
        # Initialize OpenAI client
        from openai import OpenAI
        base_client = OpenAI(api_key=api_key, base_url=base_url)
        # Wrap OpenAI client for structured JSON outputs
        self.client = from_openai(base_client, mode=Mode.TOOLS_STRICT)
        self.model = model

    def enhance(self, instructions: str, prompt: str, max_tokens: int = 16384) -> Dict:
        """Enhance a prompt using OpenAI and return structured output"""
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "developer", "content": instructions.strip()},
                    {"role": "user", "content": prompt.strip()}
                ],
                response_model=PromptEnhancementResult,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            logging.exception("OpenAI prompt enhancement failed")
            raise

        return result.dict()

# ============================================================================
# Pipeline Components
# ============================================================================

def run_command(cmd: List[str], cwd: str = None) -> str:
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(cmd)}")
        logging.error(f"Error: {e.stderr}")
        raise

def generate_keyframes(
    keyframe_prompts: List[Union[str, Dict]],
    config: Dict,
    output_dir: str,
    model_name: str = None,
    imageRouter_api_key: str = None,
    stability_api_key: str = None,
    openai_api_key: str = None,
    gemini_api_key: str = None,
    initial_image_path: str = None,
    image_size: str = None,
    reference_images_dir: Dict[str, str] = None,
    max_retries: int = 3,
) -> List[str]:
    """Generate keyframes for the video pipeline"""
    # Import the keyframe generator module
    from keyframe_generator import generate_keyframes_from_json

    # Create frames directory inside output_dir if it doesn't exist
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Create a temporary JSON file with the keyframe prompts
    temp_json_path = os.path.join(output_dir, "keyframe_prompts.json")
    with open(temp_json_path, 'w') as f:
        import json
        # Ensure keyframe_prompts is in the correct format
        if keyframe_prompts and isinstance(keyframe_prompts[0], str):
            # Convert list of strings to list of dicts
            formatted_prompts = [
                {"segment": i+1, "prompt": prompt}
                for i, prompt in enumerate(keyframe_prompts)
            ]
        else:
            formatted_prompts = keyframe_prompts
        json.dump({"keyframe_prompts": formatted_prompts}, f, indent=2)

    logging.info(f"Starting sequential keyframe generation with {len(keyframe_prompts)} frames")

    # Generate keyframes sequentially to maintain character consistency
    return generate_keyframes_from_json(
        json_file=temp_json_path,
        output_dir=frames_dir,  # Use dedicated frames directory
        model_name=model_name,
        imageRouter_api_key=imageRouter_api_key,
        stability_api_key=stability_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        initial_image_path=initial_image_path,
        image_size=image_size,
        reference_images_dir=reference_images_dir,
        max_retries=max_retries
    )

def stitch_video_segments(video_paths: List[str], output_file: str) -> Optional[str]:
    """Stitch together video segments into a final video using ffmpeg"""
    if not video_paths:
        logging.warning("No video paths provided for stitching")
        return None

    # Create a temporary file listing the segments
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
        list_file = f.name

    # Use ffmpeg to concatenate the segments
    cmd = ["ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file, "-y"]
    run_command(cmd)

    # Clean up the temporary file
    os.unlink(list_file)
    return output_file

def generate_single_video_segment(
    wan2_dir: str,
    config: Dict,
    prompt_item: Dict,
    output_dir: str,
    flf2v_model_dir: str,
    frame_num: int = 81,
    gpu_ids: List[int] = None,
    single_keyframe_mode: bool = False
) -> str:
    """Generate a single video segment using the FLF2V model

    Args:
        wan2_dir: Path to the Wan2.1 base directory
        config: Configuration dictionary
        prompt_item: Video prompt item with segment and prompt text
        output_dir: Directory to save output videos
        flf2v_model_dir: Path to the FLF2V model directory
        frame_num: Number of frames to generate
        gpu_ids: List of GPU IDs to use for this segment generation
        single_keyframe_mode: If True, use single keyframe for I2V generation (for Veo3)

    Returns:
        Path to the generated video file
    """
    seg, prompt_text = prompt_item["segment"], prompt_item["prompt"]

    # Configure GPU usage for this process
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logging.info(f"Segment {seg} using GPUs: {gpu_ids}")
        print(f"{Colors.BOLD}{Colors.YELLOW}Segment {seg}{Colors.RESET} using GPUs: {Colors.CYAN}{gpu_ids}{Colors.RESET}")

    # ALWAYS use this exact directory structure - no exceptions
    base_dir = os.getcwd()
    frames_dir = os.path.join(base_dir, "output", "frames")
    videos_dir = os.path.join(base_dir, "output", "videos")

    # Create fresh directories if needed (safe in multiprocessing)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Display segment information
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Generating video for segment {seg}:{Colors.RESET}")
    print(f"{Colors.CYAN}{prompt_text[:100]}...{Colors.RESET}")

    # For the first segment, use the initial_image if provided
    if seg == 1 and config.get("initial_image"):
        initial_image = config.get("initial_image")
        # Always use absolute paths for consistency
        if not os.path.isabs(initial_image):
            first_file = os.path.abspath(os.path.join(os.getcwd(), initial_image))
        else:
            first_file = os.path.abspath(initial_image)

        # Log the exact path and check if it exists
        logging.info(f"Using initial image: {first_file}")
        if not os.path.exists(first_file):
            logging.error(f"Initial image not found at: {first_file}")
            logging.error(f"Current directory: {os.getcwd()}")
            logging.error(f"Directory contents: {os.listdir(os.path.dirname(first_file) if os.path.dirname(first_file) else './')}")
            raise FileNotFoundError(f"Initial image not found: {first_file}")
    else:
        # For subsequent segments, use previous keyframe
        first_file = os.path.join(frames_dir, f"segment_{seg-1:02d}.png")
        if not os.path.exists(first_file):
            logging.error(f"Previous keyframe not found: {first_file}")
            logging.error(f"Directory contents: {os.listdir(frames_dir)}")
            raise FileNotFoundError(f"Previous frame not found: {first_file}")

    # Path for this segment's keyframe
    last_file = os.path.join(frames_dir, f"segment_{seg:02d}.png")
    if not os.path.exists(last_file):
        logging.error(f"Keyframe not found: {last_file}")
        logging.error(f"Directory contents: {os.listdir(frames_dir)}")
        raise FileNotFoundError(f"Last frame not found at {last_file}")

    # Output video file path
    video_file = os.path.join(videos_dir, f"segment_{seg:02d}.mp4")

    logging.info(f"First frame path (must exist): {os.path.abspath(first_file)}")
    logging.info(f"Last frame path (must exist): {os.path.abspath(last_file)}")

    if not os.path.exists(first_file):
        raise FileNotFoundError(f"First frame not found at {first_file}")
    if not os.path.exists(last_file):
        raise FileNotFoundError(f"Last frame not found at {last_file}")

    logging.info(f"Generating video segment {seg}: {prompt_text}")

    # When running in parallel mode, the gpu_count is determined by the number of GPUs assigned to this process
    # Otherwise calculate it from total_gpus and parallel_segments
    if gpu_ids:
        gpu_count = len(gpu_ids)
    else:
        total_gpus = config.get("total_gpus", 1)
        parallel_segments = config.get("parallel_segments", 1)
        gpu_count = max(1, total_gpus // max(1, parallel_segments))

    logging.info(f"Using {gpu_count} GPUs for video segment {seg}")

    # Base command with common parameters
    base_cmd = [
        "--task", "flf2v-14B",
        "--size", config.get("size", "1280*720"),
        "--ckpt_dir", flf2v_model_dir,
        "--first_frame", first_file,
        "--last_frame", last_file,
        "--frame_num", str(frame_num),
        "--prompt", prompt_text,
        "--save_file", video_file,
        "--sample_guide_scale", str(config.get("guide_scale", 5.0)),
        "--sample_steps", str(config.get("sample_steps", 40)),
        "--sample_shift", str(config.get("sample_shift", 5.0))
    ]

    # Choose between distributed or non-distributed execution
    if gpu_count > 1:
        # Distributed execution with torchrun
        # Use different ports for different segment processes to avoid conflicts
        port = 29500 + seg  # Use different ports based on segment number
        logging.info(f"Using distributed execution with torchrun for segment {seg} (port {port})")
        cmd = ["torchrun", f"--nproc_per_node={gpu_count}", f"--rdzv_endpoint=localhost:{port}", "generate.py"]
        # Add FSDP flags for distributed training
        cmd.extend(base_cmd)
        cmd.extend(["--dit_fsdp", "--t5_fsdp"])
    else:
        # Non-distributed execution - direct Python call
        logging.info("Using non-distributed execution")
        cmd = ["python", "generate.py"]
        cmd.extend(base_cmd)

    # Run the command
    run_command(cmd, cwd=wan2_dir)
    logging.info(f"Saved video segment {seg} to {video_file}")
    print(f"{Colors.BOLD}{Colors.GREEN}Segment {seg} completed{Colors.RESET}: Saved to {Colors.UNDERLINE}{video_file}{Colors.RESET}")

    # Return the path to the generated video file
    return video_file

def generate_video_segments(
    wan2_dir: str,
    config: Dict,
    video_prompts: List[Dict],
    output_dir: str,
    flf2v_model_dir: str,
    frame_num: int = 81,
    single_keyframe_mode: bool = False
) -> List[str]:
    """Generate video segments using the FLF2V model

    This function can run in parallel if configured in the config file.
    """
    # Use consistent directory structure with other functions
    frames_dir = os.path.join(output_dir, "frames")
    videos_dir = os.path.join(output_dir, "videos")

    # Create fresh directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # List available frame files for debugging
    frame_files = os.listdir(frames_dir) if os.path.exists(frames_dir) else []
    logging.info(f"SIMPLE PATHS - Video generation:")
    logging.info(f"Available frame files in {frames_dir}: {frame_files}")
    logging.info(f"Videos will be saved to: {videos_dir}")

    # Get GPU configuration
    total_gpus = config.get("total_gpus", 1)
    parallel_segments = config.get("parallel_segments", 1)

    # Calculate GPUs per segment based on available GPUs and parallelism
    if parallel_segments <= 0:
        parallel_segments = 1
        logging.warning(f"Invalid parallel_segments value, adjusted to 1")

    # Calculate GPUs per segment, ensure at least 1 GPU per segment
    gpus_per_segment = max(1, total_gpus // parallel_segments)

    # Adjust parallel_segments if it would result in too many segments for available GPUs
    if parallel_segments > total_gpus:
        old_parallel = parallel_segments
        parallel_segments = total_gpus
        logging.warning(f"Requested {old_parallel} parallel segments, but only {total_gpus} GPUs available")
        logging.warning(f"Adjusted to {parallel_segments} parallel segments with 1 GPU each")

    logging.info(f"GPU configuration: {total_gpus} total GPUs, {parallel_segments} parallel segments, {gpus_per_segment} GPUs per segment")

    # Check if we're in single-keyframe mode (for Veo3)
    if single_keyframe_mode:
        logging.info("Using single-keyframe mode for video generation")
        return generate_video_segments_single_keyframe(config, video_prompts, output_dir)

    # Skip parallelization if only one segment or parallel_segments=1
    if len(video_prompts) <= 1 or parallel_segments <= 1:
        logging.info("Running video generation sequentially")
        return generate_video_segments_sequential(wan2_dir, config, video_prompts, output_dir, flf2v_model_dir, frame_num)

    # Display parallelization strategy
    logging.info(f"Generating {len(video_prompts)} segments with parallelism strategy:")
    logging.info(f"  - {parallel_segments} segments in parallel")
    logging.info(f"  - {gpus_per_segment} GPUs per segment")
    logging.info(f"  - {total_gpus} total GPUs available")

    print(f"\n{Colors.BOLD}{Colors.PURPLE}Parallel Video Generation:{Colors.RESET}")
    print(f"{Colors.CYAN}Running {parallel_segments} segments in parallel, each using {gpus_per_segment} GPUs{Colors.RESET}")

    # Import multiprocessing here to avoid issues with recursive imports
    import multiprocessing as mp
    from functools import partial

    # Allocate GPUs to each segment
    all_gpu_assignments = []
    for i in range(0, len(video_prompts), parallel_segments):
        batch = video_prompts[i:i+parallel_segments]
        gpu_assignments = []

        for j, _ in enumerate(batch):
            # Determine which GPUs to use for this segment
            start_gpu = j * gpus_per_segment % total_gpus
            gpu_ids = [(start_gpu + k) % total_gpus for k in range(gpus_per_segment)]
            gpu_assignments.append(gpu_ids)

        all_gpu_assignments.extend(gpu_assignments)

    # Create a partial function with the common arguments
    gen_segment_partial = partial(
        process_segment,
        wan2_dir=wan2_dir,
        config=config,
        output_dir=output_dir,
        flf2v_model_dir=flf2v_model_dir,
        frame_num=frame_num
    )

    # Prepare the arguments for each job
    job_args = [(prompt, gpu_ids) for prompt, gpu_ids in zip(video_prompts, all_gpu_assignments[:len(video_prompts)])]

    # Run jobs in parallel
    with mp.Pool(processes=parallel_segments) as pool:
        try:
            # Map will divide work evenly among processes
            results = pool.starmap(gen_segment_partial, job_args)
            video_paths = [path for path in results if path]
        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            logging.info("Falling back to sequential processing")
            if single_keyframe_mode:
                return generate_video_segments_single_keyframe(config, video_prompts, output_dir)
            else:
                return generate_video_segments_sequential(wan2_dir, config, video_prompts, output_dir, flf2v_model_dir, frame_num)

    # Sort video paths by segment number to ensure correct order
    video_paths.sort()
    return video_paths


def process_segment(prompt_item, gpu_ids, wan2_dir, config, output_dir, flf2v_model_dir, frame_num):
    """Process a single segment in a separate process"""
    try:
        return generate_single_video_segment(
            wan2_dir=wan2_dir,
            config=config,
            prompt_item=prompt_item,
            output_dir=output_dir,
            flf2v_model_dir=flf2v_model_dir,
            frame_num=frame_num,
            gpu_ids=gpu_ids
        )
    except Exception as e:
        seg = prompt_item.get("segment", "unknown")
        logging.error(f"Error processing segment {seg}: {e}")
        return None


def generate_video_segments_single_keyframe(
    config: Dict,
    video_prompts: List[Dict],
    output_dir: str
) -> List[str]:
    """
    Generate video segments using single keyframe for I2V generation (e.g., Veo3).

    Args:
        config: Configuration dictionary
        video_prompts: List of video prompts with keyframe paths
        output_dir: Directory to save generated videos

    Returns:
        List of paths to generated video files
    """
    import generators.factory as factory
    from generators.exceptions import InvalidInputError, GenerationError

    video_paths = []

    # Get the video generation backend
    backend = config.get('default_video_generation_backend', 'veo3')
    logging.info(f"Using {backend} for single-keyframe video generation")

    # Get segment duration
    segment_duration = config.get('segment_duration_seconds', 5.0)

    # Get I2I mode configuration
    i2i_config = config.get('i2i_mode', {})
    keyframe_position = i2i_config.get('keyframe_position', 'first')

    # Use consistent directory structure with other functions
    frames_dir = os.path.join(output_dir, "frames")
    videos_dir = os.path.join(output_dir, "videos")

    # Create directories if they don't exist
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Ensure we're using absolute paths for keyframe files
    frames_dir_abs = os.path.abspath(frames_dir)
    logging.info(f"Absolute frames directory path for single-keyframe mode: {frames_dir_abs}")

    for prompt_item in video_prompts:
        seg, prompt_text = prompt_item["segment"], prompt_item["prompt"]

        # Select the keyframe based on position setting
        if keyframe_position == 'first' and 'first_frame' in prompt_item:
            keyframe_path = prompt_item['first_frame']
        elif keyframe_position == 'last' and 'last_frame' in prompt_item:
            keyframe_path = prompt_item['last_frame']
        elif keyframe_position == 'middle':
            # For middle, use first frame if available (could be enhanced later)
            keyframe_path = prompt_item.get('first_frame', prompt_item.get('last_frame'))
        else:
            # Default to first frame
            keyframe_path = prompt_item.get('first_frame', prompt_item.get('last_frame'))

        # Fix keyframe path to use absolute path
        if keyframe_path:
            if keyframe_path == 'provided_start_image.png':
                # This should be segment_00.png
                keyframe_path = os.path.join(frames_dir_abs, 'segment_00.png')
            elif keyframe_path.startswith('segment_') and keyframe_path.endswith('.png'):
                # Convert relative segment reference to absolute path
                keyframe_path = os.path.join(frames_dir_abs, keyframe_path)
            else:
                # For any other reference, try to resolve it
                keyframe_path = os.path.join(frames_dir_abs, keyframe_path)

        if not keyframe_path:
            logging.error(f"No keyframe found for segment {seg}")
            continue

        logging.info(f"Generating video for segment {seg} using keyframe: {keyframe_path}")

        try:
            # Create video generator
            generator = factory.create_video_generator(backend, config)

            # Generate video from single keyframe
            video_file = os.path.join(output_dir, f"segment_{seg:03d}.mp4")

            # Call generator's generate_video method
            generator.generate_video(
                prompt=prompt_text,
                input_image_path=keyframe_path,
                output_path=video_file,
                duration=segment_duration
            )

            video_paths.append(video_file)
            logging.info(f"Generated video segment {seg}: {video_file}")

        except (InvalidInputError, GenerationError) as e:
            logging.error(f"Failed to generate video for segment {seg}: {e}")

            # Try fallback if configured
            fallback_generator = factory.get_fallback_generator(config, backend)
            if fallback_generator:
                try:
                    logging.info(f"Trying fallback generator for segment {seg}")
                    fallback_generator.generate_video(
                        prompt=prompt_text,
                        input_image_path=keyframe_path,
                        output_path=video_file,
                        duration=segment_duration
                    )
                    video_paths.append(video_file)
                    logging.info(f"Fallback succeeded for segment {seg}")
                except Exception as fallback_error:
                    logging.error(f"Fallback also failed for segment {seg}: {fallback_error}")
            else:
                logging.error(f"No fallback available for segment {seg}")

        except Exception as e:
            logging.error(f"Unexpected error generating video for segment {seg}: {e}")

    return video_paths

def generate_video_segments_sequential(
    wan2_dir: str,
    config: Dict,
    video_prompts: List[Dict],
    output_dir: str,
    flf2v_model_dir: str,
    frame_num: int = 81,
) -> List[str]:
    """Generate video segments sequentially using the FLF2V model"""
    video_paths = []

    # Use consistent directory structure with other functions
    frames_dir = os.path.join(output_dir, "frames")
    videos_dir = os.path.join(output_dir, "videos")

    for item in video_prompts:
        seg, prompt_text = item["segment"], item["prompt"]

        # Determine the first frame for this segment
        if seg == 1:
            # For the first segment, check if initial_image is provided
            if config.get("initial_image"):
                initial_image = config.get("initial_image")
                # Always use absolute paths for consistency
                if not os.path.isabs(initial_image):
                    first_file = os.path.abspath(os.path.join(os.getcwd(), initial_image))
                else:
                    first_file = os.path.abspath(initial_image)

                # Log the exact path and check if it exists
                logging.info(f"Using initial image: {first_file}")
                if not os.path.exists(first_file):
                    logging.error(f"Initial image not found at: {first_file}")
                    logging.error(f"Current directory: {os.getcwd()}")
                    logging.error(f"Directory contents: {os.listdir(os.path.dirname(first_file) if os.path.dirname(first_file) else './')}")
                    raise FileNotFoundError(f"Initial image not found: {first_file}")
            else:
                # For segment 1 without initial_image, the first keyframe IS the first frame
                first_file = os.path.join(frames_dir, f"segment_{seg:02d}.png")
                logging.info(f"No initial image provided for segment 1, using first keyframe as both first and last frame")
        else:
            # For subsequent segments, use previous keyframe
            first_file = os.path.join(frames_dir, f"segment_{seg-1:02d}.png")
            if not os.path.exists(first_file):
                logging.error(f"Previous keyframe not found: {first_file}")
                logging.error(f"Directory contents: {os.listdir(frames_dir)}")
                raise FileNotFoundError(f"Previous frame not found: {first_file}")

        # Path for this segment's keyframe
        last_file = os.path.join(frames_dir, f"segment_{seg:02d}.png")
        if not os.path.exists(last_file):
            logging.error(f"Keyframe not found: {last_file}")
            logging.error(f"Directory contents: {os.listdir(frames_dir)}")
            raise FileNotFoundError(f"Last frame not found at {last_file}")

        # Output video file path
        video_file = os.path.join(videos_dir, f"segment_{seg:02d}.mp4")

        logging.info(f"First frame path (must exist): {os.path.abspath(first_file)}")
        logging.info(f"Last frame path (must exist): {os.path.abspath(last_file)}")

        if not os.path.exists(first_file):
            raise FileNotFoundError(f"First frame not found at {first_file}")
        if not os.path.exists(last_file):
            raise FileNotFoundError(f"Last frame not found at {last_file}")

        logging.info(f"Generating video segment {seg}: {prompt_text}")

        # Get GPU count from config or use default of 8
        gpu_count = config.get("gpu_count", 8)
        logging.info(f"Using {gpu_count} GPUs for video generation")

        # Base command with common parameters
        base_cmd = [
            "--task", "flf2v-14B",
            "--size", config.get("size", "1280*720"),
            "--ckpt_dir", flf2v_model_dir,
            "--first_frame", first_file,
            "--last_frame", last_file,
            "--frame_num", str(frame_num),
            "--prompt", prompt_text,
            "--save_file", video_file,
            "--sample_guide_scale", str(config.get("guide_scale", 5.0)),
            "--sample_steps", str(config.get("sample_steps", 40)),
            "--sample_shift", str(config.get("sample_shift", 5.0))
        ]

        # Choose between distributed or non-distributed execution
        if gpu_count > 1:
            # Distributed execution with torchrun
            # Use different ports for different segment processes to avoid conflicts
            port = 29500 + int(seg)  # Use different ports based on segment number
            logging.info(f"Using distributed execution with torchrun for segment {seg} (port {port})")
            cmd = ["torchrun", f"--nproc_per_node={gpu_count}", f"--rdzv_endpoint=localhost:{port}", "generate.py"]
            # Add FSDP flags for distributed training
            cmd.extend(base_cmd)
            cmd.extend(["--dit_fsdp", "--t5_fsdp"])
        else:
            # Non-distributed execution - direct Python call
            logging.info("Using non-distributed execution")
            cmd = ["python", "generate.py"]
            cmd.extend(base_cmd)

        # Run the command
        run_command(cmd, cwd=wan2_dir)
        video_paths.append(video_file)
        logging.info(f"Saved video to {video_file}")

    return video_paths

def concatenate_videos(segment_paths: List[str], output_file: str) -> None:
    """Concatenate multiple video segments into a single output video"""
    logging.info(f"Concatenating {len(segment_paths)} segments into {output_file}")

    # Create a temporary file listing the segments
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
        list_file = f.name

    # Use ffmpeg to concatenate the segments
    cmd = ["ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file]
    run_command(cmd)

    # Clean up the temporary file
    os.unlink(list_file)

# ============================================================================
# Prompt Enhancement Function
# ============================================================================

def enhance_prompt(prompt: str, config: Dict, output_dir: str) -> Dict:
    """Enhance the input prompt using OpenAI API with colorful output"""
    # Display original prompt in green
    print(f"\n{Colors.BOLD}{Colors.GREEN}Original Prompt:{Colors.RESET}")
    print(f"{Colors.GREEN}{prompt}{Colors.RESET}\n")

    logging.info("Enhancing input prompt...")

    # Extract configuration
    api_key = config.get("openai_api_key")
    base_url = config.get("openai_base_url")
    model = config.get("prompt_enhancement_model", "gpt-4o-mini")
    default_backend = config.get("default_backend", "wan2.1")

    if not api_key:
        logging.warning("No OpenAI API key provided, skipping prompt enhancement")
        return {"keyframe_prompts": [], "video_prompts": []}

    # Check if using Minimax or Veo3 backend to add character limit constraint
    backend_constraint = ""
    if default_backend.lower() == "minimax":
        backend_constraint = "\n\nIMPORTANT: When generating video prompts for Minimax backend, each video prompt must be 500 characters or less. Keep descriptions concise and focused while maintaining essential visual and action details."
    elif default_backend.lower() == "veo3":
        backend_constraint = "\n\nIMPORTANT: When generating video prompts for Veo3 backend, each video prompt must be 1000 characters or less. Keep descriptions concise and focused while maintaining essential visual and action details."

    try:
        # Set up the OpenAI client
        from openai import OpenAI
        from instructor import from_openai, Mode

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        instructor_client = from_openai(client, mode=Mode.TOOLS)

        # Define the messages with potential backend constraint
        messages = [
            {"role": "system", "content": PROMPT_ENHANCEMENT_INSTRUCTIONS + backend_constraint},
            {"role": "user", "content": prompt}
        ]

        # Get enhanced prompts
        logging.info(f"Making request to OpenAI API with model: {model}")

        # Basic parameters that all models support
        completion_params = {
            "model": model,
            "response_model": PromptEnhancementResult,
            "messages": messages
        }

        # Only add temperature if not using o4-mini or gpt-5 (which don't support custom temperature)
        if not (model.startswith("o4-mini") or "gpt-5" in model):
            completion_params["temperature"] = 0.7
            completion_params["seed"] = 42
            logging.info(f"Using temperature=0.7 and seed=42 with model {model}")
        else:
            logging.info(f"Using default parameters for model {model} (no temperature/seed)")

        # Make the API call with appropriate parameters
        result = instructor_client.chat.completions.create(**completion_params)

        # Convert to dict format
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result.dict()

        # The response is already in the right structure, add debugging output
        print(f"DEBUG - Response keys: {result_dict.keys()}")

        # Create a copy of the result to modify as needed
        enhanced_data = result_dict

        # Display the enhanced segmented prompts in colors
        print(f"\n{Colors.BOLD}{Colors.PURPLE}Enhanced Prompt Segments:{Colors.RESET}")

        # Get the segment count from the segmentation logic
        num_segments = result_dict["segmentation_logic"]["number_of_segments"]
        print(f"DEBUG - Number of segments: {num_segments}")

        # Create formatted output for each segment
        for i in range(len(result_dict["keyframe_prompts"])):
            keyframe_info = result_dict["keyframe_prompts"][i]
            video_info = result_dict["video_prompts"][i]

            segment_num = keyframe_info["segment"]
            keyframe_prompt = keyframe_info["prompt"]
            video_prompt = video_info["prompt"]

            # Display segment number in yellow/bold
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Segment {segment_num}:{Colors.RESET}")
            # Display keyframe prompt in blue with label
            print(f"{Colors.BOLD}Keyframe:{Colors.RESET} {Colors.BLUE}{keyframe_prompt}{Colors.RESET}")
            # Display video prompt in cyan with label
            print(f"\n{Colors.BOLD}Video Clip:{Colors.RESET} {Colors.CYAN}{video_prompt}{Colors.RESET}")

        print("\n") # Add spacing for readability
        logging.info(f"Enhanced prompt into {len(enhanced_data['keyframe_prompts'])} segments")

        # Save the enhanced prompt to the output directory
        json_path = os.path.join(output_dir, "enhanced_prompt.json")
        with open(json_path, "w") as f:
            json.dump(enhanced_data, f, indent=2)

        logging.info(f"Saved enhanced prompt data to {json_path}")
        return enhanced_data

    except Exception as e:
        logging.error(f"Error enhancing prompt: {e}")
        raise

# ============================================================================
# Chaining Mode Functions
# ============================================================================

def generate_video_chaining_mode(
    config: Dict,
    video_prompts: List[Dict],
    output_dir: str,  # Absolute path, prepared by run_pipeline
    segment_duration: float,
) -> List[str]:
    """Generate video segments sequentially using the I2V model in chaining mode"""
    video_paths = []

    # Determine backend and initialize generator
    primary_backend_name = config.get("default_backend", "wan2.1")
    if not primary_backend_name:
        logging.error("No 'default_backend' specified in configuration.")
        raise ValueError("Missing 'default_backend' in configuration.")

    attempted_backends_global = set() # Keep track of backends attempted across all segments for primary generator selection
    current_generator = None

    try:
        logging.info(f"Attempting to initialize primary backend: {primary_backend_name}")
        current_generator = create_video_generator(primary_backend_name, config)
        attempted_backends_global.add(primary_backend_name)
    except VideoGenerationError as e:
        logging.error(f"Failed to initialize primary backend {primary_backend_name}: {e}")
        fallback_generator_init = get_fallback_generator(primary_backend_name, config, attempted_backends_global)
        if fallback_generator_init:
            current_generator = fallback_generator_init
            primary_backend_name = current_generator.get_backend_name()
            logging.info(f"Successfully initialized fallback generator: {primary_backend_name}")
            attempted_backends_global.add(primary_backend_name)
        else:
            logging.error("No available video generation backends could be initialized.")
            raise

    if not current_generator:
        logging.error("Failed to initialize any video generator.")
        raise VideoGenerationError("Could not initialize any video generator.")

    # Use the absolute output_dir passed from run_pipeline
    frames_dir = os.path.join(output_dir, "frames")
    videos_dir = os.path.join(output_dir, "videos")
    extracted_frames_dir = os.path.join(output_dir, "extracted_frames")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(extracted_frames_dir, exist_ok=True)

    for idx, item in enumerate(video_prompts):
        seg, prompt_text = item["segment"], item["prompt"]

        logging.info(f"Processing segment {seg}/{len(video_prompts)} in chaining mode")

        # Determine input image for this segment
        if seg == 1:
            # For the first segment, use the initial_image if provided
            if config.get("initial_image"):
                input_image = config.get("initial_image")
                # Always use absolute paths for consistency
                if not os.path.isabs(input_image):
                    first_file = os.path.abspath(os.path.join(os.getcwd(), input_image))
                else:
                    first_file = os.path.abspath(input_image)

                logging.info(f"Using initial image for first segment: {input_image}")
                if not os.path.exists(input_image):
                    logging.error(f"Initial image not found at: {input_image}")
                    raise FileNotFoundError(f"Initial image not found: {input_image}")
            else:
                # No initial image provided, need to generate one
                from keyframe_generator import generate_keyframe

                # Create a starting frame using the first segment's prompt
                logging.info("No initial image provided, generating one for first segment")
                input_image = os.path.join(frames_dir, "segment_00.png")

                # If we already generated this frame during keyframe generation, use it
                if os.path.exists(input_image):
                    logging.info(f"Using existing segment_00.png: {input_image}")
                else:
                    # Generate initial frame using available API
                    initial_frame_prompt = f"First frame establishing shot: {prompt_text}"

                    # Check which model to use based on image_generation_model parameter
                    image_generation_model = config.get("image_generation_model", "stabilityai/sd3:stable")

                    if "openai" in image_generation_model.lower():
                        # Use OpenAI for initial frame
                        if not config.get("openai_api_key"):
                            logging.error(f"Selected model {image_generation_model} but no OpenAI API key provided")
                            raise ValueError("OpenAI API key required for selected model")

                        logging.info(f"Generating initial frame with OpenAI {image_generation_model}")
                        from keyframe_generator import generate_keyframe_with_openai
                        generate_keyframe_with_openai(
                            prompt=initial_frame_prompt,
                            output_path=input_image,
                            openai_api_key=config.get("openai_api_key"),
                            size=config.get("image_size", "1536x1024")
                        )
                    elif "stability" in image_generation_model.lower():
                        # Use Stability AI for initial frame
                        if not config.get("stability_api_key"):
                            logging.error(f"Selected model {image_generation_model} but no Stability AI API key provided")
                            raise ValueError("Stability AI API key required for selected model")

                        logging.info(f"Generating initial frame with Stability AI {image_generation_model}")
                        from keyframe_generator import generate_keyframe_with_stability
                        generate_keyframe_with_stability(
                            prompt=initial_frame_prompt,
                            output_path=input_image,
                            stability_api_key=config.get("stability_api_key")
                        )
                    else:
                        # Try ImageRouter
                        if config.get("image_router_api_key"):
                            logging.info(f"Generating initial frame with ImageRouter using model {image_generation_model}")
                            from keyframe_generator import generate_keyframe_with_imageRouter
                            generate_keyframe_with_imageRouter(
                                prompt=initial_frame_prompt,
                                output_path=input_image,
                                model_name=image_generation_model,
                                imageRouter_api_key=config.get("image_router_api_key")
                            )
                        else:
                            logging.error(f"No API key provided for model {image_generation_model}")
                            raise ValueError("Initial image required or appropriate API key for generation")

                    logging.info(f"Generated initial frame at {input_image}")
        else:
            # For subsequent segments, use the last frame from the previous segment
            prev_segment_video = video_paths[-1]  # Most recently added video path

            # Extract the last frame from the previous segment
            input_image = os.path.join(extracted_frames_dir, f"segment_{seg-1:02d}_last_frame.png")

            # Extract the last frame from the previous video segment
            extracted_frame = extract_last_frame(prev_segment_video, input_image)

            if not extracted_frame or not os.path.exists(extracted_frame):
                logging.error(f"Failed to extract frame from previous segment: {prev_segment_video}")
                raise FileNotFoundError(f"Could not extract frame from {prev_segment_video}")

            logging.info(f"Using extracted frame from previous segment: {input_image}")

        # Ensure input_image is an absolute path
        if not os.path.isabs(input_image):
            input_image = os.path.abspath(input_image)

        video_file_output_path = os.path.join(videos_dir, f"segment_{seg:02d}.mp4")
        if not os.path.isabs(video_file_output_path):
            video_file_output_path = os.path.abspath(video_file_output_path)

        segment_generated_successfully = False
        # Attempt to generate the segment with the current generator, then fallbacks if needed
        attempt_generator = current_generator
        attempted_backends_segment = {current_generator.get_backend_name()} # Track attempted backends for this segment

        while attempt_generator:
            try:
                logging.info(f"Attempting segment {seg} with {attempt_generator.get_backend_name()}...")

                validation_errors = attempt_generator.validate_inputs(
                    prompt=prompt_text,
                    input_image_path=input_image,
                    duration=segment_duration
                )
                if validation_errors:
                    logging.error(f"Input validation failed for segment {seg} with {attempt_generator.get_backend_name()}: {validation_errors}")
                    raise InvalidInputError(f"Validation failed: {'; '.join(validation_errors)}")

                generated_video_path = attempt_generator.generate_video(
                    prompt=prompt_text,
                    input_image_path=input_image,
                    output_path=video_file_output_path,
                    duration=segment_duration,
                    frame_num=config.get("frame_num", 81) # Pass frame_num from main config if available
                )

                if not generated_video_path or not os.path.exists(generated_video_path):
                    raise VideoGenerationError(f"Generator {attempt_generator.get_backend_name()} reported success but video file not found: {generated_video_path}")

                logging.info(f"{Colors.GREEN}Segment {seg} successfully generated by {attempt_generator.get_backend_name()} at {generated_video_path}{Colors.RESET}")
                video_paths.append(generated_video_path)
                segment_generated_successfully = True
                current_generator = attempt_generator # Update main generator to this successful one for next segment
                break # Exit while loop for this segment, segment success

            except (VideoGenerationError, APIError, GenerationTimeoutError, InvalidInputError, QuotaExceededError) as e:
                logging.error(f"Error generating segment {seg} with {attempt_generator.get_backend_name()}: {e}")
                logging.info(f"Trying to find a fallback generator. Attempted for this segment: {attempted_backends_segment}")
                # Pass attempted_backends_global to influence future fallback choices if primary keeps failing
                # Also pass attempted_backends_segment to ensure we don't retry a backend that just failed for this specific segment
                combined_attempted_backends = attempted_backends_segment.union(attempted_backends_global)
                fallback_generator = get_fallback_generator(attempt_generator.get_backend_name(), config, combined_attempted_backends)

                if fallback_generator:
                    logging.info(f"Switching to fallback generator: {fallback_generator.get_backend_name()} for segment {seg}")
                    attempt_generator = fallback_generator
                    attempted_backends_segment.add(fallback_generator.get_backend_name())
                    # attempted_backends_global is updated by get_fallback_generator if a new one is chosen
                else:
                    logging.error(f"No more fallback generators available for segment {seg} after {attempt_generator.get_backend_name()} failed.")
                    attempt_generator = None
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error generating segment {seg} with {attempt_generator.get_backend_name()}: {e}")
                logging.info(f"Trying to find a fallback generator. Attempted for this segment: {attempted_backends_segment}")
                combined_attempted_backends = attempted_backends_segment.union(attempted_backends_global)
                fallback_generator = get_fallback_generator(attempt_generator.get_backend_name(), config, combined_attempted_backends)
                if fallback_generator:
                    logging.info(f"Switching to fallback generator: {fallback_generator.get_backend_name()} for segment {seg}")
                    attempt_generator = fallback_generator
                    attempted_backends_segment.add(fallback_generator.get_backend_name())
                else:
                    logging.error(f"No more fallback generators available for segment {seg} after {attempt_generator.get_backend_name()} failed.")
                    attempt_generator = None

        if not segment_generated_successfully:
            logging.error(f"{Colors.RED}Failed to generate segment {seg} after trying all available backends.{Colors.RESET}")
            raise VideoGenerationError(f"Failed to generate segment {seg}. Aborting pipeline.")

    return video_paths

# ============================================================================
# Main Pipeline Function
# ============================================================================

def run_pipeline(config_path: str, prompt_override: str = None) -> None:
    """Run the end-to-end pipeline using configuration from YAML"""
    logging.info(f"Loading configuration from {config_path}")
    base_config = load_config(config_path)

    # Use ConfigMerger to handle prompt override with proper precedence
    config_merger = ConfigMerger()
    cli_args = {'prompt': prompt_override} if prompt_override else None
    
    # Build effective configuration with CLI prompt override
    config = config_merger.build_effective_config(
        base_config=base_config,
        cli_args=cli_args,
        http_overrides=None  # No HTTP overrides in CLI context
    )

    base_dir = os.getcwd()  # Current working directory
    output_dir = os.path.join(base_dir, "output")
    frames_dir = os.path.join(output_dir, "frames")
    videos_dir = os.path.join(output_dir, "videos")

    # Delete and recreate directories to ensure no nested subdirs
    import shutil
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    if os.path.exists(videos_dir):
        shutil.rmtree(videos_dir)

    # Create clean directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Log directory structure
    logging.info(f"Frames directory: {frames_dir}")
    logging.info(f"Videos directory: {videos_dir}")

    # Save a copy of the effective configuration (after merging)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Step 1: Enhance the input prompt
    raw_prompt = config.get("prompt")
    if not raw_prompt:
        raise ValueError("No prompt provided in configuration or command line")

    # Call the enhanced prompt function with colorful output
    enhanced_data = enhance_prompt(raw_prompt, config, output_dir)

    # Step 2: Generate keyframes
    logging.info("Generating keyframes...")
    image_generation_model = config.get('image_generation_model')
    # Get API keys from config
    openai_api_key = config.get('openai_api_key')
    stability_api_key = config.get('stability_api_key')
    imageRouter_api_key = config.get('image_router_token')
    gemini_api_key = config.get('gemini_api_key')

    # Get keyframe generation settings
    model_name = config.get('keyframe_prompt_model', 'dall-e-3')
    image_size = config.get('image_size', '1024x1024')

    # Get I2I mode settings (now consolidated into main image generation)
    reference_images_dir = config.get('reference_images_dir')
    auto_generate_initial = config.get('auto_generate_initial', True)
    keyframe_position = config.get('keyframe_position', 'first')

    # Adjust image size for Stability AI if needed
    if image_generation_model and "stability" in image_generation_model.lower():
        image_size = "1024x1024"
        logging.info(f"Using Stability AI image size: {image_size}")

    initial_image = config.get('initial_image')

    # Log which API will be used based on configured model
    if image_generation_model:  # Check if image_generation_model is not None
        if "openai" in image_generation_model.lower():
            if config.get('openai_api_key'):
                logging.info(f"Using OpenAI API ({image_generation_model}) for keyframe generation")
            else:
                logging.warning(f"Selected model {image_generation_model} but no OpenAI API key provided")
        elif "stability" in image_generation_model.lower():
            if stability_api_key:
                logging.info(f"Using Stability AI API ({image_generation_model}) for keyframe generation")
            else:
                logging.warning(f"Selected model {image_generation_model} but no Stability AI API key provided")
        elif imageRouter_api_key:
            logging.info(f"Using ImageRouter API with model {image_generation_model} for keyframe generation")
        else:
            logging.warning("No suitable API keys provided for the selected image generation model")
    else:
        logging.warning("No image generation model specified in configuration")

    # Check if we need to generate an initial frame (segment_00.png)
    if not initial_image:
        # Need to generate segment_00.png as our starting point
        logging.info("No initial image provided, generating segment_00.png first")
        print(f"\n{Colors.BOLD}{Colors.PURPLE}Generating Initial Frame (segment_00.png):{Colors.RESET}")

        # Create a prompt for the initial frame based on the first segment
        if len(enhanced_data['keyframe_prompts']) > 0:
            first_segment_prompt = enhanced_data['keyframe_prompts'][0]['prompt']
            # Create a starting frame prompt by modifying the first segment's prompt
            initial_frame_prompt = f"First frame establishing shot: {first_segment_prompt}"

            # Generate the initial frame using the prompt
            from keyframe_generator import generate_keyframe_with_openai, generate_keyframe_with_stability

            # Set up output path
            frames_dir = os.path.join(output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            initial_frame_path = os.path.join(frames_dir, "segment_00.png")

            # Generate using appropriate API based on configured model
            if "gemini" in image_generation_model.lower() and config.get('gemini_api_key'):
                logging.info(f"Generating initial frame with Gemini: {initial_frame_prompt}")
                try:
                    generate_keyframe_with_gemini(
                        prompt=initial_frame_prompt,
                        output_path=initial_frame_path,
                        gemini_api_key=config.get('gemini_api_key'),
                        model_name=image_generation_model,
                        max_retries=config.get('remote_api_settings', {}).get('max_retries', 3)
                    )
                    initial_image = initial_frame_path  # Use this as our initial image
                    logging.info(f"Generated initial frame at {initial_frame_path}")
                except Exception as e:
                    logging.error(f"Failed to generate initial frame with Gemini: {e}")
                    raise
            elif "openai" in image_generation_model.lower() and config.get('openai_api_key'):
                logging.info(f"Generating initial frame with OpenAI: {initial_frame_prompt}")
                try:
                    generate_keyframe_with_openai(
                        prompt=initial_frame_prompt,
                        output_path=initial_frame_path,
                        openai_api_key=config.get('openai_api_key'),
                        size=image_size
                    )
                    initial_image = initial_frame_path  # Use this as our initial image
                    logging.info(f"Generated initial frame at {initial_frame_path}")
                except Exception as e:
                    logging.error(f"Failed to generate initial frame with OpenAI: {e}")
                    raise
            elif "stability" in image_generation_model.lower() and stability_api_key:
                logging.info(f"Generating initial frame with Stability AI: {initial_frame_prompt}")
                try:
                    generate_keyframe_with_stability(
                        prompt=initial_frame_prompt,
                        output_path=initial_frame_path,
                        stability_api_key=stability_api_key
                    )
                    initial_image = initial_frame_path  # Use this as our initial image
                    logging.info(f"Generated initial frame at {initial_frame_path}")
                except Exception as e:
                    logging.error(f"Failed to generate initial frame with Stability AI: {e}")
                    raise
            else:
                logging.error("Unable to generate initial frame, no suitable API keys provided")
                raise ValueError("Cannot generate initial frame without OpenAI, Gemini, or Stability AI API keys")

    # Determine which generation mode we're using
    generation_mode = config.get('generation_mode', 'keyframe').lower()
    logging.info(f"Running in {generation_mode} mode")

    # Get Wan2.1 directory
    wan2_dir = config.get('wan2_dir', './Wan2.1')

    if generation_mode == 'keyframe':
        # ----- KEYFRAME MODE -----
        # Generate keyframes for first-last-frame to video generation
        logging.info(f"Starting keyframe generation for {len(enhanced_data['keyframe_prompts'])} segments")
        print(f"\n{Colors.BOLD}{Colors.PURPLE}Generating Keyframes:{Colors.RESET}")

        # Generate keyframes sequentially for character consistency
        keyframe_paths = generate_keyframes(
            config=config,
            keyframe_prompts=enhanced_data['keyframe_prompts'],
            output_dir=frames_dir,
            model_name=image_generation_model,
            imageRouter_api_key=imageRouter_api_key,
            stability_api_key=stability_api_key,
            openai_api_key=config.get('openai_api_key'),
            gemini_api_key=config.get('gemini_api_key'),
            initial_image_path=initial_image,
            image_size=image_size,
            reference_images_dir=reference_images_dir,
            max_retries=config.get('remote_api_settings', {}).get('max_retries', 3)
        )

        # Ensure segment_00.png exists before proceeding to video generation
        # This is critical for the first segment to work correctly
        segment_00_path = os.path.join(frames_dir, "segment_00.png")
        if not os.path.exists(segment_00_path) and len(keyframe_paths) > 0:
            logging.info(f"Creating segment_00.png as a copy of the first keyframe")
            import shutil
            try:
                # Copy the first segment's keyframe as segment_00.png
                shutil.copy2(keyframe_paths[0], segment_00_path)
                logging.info(f"Created segment_00.png by copying {keyframe_paths[0]}")
                # Verify the file exists after copy
                if os.path.exists(segment_00_path):
                    logging.info(f"Verified segment_00.png exists at {segment_00_path}")
                else:
                    logging.error(f"Failed to create segment_00.png after copy operation")
            except Exception as e:
                logging.error(f"Error creating segment_00.png: {e}")
                # Try an alternative approach - create a symbolic link
                try:
                    os.symlink(keyframe_paths[0], segment_00_path)
                    logging.info(f"Created symlink for segment_00.png pointing to {keyframe_paths[0]}")
                except Exception as e2:
                    logging.error(f"Error creating symlink: {e2}")

        # Double-check the frames directory contents before video generation
        logging.info(f"Frames directory contents before video generation: {os.listdir(frames_dir)}")

        # Fix video prompt paths to use absolute paths to actual keyframe files
        # The OpenAI prompt enhancement generates relative paths, but we need absolute paths
        frames_dir_abs = os.path.abspath(frames_dir)
        logging.info(f"Absolute frames directory path: {frames_dir_abs}")

        # Map relative keyframe references to actual generated files
        for i, prompt_item in enumerate(enhanced_data['video_prompts']):
            # Fix first_frame path
            if 'first_frame' in prompt_item:
                first_frame_ref = prompt_item['first_frame']
                if first_frame_ref == 'provided_start_image.png':
                    # This should be segment_00.png
                    prompt_item['first_frame'] = os.path.join(frames_dir_abs, 'segment_00.png')
                elif first_frame_ref.startswith('segment_') and first_frame_ref.endswith('.png'):
                    # Convert relative segment reference to absolute path
                    prompt_item['first_frame'] = os.path.join(frames_dir_abs, first_frame_ref)
                else:
                    # For any other reference, try to resolve it
                    prompt_item['first_frame'] = os.path.join(frames_dir_abs, first_frame_ref)
                logging.info(f"Updated first_frame for segment {prompt_item.get('segment', i)}: {prompt_item['first_frame']}")

            # Fix last_frame path
            if 'last_frame' in prompt_item:
                last_frame_ref = prompt_item['last_frame']
                if last_frame_ref.startswith('segment_') and last_frame_ref.endswith('.png'):
                    # Convert relative segment reference to absolute path
                    prompt_item['last_frame'] = os.path.join(frames_dir_abs, last_frame_ref)
                else:
                    # For any other reference, try to resolve it
                    prompt_item['last_frame'] = os.path.join(frames_dir_abs, last_frame_ref)
                logging.info(f"Updated last_frame for segment {prompt_item.get('segment', i)}: {prompt_item['last_frame']}")

        # Get FLF2V model directory for keyframe mode
        flf2v_model_dir = config.get('flf2v_model_dir', './Wan2.1-FLF2V-14B-720P')

        # Step 3: Generate video segments using FLF2V model
        logging.info("Generating video segments using FLF2F model (keyframe mode)...")

    else:  # chaining mode
        # ----- CHAINING MODE -----
        # No need to generate keyframes in chaining mode
        # We'll extract frames from each video segment as we go
        logging.info("Skipping keyframe generation in chaining mode")

        # Create a directory for extracted frames if it doesn't exist
        extracted_frames_dir = os.path.join(output_dir, "extracted_frames")
        os.makedirs(extracted_frames_dir, exist_ok=True)

        # Check backend configuration - only require I2V model for local backends
        default_backend = config.get('default_backend', 'wan2.1')

        # Only check for I2V model directory if using local backend
        if default_backend in ['wan2.1', 'local']:
            i2v_model_dir = config.get('i2v_model_dir', './Wan2.1-I2V-14B-720P')
            if not os.path.exists(i2v_model_dir):
                logging.error(f"I2V model directory not found: {i2v_model_dir}")
                raise FileNotFoundError(f"I2V model directory not found: {i2v_model_dir}")
        else:
            logging.info(f"Using remote backend: {default_backend}, skipping local model checks")

        # Step 3: Generate video segments using I2V model in chaining mode
        logging.info("Generating video segments using I2V model (chaining mode)...")
        print(f"\n{Colors.BOLD}{Colors.PURPLE}Generating Video in Chaining Mode:{Colors.RESET}")

        # Generate video segments in chaining mode (must be sequential)
        video_paths = generate_video_chaining_mode(
            config=config,
            video_prompts=enhanced_data['video_prompts'],
            output_dir=output_dir,
            segment_duration=config.get('segment_duration_seconds', 5.0)  # Default to 5.0s if not in config
        )

        # Skip to video concatenation step
        logging.info(f"Generated {len(video_paths)} video segments in chaining mode")
        # End of chaining mode - continue to final video generation below

    # Only run video generation for keyframe mode (chaining mode already generated videos)
    if generation_mode == 'keyframe':
        # Check if we should use single-keyframe mode (for Veo3 with I2I)
        single_keyframe_mode = config.get('single_keyframe_mode', False)

        video_paths = generate_video_segments(
            wan2_dir=wan2_dir,
            config=config,
            video_prompts=enhanced_data['video_prompts'],
            output_dir=output_dir,
            flf2v_model_dir=flf2v_model_dir,
            frame_num=config.get('frame_num', 81),
            single_keyframe_mode=single_keyframe_mode
        )

    # Log results
    logging.info(f"Generated {len(video_paths)} video segments")
    for i, path in enumerate(video_paths):
        logging.info(f"Video segment {i+1}: {path}")

    # Stitch video segments together if we have more than one
    if len(video_paths) > 1:
        logging.info("Stitching video segments together...")
        final_output = os.path.join(output_dir, "final_video.mp4")
        stitched_video = stitch_video_segments(video_paths, final_output)
        if stitched_video:
            logging.info(f"Final stitched video saved to: {final_output}")
        else:
            logging.error("Failed to stitch video segments together")
    else:
        # If only one segment, just use that as the final output
        final_output = video_paths[0] if video_paths else None
        logging.info(f"Only one video segment generated: {final_output}")

    logging.info(f"Pipeline completed successfully. Final output: {final_output}")
    return final_output

# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Long Video Generation Pipeline")
    parser.add_argument('--config', type=str, default='pipeline_config.yaml',
                       help='Path to the pipeline configuration file')
    parser.add_argument('--prompt', type=str,
                       help='Override the prompt from the config file')

    args = parser.parse_args()
    run_pipeline(args.config, args.prompt)

if __name__ == "__main__":
    main()
