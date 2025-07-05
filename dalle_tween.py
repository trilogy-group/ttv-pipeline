"""Utilities for generating DALL·E tween frames"""

import base64
import logging
import os
import time
from typing import List

from keyframe_generator import reword_prompt_for_safety


from PIL import Image
from openai import OpenAI
from pipeline import run_command, stitch_video_segments

logger = logging.getLogger(__name__)


def generate_dalle_prompts(start_image: str, end_image: str, frame_count: int, api_key: str) -> List[str]:
    """Generate DALL·E prompts for in-between frames.

    Args:
        start_image: Path to the starting keyframe image file.
        end_image: Path to the ending keyframe image file.
        frame_count: Number of intermediate frames to describe.
        api_key: OpenAI API key.

    Returns:
        A list of text prompts for each tween frame.

    Raises:
        Exception: Propagates any errors from the OpenAI API call.
    """
    logger.info(
        "Generating %s DALL·E prompts between %s and %s",
        frame_count,
        start_image,
        end_image,
    )

    with open(start_image, "rb") as f:
        start_b64 = base64.b64encode(f.read()).decode()

    with open(end_image, "rb") as f:
        end_b64 = base64.b64encode(f.read()).decode()

    client = OpenAI(api_key=api_key)

    system_message = (
        "You generate concise DALL·E prompts describing a smooth visual "
        "transition between two images. Maintain the artistic style, color "
        "palette and key visual elements from the start and end frames to "
        "ensure continuity. Number the prompts starting at 1 and do not "
        "include additional commentary."
    )
    user_content = [
        {
            "type": "text",
            "text": (
                f"Create {frame_count} intermediate image descriptions for DALL·E 3 "
                "that morph the first image into the second. Reply with one "
                "numbered prompt per line."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{start_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{end_b64}"}},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=frame_count * 50,
        )
    except Exception as exc:
        logger.error("OpenAI request failed: %s", exc)
        raise

    text = response.choices[0].message.content.strip()
    prompts = [line.split(".", 1)[-1].strip() for line in text.splitlines() if line.strip()]
    prompts = [p for p in prompts if p]

    if len(prompts) != frame_count:
        logger.warning(
            "Expected %s prompts but received %s", frame_count, len(prompts)
        )

    return prompts


def generate_dalle_images(
    prompts: List[str],
    output_dir: str,
    api_key: str,
    start_index: int = 0,
    max_retries: int = 2,
) -> List[str]:
    """Generate images from prompts using DALL·E 3.

    Args:
        prompts: List of text prompts to render.
        output_dir: Directory where generated frames will be saved.
        api_key: OpenAI API key.
        start_index: Starting index for output file numbering.
        max_retries: Number of retries for each API call on failure.

    Returns:
        Paths to the saved image files in order.

    Raises:
        Exception: Propagates any errors after exhausting retries.
    """
    client = OpenAI(api_key=api_key)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    for idx, prompt in enumerate(prompts, start=start_index):
        frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
        logger.info("Generating frame %s with DALL·E 3", idx)

        for retry in range(max_retries + 1):
            try:
                generation_prompt = prompt
                if retry == 2:
                    generation_prompt = reword_prompt_for_safety(prompt, api_key)
                    logger.info("Reworded prompt on retry %s: %s", retry, generation_prompt)

                response = client.images.generate(
                    model="dall-e-3",
                    prompt=generation_prompt,
                    n=1,
                    size="1024x1024",
                    quality="standard",
                    response_format="b64_json",
                )
                break
            except Exception as exc:
                logger.error(
                    "Error generating frame %s (attempt %s/%s): %s",
                    idx,
                    retry + 1,
                    max_retries,
                    exc,
                )

                if retry == max_retries:
                    raise
                time.sleep(2)

        image_base64 = response.data[0].b64_json
        with open(frame_path, "wb") as img_file:
            img_file.write(base64.b64decode(image_base64))

        saved_paths.append(os.path.abspath(frame_path))

    return saved_paths


def combine_images_to_gif(image_paths: List[str], gif_path: str, duration: float = 0.2) -> str:
    """Combine a list of images into an animated GIF.

    Args:
        image_paths: Ordered list of image file paths to include in the GIF.
        gif_path: Path where the resulting GIF should be saved.
        duration: Delay between frames in seconds.

    Returns:
        Path to the created GIF.
    """
    logger.info("Creating GIF %s from %s images", gif_path, len(image_paths))

    if not image_paths:
        raise ValueError("image_paths must contain at least one frame")

    frames = [Image.open(p) for p in image_paths]
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    first_frame, *rest = frames
    first_frame.save(
        gif_path,
        save_all=True,
        append_images=rest,
        duration=int(duration * 1000),
        loop=0,
    )

    logger.info("Saved animated GIF to %s", gif_path)
    return os.path.abspath(gif_path)


def generate_flf2v_tween(
    start_frame: str,
    end_frame: str,
    output_path: str,
    wan2_dir: str,
    flf2v_model_dir: str,
    frame_num: int = 16,
) -> str:
    """Generate a short video tween between two keyframes using Wan2.1 FLF2V."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    base_cmd = [
        "--task",
        "flf2v-14B",
        "--size",
        "1280*720",
        "--ckpt_dir",
        flf2v_model_dir,
        "--first_frame",
        start_frame,
        "--last_frame",
        end_frame,
        "--frame_num",
        str(frame_num),
        "--prompt",
        "smooth transition",
        "--save_file",
        output_path,
        "--sample_guide_scale",
        "5.0",
        "--sample_steps",
        "40",
        "--sample_shift",
        "5.0",
    ]

    cmd = ["python", "generate.py"] + base_cmd
    run_command(cmd, cwd=wan2_dir)
    return os.path.abspath(output_path)


def combine_videos(video_paths: List[str], out_file: str) -> str:
    """Concatenate video files into a single output."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    return stitch_video_segments(video_paths, out_file) or out_file
