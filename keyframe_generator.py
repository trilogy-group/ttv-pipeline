#!/usr/bin/env python3
"""
Google Gemini Flash Image Generator for Keyframes

This script generates keyframe images for the video pipeline using Google's
Gemini Flash image generation model.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import io
import base64
import requests
from typing import Optional

from PIL import Image

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

def generate_keyframe_with_imageRouter(prompt, output_path, model_name, imageRouter_api_key):
    """Generate a keyframe image from a prompt using ImageRouter API"""
    logging.info(f"Using ImageRouter API with model: {model_name}")
    try:
        # Set up the request
        url = "https://ir-api.myqa.cc/v1/openai/images/generations"
        payload = {
            "prompt": prompt,
            "model": model_name,
            "quality": "auto"
        }
        headers = {
            "Authorization": f"Bearer {imageRouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Make the request
        logging.info("Sending request to ImageRouter API")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the response
        result = response.json()
        logging.info(f"ImageRouter API response received: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if not isinstance(result, dict) or 'data' not in result or not result['data']:
            raise ValueError(f"Invalid response format: {result}")
            
        # Get the first image data
        image_data = result['data'][0]
        
        # Handle both URL and b64_json formats
        if 'url' in image_data and image_data['url']:
            logging.info("Using URL from ImageRouter response")
            image_response = requests.get(image_data['url'])
            image_response.raise_for_status()
            image_bytes = image_response.content
        elif 'b64_json' in image_data and image_data['b64_json']:
            logging.info("Using b64_json from ImageRouter response")
            image_bytes = base64.b64decode(image_data['b64_json'])
        else:
            raise ValueError(f"No image data found in response: {image_data}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        logging.info(f"Saving image to {output_path} ({len(image_bytes) / 1024:.2f} KB)")
        img = Image.open(io.BytesIO(image_bytes))
        img.save(output_path)
        logging.info(f"Image saved: {os.path.abspath(output_path)}")
        
        return os.path.abspath(output_path)
        
    except Exception as e:
        logging.error(f"Error using ImageRouter API: {e}")
        raise

def generate_keyframe_with_stability(prompt, output_path, stability_api_key, input_image_path=None):
    """Generate a keyframe image using Stability AI API with optional image-to-image"""
    try:
        # Set headers
        headers = {
            "Authorization": f"Bearer {stability_api_key}",
            "Accept": "application/json"
        }
        
        # Initialize response variable to avoid scope issues
        response = None
        
        # Determine if we're doing text-to-image or image-to-image
        if input_image_path and os.path.exists(input_image_path):
            logging.info(f"Using input image for image-to-image: {input_image_path}")
            
            # Image-to-image endpoint
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
            
            # For image-to-image we need multipart/form-data with properly sized image
            try:
                # Open and resize the image to 1024x1024 (one of the accepted dimensions)
                with Image.open(input_image_path) as img:
                    # Get original dimensions
                    orig_width, orig_height = img.size
                    logging.info(f"Original image dimensions: {orig_width}x{orig_height}")
                    
                    # Resize to 1024x1024 (square) which is guaranteed to be accepted
                    resized_img = img.resize((1024, 1024), Image.LANCZOS)
                    
                    # Save to a temporary file - ensure consistent naming without nested paths
                    temp_dir = os.path.dirname(input_image_path)
                    temp_base = os.path.basename(input_image_path)
                    temp_img_path = os.path.join(temp_dir, f"{temp_base}.resized.png")
                    resized_img.save(temp_img_path)
                    logging.info(f"Resized image to 1024x1024 and saved to {temp_img_path}")
                
                # Form data
                data = {
                    'text_prompts[0][text]': prompt,
                    'text_prompts[0][weight]': '1',
                    'image_strength': '0.35',  # Lower = more influence from init image
                    'cfg_scale': '7',
                    'samples': '1',
                    'steps': '30'
                }
                
                # Use resized image file
                with open(temp_img_path, 'rb') as img_file:
                    files = {
                        'init_image': (os.path.basename(temp_img_path), img_file, 'image/png')
                    }
                    
                    logging.info(f"Sending image-to-image request to: {url}")
                    logging.info(f"With prompt: {prompt}")
                    
                    # Send request
                    response = requests.post(url, headers=headers, files=files, data=data)
                    logging.info(f"Response status code: {response.status_code}")
                    
            except Exception as e:
                logging.error(f"Error in image-to-image processing: {e}")
                raise
                
        else:
            # Text-to-image endpoint
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers["Content-Type"] = "application/json"
            
            # JSON payload
            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    }
                ],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30
            }
            
            logging.info(f"Sending text-to-image request to: {url}")
            logging.info(f"With prompt: {prompt}")
            
            # Send request
            response = requests.post(url, headers=headers, json=payload)
            logging.info(f"Response status code: {response.status_code}")
        
        # Better error handling
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_text = response.text
            logging.error(f"Stability AI API error: {e}")
            logging.error(f"Error details: {error_text}")
            raise
        
        # Parse response
        result = response.json()
        logging.info("Stability AI API response received")
        
        if not isinstance(result, dict) or 'artifacts' not in result or not result['artifacts']:
            raise ValueError(f"Invalid response format from Stability AI: {result}")
        
        # Get image data from first artifact
        image_data = result['artifacts'][0]
        save_base64_image(image_data['base64'], output_path)
        
        return os.path.abspath(output_path)
        
    except Exception as e:
        logging.error(f"Error using Stability AI API: {e}")
        raise

def save_base64_image(base64_str, save_path):
    """Save a base64 encoded image to a file"""
    try:
        # Fix potential nested path issues
        if '/frames/frames/' in save_path:
            save_path = save_path.replace('/frames/frames/', '/frames/')
            logging.warning(f"Corrected nested path to: {save_path}")
        
        # Simple base directory detection and correction
        base_dir = os.getcwd()
        if os.path.join(base_dir, 'output', 'frames', 'frames') in save_path:
            save_path = save_path.replace(
                os.path.join(base_dir, 'output', 'frames', 'frames'),
                os.path.join(base_dir, 'output', 'frames')
            )
            logging.warning(f"Fixed deeply nested path to: {save_path}")
        
        # Decode the base64 string
        img_data = base64.b64decode(base64_str)
        file_size_kb = len(img_data) / 1024
        logging.info(f"Saving image to {save_path} ({file_size_kb:.2f} KB)")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the decoded image data
        with open(save_path, "wb") as file:
            file.write(img_data)
            
        logging.info(f"Image saved: {save_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save base64 image: {e}")
        return False

def reword_prompt_for_safety(prompt, openai_api_key):
    """Use OpenAI to reword a prompt to avoid content moderation issues"""
    try:
        from openai import OpenAI
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Create a system message that asks for rewording
        system_message = """
        You are a helpful assistant that rewrites prompts to be safe and compliant with AI image generation guidelines.
        Reword the provided prompt to maintain its meaning but avoid any content that might trigger moderation systems.
        Focus on:
        1. Removing potentially problematic terms
        2. Using more neutral language
        3. Preserving the core visual concept
        4. Maintaining artistic style descriptions
        
        Respond ONLY with the reworded prompt, nothing else.
        """
        
        # Generate a reworded prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please reword this image generation prompt to avoid content moderation issues: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        reworded_prompt = response.choices[0].message.content.strip()
        logging.info(f"Reworded prompt: {reworded_prompt}")
        return reworded_prompt
    except Exception as e:
        logging.error(f"Error rewording prompt: {e}")
        # Return original prompt if rewording fails
        return prompt

def generate_keyframe_with_openai(prompt, output_path, openai_api_key, input_image_path=None, mask_path=None, size="1536x1024", max_retries=2):
    """Generate a keyframe image using OpenAI's gpt-image-1 model with automatic retries"""
    import openai
    from openai import OpenAI
    import base64
    from io import BytesIO
    import time
    
    # Import Colors for reworded prompts display
    from pipeline import Colors
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        if input_image_path and os.path.exists(input_image_path):
            # Image-to-image generation
            logging.info(f"Using input image: {input_image_path}")
            
            # Prepare arguments for the API call
            edit_args = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "n": 1,
                "size": size
                # gpt-image-1 doesn't need explicit response_format
            }
            
            # Open the input image
            edit_args["image"] = open(input_image_path, "rb")
            
            # If a mask is provided, include it in the API call
            if mask_path and os.path.exists(mask_path):
                logging.info(f"Using mask image: {mask_path}")
                edit_args["mask"] = open(mask_path, "rb")
            
            # Using the images.edit endpoint with retry mechanism
            for retry in range(max_retries + 1):  # +1 for initial attempt
                try:
                    if retry > 0:
                        logging.info(f"Retry {retry}/{max_retries} for prompt...")
                        # If this is the second retry, reword the prompt
                        if retry == 2:
                            prompt = reword_prompt_for_safety(prompt, openai_api_key)
                            edit_args["prompt"] = prompt
                            print(f"\n{Colors.BOLD}Reworded prompt:{Colors.RESET} {Colors.YELLOW}{prompt}{Colors.RESET}")
                    
                    # Make the API call
                    response = client.images.edit(**edit_args)
                    break  # If successful, exit retry loop
                except Exception as e:
                    logging.error(f"Error on attempt {retry+1}: {e}")
                    if retry == max_retries:  # If we've exhausted our retries
                        raise  # Re-raise the last exception
                    # Add a short delay between retries
                    time.sleep(2)
            
            # Process base64 response
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image directly from bytes
            with open(output_path, "wb") as f:
                f.write(image_bytes)
                
            logging.info(f"Image saved to: {output_path}")
            return os.path.abspath(output_path)
            
        else:
            # Text-to-image generation
            # Using the images.generate endpoint with retry mechanism
            for retry in range(max_retries + 1):  # +1 for initial attempt
                try:
                    if retry > 0:
                        logging.info(f"Retry {retry}/{max_retries} for prompt...")
                        # If this is the second retry, reword the prompt
                        if retry == 2:
                            prompt = reword_prompt_for_safety(prompt, openai_api_key)
                            print(f"\n{Colors.BOLD}Reworded prompt:{Colors.RESET} {Colors.YELLOW}{prompt}{Colors.RESET}")
                    
                    # Make the API call
                    response = client.images.generate(
                        model="gpt-image-1",
                        prompt=prompt,
                        n=1,
                        size=size
                    )
                    break  # If successful, exit retry loop
                except Exception as e:
                    logging.error(f"Error on attempt {retry+1}: {e}")
                    if retry == max_retries:  # If we've exhausted our retries
                        raise  # Re-raise the last exception
                    # Add a short delay between retries
                    time.sleep(2)
            
            # Process base64 response
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image directly from bytes
            with open(output_path, "wb") as f:
                f.write(image_bytes)
                
            logging.info(f"Image saved to: {output_path}")
            return os.path.abspath(output_path)
            
    except Exception as e:
        logging.error(f"Error using OpenAI API: {e}")
        raise

def create_mask_for_image(image_path, output_mask_path, openai_api_key, prompt=None):
    """Creates a mask for the specified image using OpenAI API"""
    try:
        import openai
        from openai import OpenAI
        import base64
        from io import BytesIO
        from PIL import Image
        
        # If no specific prompt is provided, use a default one
        if not prompt:
            prompt = "Generate a mask for this image where white (255) represents the main subject/character and black (0) represents the background. Return an alpha channel mask."
        
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Open the input image
        with open(image_path, "rb") as img_file:
            input_image = img_file.read()
        
        # Use OpenAI to generate the mask
        logging.info(f"Generating mask with prompt: {prompt}")
        response = client.images.edit(
            model="gpt-image-1",
            image=open(image_path, "rb"),
            prompt=prompt
        )
        
        # Save the mask
        mask_base64 = response.data[0].b64_json
        mask_bytes = base64.b64decode(mask_base64)
        
        # Create an RGBA image from the bytes
        mask_img = Image.open(BytesIO(mask_bytes))
        
        # Convert to grayscale and then to RGBA with alpha channel
        mask_grayscale = mask_img.convert("L")
        mask_rgba = mask_grayscale.convert("RGBA")
        mask_rgba.putalpha(mask_grayscale)  # Use grayscale as alpha
        
        # Save the mask
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        mask_rgba.save(output_mask_path, format="PNG")
        
        logging.info(f"Mask saved to: {output_mask_path}")
        return output_mask_path
        
    except Exception as e:
        logging.error(f"Error creating mask: {e}")
        return None

def generate_keyframe(prompt, output_path, model_name, imageRouter_api_key=None, stability_api_key=None, openai_api_key=None, input_image_path=None, mask_path=None, size=None, create_mask=False):
    """Wrapper function to generate a keyframe using the appropriate API"""
    # Import Colors class for colored output
    from pipeline import Colors
    
    # Print the prompt in blue right before generating the image
    print(f"\n{Colors.BOLD}Keyframe prompt:{Colors.RESET} {Colors.BLUE}{prompt}{Colors.RESET}")
    
    logging.info(f"Generating keyframe with model: {model_name}")
    
    # Set default model if none provided
    if model_name is None:
        model_name = "stabilityai/sdxl-turbo:free"
    
    logging.info(f"Generating image with model: {model_name}")
    
    # Use OpenAI if specified and API key is provided
    if "openai" in model_name.lower() and openai_api_key:
        # Default size for OpenAI's gpt-image-1 is 1536x1024
        openai_size = size or "1536x1024"
        
        # Auto-create a mask if requested and doing image-to-image
        if create_mask and input_image_path and not mask_path:
            logging.info("Auto-creating mask for image-to-image generation")
            mask_dir = os.path.dirname(output_path)
            mask_filename = f"{os.path.basename(input_image_path).split('.')[0]}_mask.png"
            auto_mask_path = os.path.join(mask_dir, mask_filename)
            
            mask_path = create_mask_for_image(
                image_path=input_image_path,
                output_mask_path=auto_mask_path,
                openai_api_key=openai_api_key
            )
            
            if mask_path:
                logging.info(f"Successfully created mask at {mask_path}")
            else:
                logging.warning("Failed to create mask, proceeding without it")
        
        return generate_keyframe_with_openai(
            prompt=prompt, 
            output_path=output_path, 
            openai_api_key=openai_api_key, 
            input_image_path=input_image_path,
            mask_path=mask_path,
            size=openai_size
        )
    
    # Use Stability AI if API key is provided
    elif stability_api_key:
        return generate_keyframe_with_stability(prompt, output_path, stability_api_key, input_image_path)
    
    # Fall back to ImageRouter if that API key is provided
    elif imageRouter_api_key:
        return generate_keyframe_with_imageRouter(prompt, output_path, model_name, imageRouter_api_key)
    
    # No valid API keys
    else:
        api_services = []
        if not openai_api_key and "openai" in model_name.lower():
            api_services.append("OpenAI")
        if not stability_api_key:
            api_services.append("Stability AI")
        if not imageRouter_api_key:
            api_services.append("ImageRouter")
            
        raise ValueError(f"API key required but not provided for services: {', '.join(api_services)}.")

def generate_keyframes_from_json(json_file, output_dir, model_name=None, imageRouter_api_key=None, stability_api_key=None, openai_api_key=None, initial_image_path=None, image_size=None):
    """Generate all keyframes sequentially from a JSON file with prompt data for character consistency"""
    
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # IMPORTANT: Ensure we don't nest directories - strip any potential nested paths
    # If output_dir is 'output/frames/frames', simplify to just 'output/frames'
    base_dir = os.getcwd()
    if 'output' in output_dir:
        parts = output_dir.split('output')
        if len(parts) > 1:
            # Use just the first path after 'output'
            path_parts = parts[1].strip('/').split('/')
            if path_parts and path_parts[0] == 'frames':
                # Use only the simple path: output/frames
                output_dir = os.path.join(base_dir, "output", "frames")
            else:
                output_dir = os.path.join(base_dir, "output", "frames")
        else:
            output_dir = os.path.join(base_dir, "output", "frames")
    else:
        # Default to a simple directory
        output_dir = os.path.join(base_dir, "output", "frames")
    
    # Force a clean start
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("segment_") and f.endswith(".png"):
                try:
                    os.remove(os.path.join(output_dir, f))
                except:
                    pass
    
    # Create the directory fresh
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the path - MUST be simple
    logging.info(f"SIMPLIFIED: Keyframes will be saved to: {output_dir}")
    
    # Track generated files
    generated_files = []
    
    # Check if the JSON has the expected structure
    if "keyframe_prompts" not in data:
        raise ValueError("JSON file does not contain 'keyframe_prompts' key")
    
    # Variable to track the previous image path for sequential generation
    prev_image_path = initial_image_path
    
    # Generate each keyframe sequentially, using the previous keyframe as input
    for item in sorted(data["keyframe_prompts"], key=lambda x: x.get("segment", 0)):
        segment = item.get("segment")
        prompt = item.get("prompt")
        
        if not segment or not prompt:
            logging.warning(f"Skipping invalid keyframe prompt item: {item}")
            continue
        
        # Ensure we don't accidentally create nested directories 
        output_path = os.path.join(output_dir, f"segment_{segment:02d}.png")
        logging.info(f"Will save keyframe to: {os.path.abspath(output_path)}")
        
        # Removed redundant "Generating keyframe X/Y" message
        # The keyframe message is already shown in colored output
        
        try:
            # Generate keyframe image
            if prev_image_path:
                # Only keep one of the prev image messages
                logging.info(f"Using previous keyframe as input: {prev_image_path}")
                generated_file = generate_keyframe(
                    prompt=prompt,
                    output_path=output_path,
                    model_name=model_name,
                    imageRouter_api_key=imageRouter_api_key,
                    stability_api_key=stability_api_key,
                    openai_api_key=openai_api_key,
                    input_image_path=prev_image_path,
                    mask_path=None,
                    size=image_size,
                    create_mask=False
                )
            else:
                # For first keyframe, use initial image if provided
                initial_input = initial_image_path if segment == 1 and initial_image_path else None
                if initial_input:
                    logging.info(f"Using initial image as input for first keyframe: {initial_input}")
                
                generated_file = generate_keyframe(
                    prompt=prompt,
                    output_path=output_path,
                    model_name=model_name,
                    imageRouter_api_key=imageRouter_api_key,
                    stability_api_key=stability_api_key,
                    openai_api_key=openai_api_key,
                    input_image_path=initial_input,
                    mask_path=None,
                    size=image_size,
                    create_mask=False
                )
            
            generated_files.append(generated_file)
            
            # Update prev_image_path for the next iteration
            prev_image_path = generated_file
            
            # Display colorful progress information
            print(f"{Colors.BOLD}{Colors.YELLOW}Segment {segment}{Colors.RESET} keyframe {Colors.GREEN}generated{Colors.RESET}")
        except Exception as e:
            # Display error in red
            print(f"{Colors.BOLD}{Colors.RED}Error generating keyframe for segment {segment}: {e}{Colors.RESET}")
            logging.error(f"Error generating keyframe for segment {segment}: {e}")
            logging.error(f"Skipping to next segment")
    
    return generated_files

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate keyframe images using Google Gemini")
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single image generation
    single_parser = subparsers.add_parser("single", help="Generate a single keyframe")
    single_parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    single_parser.add_argument("--output", required=True, help="Output file path")
    single_parser.add_argument("--model", help="Model name (default: stabilityai/sdxl-turbo:free)")
    single_parser.add_argument("--imageRouter-api-key", help="ImageRouter API key")
    single_parser.add_argument("--stability-api-key", help="Stability AI API key")
    single_parser.add_argument("--input-image", help="Input image for image-to-image generation")
    
    # Batch generation from JSON
    batch_parser = subparsers.add_parser("batch", help="Generate keyframes from a JSON file")
    batch_parser.add_argument("--json", required=True, help="JSON file with keyframe prompts")
    batch_parser.add_argument("--output-dir", required=True, help="Output directory for images")
    batch_parser.add_argument("--model", help="Model name (default: stabilityai/sdxl-turbo:free)")
    batch_parser.add_argument("--imageRouter-api-key", help="ImageRouter API key")
    batch_parser.add_argument("--stability-api-key", help="Stability AI API key")
    batch_parser.add_argument("--initial-image", help="Initial image to start the sequential generation")
    
    args = parser.parse_args()
    
    if args.command == "single":
        generate_keyframe(
            prompt=args.prompt, 
            output_path=args.output, 
            model_name=args.model, 
            imageRouter_api_key=args.imageRouter_api_key, 
            stability_api_key=args.stability_api_key, 
            openai_api_key=None,  # Need to add this parameter
            input_image_path=args.input_image,
            mask_path=None,
            size=None,
            create_mask=False
        )
    elif args.command == "batch":
        generate_keyframes_from_json(
            args.json, 
            args.output_dir, 
            args.model, 
            args.imageRouter_api_key, 
            args.stability_api_key, 
            args.initial_image
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
