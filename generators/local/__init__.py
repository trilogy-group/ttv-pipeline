"""
Local video generators that run on the user's hardware
"""

from .wan21_generator import Wan21Generator
from .hunyuan_video_generator import HunyuanVideoGenerator

__all__ = ['Wan21Generator', 'HunyuanVideoGenerator']
