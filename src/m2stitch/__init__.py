"""M2Stitch.

A MIST-inspired unofficial stitching imprementation in Python for
microscope images
"""

__author__ = """Yohsuke T. Fukai"""
__email__ = "ysk@yfukai.net"

from .stitching import stitch_images

__all__ = ["stitch_images"]
