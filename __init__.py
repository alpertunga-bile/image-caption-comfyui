from sys import path
from os.path import dirname

path.append(dirname(__file__))

print(" Image Caption ComfyUI Node ".center(100, "-"))

from image_caption import ImageCaptionNode

NODE_CLASS_MAPPINGS = {
    "Image Caption Node": ImageCaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Caption Node": "Image Caption Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("/_\ Loaded Successfully")
print("-" * 100)
