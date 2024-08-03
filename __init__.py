from sys import path
from os.path import dirname, exists, join
from os import mkdir

from folder_paths import models_dir

path.append(dirname(__file__))

print(" Image Caption ComfyUI Node ".center(100, "-"))

models_folder_path = join(models_dir, "image_captioners")

if exists(models_folder_path) is False:
    mkdir(models_folder_path)
    print(
        f"/_\ {models_folder_path} is created. Please put your model folders under this folder"
    )

print("/_\ Loading Image Caption")
from image_caption import ImageCaptionNode

print("/_\ Loading Insert Prompt Node")
from image_caption import InsertPromptNode

NODE_CLASS_MAPPINGS = {
    "Image Caption Node": ImageCaptionNode,
    "Insert Prompt Node": InsertPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Caption Node": "Image Caption Node",
    "Insert Prompt Node": "Insert Prompt Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("/_\ Loaded Successfully")
print("-" * 100)
