from folder_paths import (
    get_input_directory,
    get_annotated_filepath,
    exists_annotated_filepath,
    models_dir,
)
from os import listdir
from os.path import isfile, join, isdir, exists
from PIL import Image
from hashlib import sha256

from comfy.model_management import get_torch_device, should_use_fp16, should_use_bf16

from torch import bfloat16 as torch_bfloat16
from torch import float16 as torch_float16
from torch import float32 as torch_float32

import transformers

from json import load

from preprocess import preprocess


def get_torch_dtype():
    dev = get_torch_device()

    if should_use_bf16(device=dev):
        req_torch_dtype = torch_bfloat16
    elif should_use_fp16(device=dev):
        req_torch_dtype = torch_float16
    else:
        req_torch_dtype = torch_float32

    return req_torch_dtype


def get_model_class(model_path: str):
    config_filepath = join(model_path, "config.json")
    if exists(config_filepath) is False:
        raise ValueError("Config file is not found")

    with open(config_filepath) as json_file:
        config_file = load(json_file)

    model_class_name = config_file["architectures"][0]

    model_class = None

    try:
        model_class = getattr(transformers, model_class_name)
    except AttributeError:
        raise ValueError(
            "Given model's architecture is not supported in the transformers"
        )

    return model_class


class ImageCaptionNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = get_input_directory()
        image_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

        models_dir_path = join(models_dir, "image_captioners")
        model_dirs = [
            d for d in listdir(models_dir_path) if isdir(join(models_dir_path, d))
        ]

        return {
            "required": {
                "image": (
                    sorted(image_files),
                    {"image_upload": True},
                ),
                "model_name": (model_dirs,),
                "min_new_tokens": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 50, "min": 35, "max": 200, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "image_caption"
    CATEGORY = "image-caption"

    def image_caption(
        self, image: str, model_name: str, min_new_tokens: int, max_new_tokens: int
    ) -> str:
        image_path = get_annotated_filepath(image)
        img = Image.open(image_path).convert("RGB")

        model_path = join(models_dir, "image_captioners", model_name)

        torch_dtype = get_torch_dtype()
        dev = get_torch_device()

        model_class = get_model_class(model_path)

        if model_class is None:
            raise ValueError("Model type is not recognized")

        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(dev)

        processor = transformers.AutoProcessor.from_pretrained(model_path)

        inputs = processor(img, return_tensors="pt").to(dev, torch_dtype)

        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
        )

        output = processor.decode(
            out[0], skip_special_tokens=True, cleanup_tokenization_spaces=True
        )

        output = preprocess(output)

        print(output)

        return output

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not exists_annotated_filepath(image):
            return f"Invalid image file: {image}"

        return True
