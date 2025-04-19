from folder_paths import (
    get_input_directory,
    get_annotated_filepath,
    exists_annotated_filepath,
    models_dir,
)

from os import listdir
from os.path import isfile, join, isdir, exists
from PIL import Image

from comfy.sd import CLIP
from comfy.model_management import get_torch_device, soft_empty_cache
import gc

import transformers

from preprocess import preprocess
from utility import tokenize_text

import model_utils

INT_MAX = 0xFFFFFFFFFFFFFFFF
FLOAT_MAX = 1_000_000.0


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
                "clip": ("CLIP",),
                "image": (
                    sorted(image_files),
                    {"image_upload": True},
                ),
                "model_name": (model_dirs,),
                "min_new_tokens": (
                    "INT",
                    {"default": 20, "min": 0, "max": INT_MAX, "step": 1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 50, "min": 35, "max": INT_MAX, "step": 1},
                ),
                "num_beams": (
                    "INT",
                    {"default": 20, "min": 1, "max": INT_MAX, "step": 1},
                ),
                "penalty_alpha": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "top_k": ("INT", {"default": 50, "min": 0, "max": INT_MAX, "step": 1}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "preprocess_mode": (["exact_keyword", "exact_prompt", "none"],),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = ("clip_output", "string_output")
    FUNCTION = "image_caption"
    CATEGORY = "image-caption"

    def image_caption(
        self,
        clip: CLIP,
        image: str,
        model_name: str,
        min_new_tokens: int,
        max_new_tokens: int,
        num_beams: int,
        penalty_alpha: float,
        top_k: int,
        repetition_penalty: float,
        preprocess_mode: str,
    ) -> str:
        image_path = get_annotated_filepath(image)
        img = Image.open(image_path).convert("RGB")

        model_path = join(models_dir, "image_captioners", model_name)

        dev = get_torch_device()

        model = model_utils.get_model(model_path)
        processor = transformers.AutoProcessor.from_pretrained(model_path)

        try:
            inputs = processor(
                images=img,
                return_tensors="pt",
                padding=True,
                use_fast=True,
            ).to(dev)
        except:
            inputs = processor(
                images=img,
                return_tensors="pt",
            ).to(dev)

        out = model.generate(
            **inputs,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            early_stopping=True,
            num_beams=num_beams,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            remove_invalid_values=True,
            renormalize_logits=True,
        )

        output = processor.decode(
            out[0], skip_special_tokens=True, cleanup_tokenization_spaces=True
        )

        del model
        del processor
        soft_empty_cache()
        gc.collect()

        output = preprocess(output, preprocess_mode)

        print_string = f"{'  IMAGE CAPTION OUTPUT  '.center(200, '#')}\n"
        print_string += f"{output}\n\n"
        print_string += f"{'#' * 200}\n"

        print(print_string)

        return (tokenize_text(clip, output), output)

    @classmethod
    def VALIDATE_INPUTS(s, image: str, model_name: str):
        if not exists_annotated_filepath(image):
            return f"Invalid image file: {image}"

        model_path = join(models_dir, "image_captioners", model_name)
        if not exists(model_path):
            return f"{model_path} is not exists"

        return True
