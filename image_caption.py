from folder_paths import (
    get_input_directory,
    get_annotated_filepath,
    exists_annotated_filepath,
    models_dir,
)
from os import listdir
from os.path import isfile, join, isdir, exists
from PIL import Image

from comfy.model_management import get_torch_device, should_use_fp16, should_use_bf16
from comfy.sd import CLIP

from torch import bfloat16 as torch_bfloat16
from torch import float16 as torch_float16
from torch import float32 as torch_float32

import transformers

from json import load

from preprocess import preprocess

from re import compile


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


def tokenize_text(clip: CLIP, text: str) -> list:
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


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
                    {"default": 1, "min": 1, "max": INT_MAX, "step": 1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": FLOAT_MAX, "step": 0.1},
                ),
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
        repetition_penalty: float,
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
            **inputs,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            early_stopping=True,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            remove_invalid_values=True,
        )

        output = processor.decode(
            out[0], skip_special_tokens=True, cleanup_tokenization_spaces=True
        )

        output = preprocess(output)

        print_string = f"{'  IMAGE CAPTION OUTPUT  '.center(200, '#')}\n"
        print_string += f"{output}\n\n"
        print_string += f"{'#'*200}\n"

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


class InsertPromptNode:
    _format_prompt_regex = compile(r"\s*{\s*prompt_string\s*}\s*")

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_string": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "prompt_format": (
                    "STRING",
                    {
                        "default": "{prompt_string}",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = ("clip_output", "string_output")
    FUNCTION = "format_prompt"
    CATEGORY = "image-caption"

    def format_prompt(self, clip: CLIP, prompt_string: str, prompt_format: str):
        formatted_str = self._format_prompt_regex.sub(prompt_string, prompt_format)

        formatted_str = preprocess(formatted_str)

        print_string = f"{'  INSERT PROMPT NODE OUTPUT  '.center(200, '#')}\n"
        print_string += f"{formatted_str}\n\n"
        print_string += f"{'#'*200}\n"

        print(print_string)

        return (tokenize_text(clip, formatted_str), formatted_str + ", ")
