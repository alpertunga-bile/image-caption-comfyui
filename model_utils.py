from json import load
from os.path import join, exists
import transformers

from utility import (
    get_torch_dtype,
    check_torch_version_is_enough,
)

from comfy.model_management import is_device_cuda, get_torch_device

import torch


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
            f"Given model's architecture is not supported in the transformers version {transformers.__version__}"
        )

    return model_class


def get_model(model_path: str):
    dev = get_torch_device()

    model_class = get_model_class(model_path)

    if model_class is None:
        raise ValueError("Model type is not recognized")

    model_configs = {}

    model_configs["torch_dtype"] = get_torch_dtype()

    if check_torch_version_is_enough(2, 1):
        model_configs["attn_implementation"] = "sdpa"

    try:
        model = model_class.from_pretrained(model_path, **model_configs).to(dev)
    except:
        model_configs["attn_implementation"] = None

        model = model_class.from_pretrained(model_path, **model_configs).to(dev)

    if check_torch_version_is_enough(2, 0) and is_device_cuda(dev):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model


__all__ = [get_model]
