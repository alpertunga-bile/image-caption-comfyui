from typing import Tuple
from functools import lru_cache
import transformers
import torch
from comfy.model_management import get_torch_device, should_use_fp16, should_use_bf16
from comfy.sd import CLIP


def check_required_package_version(
    current_major: int, current_minor: int, required_major: int, required_minor: int
) -> bool:
    major_check = current_major >= required_major
    minor_check = current_major == required_major and current_minor >= required_minor

    return major_check or minor_check


def get_major_minor_versions(
    version_str: str, split_char: str = "."
) -> Tuple[int, int]:
    version_splitted = version_str.split(split_char)
    version_major = int(version_splitted[0])
    version_minor = int(version_splitted[1])

    return (version_major, version_minor)


@lru_cache
def check_package_version(
    version_str: str, min_major: int, min_minor: int, split_char: str = "."
) -> bool:
    package_major, package_minor = get_major_minor_versions(version_str, split_char)

    return check_required_package_version(
        package_major, package_minor, min_major, min_minor
    )


def check_torch_version_is_enough(min_major: int, min_minor: int) -> bool:
    return check_package_version(torch.__version__, min_major, min_minor)


def check_transformers_version(min_major: int, min_minor: int) -> bool:
    return check_package_version(transformers.__version__, min_major, min_minor)


def get_torch_dtype():
    dev = get_torch_device()

    if should_use_bf16(device=dev):
        req_torch_dtype = torch.bfloat16
    elif should_use_fp16(device=dev):
        req_torch_dtype = torch.float16
    else:
        req_torch_dtype = torch.float32

    return req_torch_dtype


def tokenize_text(clip: CLIP, text: str) -> list:
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


__all__ = [get_torch_dtype, check_torch_version_is_enough, check_transformers_version]
