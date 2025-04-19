from re import compile
from comfy.sd import CLIP

from preprocess import preprocess
from utility import tokenize_text


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
        print_string += f"{'#' * 200}\n"

        print(print_string)

        return (tokenize_text(clip, formatted_str), formatted_str + ", ")
