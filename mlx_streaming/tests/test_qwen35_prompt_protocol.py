from __future__ import annotations

import base64
from io import BytesIO

import mlx.core as mx
from PIL import Image

from mlx_streaming.models.qwen35 import Qwen35MoeAdapter
from mlx_streaming.server import validate_request
from mlx_streaming.tests.fakes.qwen35 import FakeQwen35Model


class _Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


def _image_url():
    output = BytesIO()
    Image.new("RGB", (1, 1), (1, 2, 3)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_validated_openai_prompt_reaches_official_qwen_template_unchanged():
    tool = {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
    request = validate_request(
        {
            "model": "qwen3.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_url()},
                        },
                    ],
                }
            ],
            "tools": [tool],
        },
        "qwen3.5",
        32,
    )
    prepared = {
        "input_ids": mx.array([[1, 2]]),
        "attention_mask": mx.array([[1, 1]]),
        "pixel_values": mx.array([[3.0]]),
        "image_grid_thw": mx.array([[1, 1, 1]]),
    }
    processor = _Processor()
    adapter = Qwen35MoeAdapter(
        input_preparer=lambda _processor, **_kwargs: dict(prepared)
    )

    adapter.prepare_inputs(
        FakeQwen35Model(),
        processor,
        request.messages,
        request.images,
        tools=[definition.as_openai_dict() for definition in request.tools],
        enable_thinking=request.enable_thinking,
    )

    assert processor.calls == [
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown?"},
                        {"type": "image"},
                    ],
                }
            ],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": [tool],
                "enable_thinking": True,
            },
        )
    ]
    assert request.images[0].getpixel((0, 0)) == (1, 2, 3)
