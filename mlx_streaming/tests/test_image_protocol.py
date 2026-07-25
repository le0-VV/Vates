from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from mlx_streaming.protocol.images import (
    ImageLimits,
    ImageProtocolError,
    fetch_public_https,
    normalise_messages,
)


def _png_data_url(size=(2, 1), colour=(12, 34, 56)):
    output = BytesIO()
    Image.new("RGB", size, colour).save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode()
    return f"data:image/png;base64,{encoded}", output.getvalue()


def test_normalise_messages_preserves_text_image_order_and_rgb_pixels():
    data_url, _ = _png_data_url()
    result = normalise_messages(
        [
            {"role": "system", "content": "Be exact"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "after"},
                ],
            },
        ]
    )

    assert result.messages == [
        {"role": "system", "content": "Be exact"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        },
    ]
    assert len(result.images) == 1
    assert result.images[0].mode == "RGB"
    assert result.images[0].getpixel((0, 0)) == (12, 34, 56)


def test_normalise_messages_accepts_bounded_https_through_injected_fetcher():
    _, png = _png_data_url()
    seen = []
    result = normalise_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://images.example/image.png"},
                    }
                ],
            }
        ],
        fetcher=lambda url, limits: seen.append((url, limits)) or png,
    )

    assert seen[0][0] == "https://images.example/image.png"
    assert len(result.images) == 1


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": []},
        {"role": "user", "content": [{"type": "audio", "audio": "x"}]},
        {
            "role": "system",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                }
            ],
        },
    ],
)
def test_normalise_messages_rejects_invalid_content_shapes(message):
    with pytest.raises(ImageProtocolError):
        normalise_messages([message])


@pytest.mark.parametrize(
    "url",
    [
        "data:image/gif;base64,AA==",
        "data:image/png;base64,not-base64!",
        "http://example.com/a.png",
        "file:///tmp/a.png",
        "https://user:pass@example.com/a.png",
        "https://example.com/a.png#fragment",
    ],
)
def test_normalise_messages_rejects_unsupported_or_unsafe_urls(url):
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url}}],
    }
    with pytest.raises(ImageProtocolError):
        normalise_messages([message], fetcher=lambda _url, _limits: b"unused")


def test_data_url_enforces_encoded_decoded_image_and_pixel_limits():
    data_url, png = _png_data_url(size=(4, 4))
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": data_url}}],
    }

    with pytest.raises(ImageProtocolError, match="encoded"):
        normalise_messages(
            [message],
            limits=ImageLimits(max_encoded_bytes=4),
        )
    with pytest.raises(ImageProtocolError, match="decoded"):
        normalise_messages(
            [message],
            limits=ImageLimits(max_decoded_bytes=len(png) - 1),
        )
    with pytest.raises(ImageProtocolError, match="pixels"):
        normalise_messages(
            [message],
            limits=ImageLimits(max_pixels=15),
        )


def test_image_count_limit_is_enforced():
    data_url, _ = _png_data_url()
    message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
    with pytest.raises(ImageProtocolError, match="images"):
        normalise_messages([message], limits=ImageLimits(max_images=1))


class _Response:
    def __init__(self, status, body=b"", location=None):
        self.status_code = status
        self.headers = {}
        if location is not None:
            self.headers["Location"] = location
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size):
        assert chunk_size > 0
        yield self._body

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def _resolver(mapping):
    def resolve(host, port, **_kwargs):
        return [(2, 1, 6, "", (mapping[host], port))]

    return resolve


def test_https_fetch_revalidates_redirects_and_disables_automatic_redirects():
    _, png = _png_data_url()
    first = _Response(302, location="https://cdn.example/final.png")
    second = _Response(200, body=png)
    session = _Session([first, second])

    body = fetch_public_https(
        "https://images.example/start.png",
        ImageLimits(),
        session=session,
        resolver=_resolver(
            {
                "images.example": "93.184.216.34",
                "cdn.example": "1.1.1.1",
            }
        ),
    )

    assert body == png
    assert [call[0] for call in session.calls] == [
        "https://images.example/start.png",
        "https://cdn.example/final.png",
    ]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    assert first.closed and second.closed


@pytest.mark.parametrize(
    "host,address",
    [
        ("localhost", "127.0.0.1"),
        ("private.example", "10.0.0.4"),
        ("link.example", "169.254.1.1"),
        ("loop.example", "::1"),
        ("multicast.example", "224.0.0.1"),
        ("reserved.example", "240.0.0.1"),
    ],
)
def test_https_fetch_rejects_non_global_targets_before_request(host, address):
    session = _Session([])
    with pytest.raises(ImageProtocolError, match="public"):
        fetch_public_https(
            f"https://{host}/image.png",
            ImageLimits(),
            session=session,
            resolver=_resolver({host: address}),
        )
    assert session.calls == []


def test_https_fetch_enforces_streamed_byte_limit():
    session = _Session([_Response(200, body=b"12345")])
    with pytest.raises(ImageProtocolError, match="decoded"):
        fetch_public_https(
            "https://images.example/image.png",
            ImageLimits(max_decoded_bytes=4),
            session=session,
            resolver=_resolver({"images.example": "93.184.216.34"}),
        )
