"""Bounded OpenAI image attachments with public-HTTPS SSRF guards."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import socket
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Callable
from urllib.parse import urljoin, urlsplit

import requests
from PIL import Image, UnidentifiedImageError


class ImageProtocolError(ValueError):
    """An image part is malformed, unsafe or exceeds configured limits."""


@dataclass(frozen=True)
class ImageLimits:
    max_images: int = 4
    max_encoded_bytes: int = 11 * 1024 * 1024
    max_decoded_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 16_777_216
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    max_redirects: int = 3


@dataclass(frozen=True)
class NormalisedContent:
    messages: list[dict]
    images: list[Image.Image]


ImageFetcher = Callable[[str, ImageLimits], bytes]
Resolver = Callable[..., list[tuple]]
_REDIRECTS = {301, 302, 303, 307, 308}
_MIME_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
}


def _url_parts(url: str):
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ImageProtocolError("image URL is malformed") from exc
    if parts.scheme != "https":
        raise ImageProtocolError("remote images require HTTPS")
    if not parts.hostname:
        raise ImageProtocolError("image URL requires a hostname")
    if parts.username is not None or parts.password is not None:
        raise ImageProtocolError("image URL credentials are forbidden")
    if parts.fragment:
        raise ImageProtocolError("image URL fragments are forbidden")
    return parts, port or 443


def _require_public_target(url: str, resolver: Resolver) -> None:
    parts, port = _url_parts(url)
    host = parts.hostname
    assert host is not None
    if host.lower() == "localhost":
        raise ImageProtocolError("image host must resolve only to public addresses")
    try:
        literal = ipaddress.ip_address(host)
        addresses = [literal]
    except ValueError:
        try:
            records = resolver(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ImageProtocolError("image host could not be resolved") from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except (IndexError, ValueError) as exc:
                raise ImageProtocolError("image host resolution is malformed") from exc
    if not addresses or any(
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        for address in addresses
    ):
        raise ImageProtocolError("image host must resolve only to public addresses")


def fetch_public_https(
    url: str,
    limits: ImageLimits,
    *,
    session=None,
    resolver: Resolver = socket.getaddrinfo,
) -> bytes:
    """Fetch one image while revalidating every manually followed redirect."""

    own_session = session is None
    client = requests.Session() if own_session else session
    current = url
    try:
        for redirect_count in range(limits.max_redirects + 1):
            _require_public_target(current, resolver)
            try:
                response = client.get(
                    current,
                    stream=True,
                    allow_redirects=False,
                    timeout=(
                        limits.connect_timeout_seconds,
                        limits.read_timeout_seconds,
                    ),
                )
            except requests.RequestException as exc:
                raise ImageProtocolError("HTTPS image fetch failed") from exc
            try:
                if response.status_code in _REDIRECTS:
                    location = response.headers.get("Location")
                    if not location:
                        raise ImageProtocolError(
                            "image redirect is missing a Location header"
                        )
                    if redirect_count >= limits.max_redirects:
                        raise ImageProtocolError("image redirect limit exceeded")
                    current = urljoin(current, location)
                    continue
                if not 200 <= response.status_code < 300:
                    raise ImageProtocolError(
                        f"HTTPS image fetch returned {response.status_code}"
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except ValueError as exc:
                        raise ImageProtocolError(
                            "image Content-Length is malformed"
                        ) from exc
                    if declared_bytes > limits.max_decoded_bytes:
                        raise ImageProtocolError("decoded image exceeds byte limit")
                chunks = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > limits.max_decoded_bytes:
                        raise ImageProtocolError("decoded image exceeds byte limit")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                response.close()
    finally:
        if own_session:
            client.close()
    raise ImageProtocolError("image redirect limit exceeded")


def _load_image(
    data: bytes,
    limits: ImageLimits,
    *,
    declared_mime: str | None,
) -> Image.Image:
    if len(data) > limits.max_decoded_bytes:
        raise ImageProtocolError("decoded image exceeds byte limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(data))
            image.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ImageProtocolError("image cannot be safely decoded") from exc
    if image.width * image.height > limits.max_pixels:
        raise ImageProtocolError("image exceeds pixels limit")
    if image.format not in _MIME_FORMATS.values():
        raise ImageProtocolError("image format must be PNG or JPEG")
    if declared_mime is not None and image.format != _MIME_FORMATS[declared_mime]:
        raise ImageProtocolError("data URL MIME type does not match image bytes")
    return image.convert("RGB")


def _decode_data_url(url: str, limits: ImageLimits) -> tuple[bytes, str]:
    prefix, separator, payload = url.partition(",")
    if not separator or not prefix.startswith("data:") or not prefix.endswith(";base64"):
        raise ImageProtocolError("image data URL must use base64")
    mime = prefix[5:-7].lower()
    if mime not in _MIME_FORMATS:
        raise ImageProtocolError("image data URL MIME type is unsupported")
    try:
        encoded = payload.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ImageProtocolError("image data URL base64 must be ASCII") from exc
    if len(encoded) > limits.max_encoded_bytes:
        raise ImageProtocolError("encoded image exceeds byte limit")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageProtocolError("image data URL contains malformed base64") from exc
    if len(decoded) > limits.max_decoded_bytes:
        raise ImageProtocolError("decoded image exceeds byte limit")
    return decoded, mime


def normalise_messages(
    messages: list[dict],
    limits: ImageLimits | None = None,
    fetcher: ImageFetcher | None = None,
) -> NormalisedContent:
    """Convert OpenAI image_url parts into Qwen image placeholders and PIL images."""

    active_limits = limits or ImageLimits()
    active_fetcher = fetcher or fetch_public_https
    normalised = []
    images = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ImageProtocolError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            normalised.append(dict(message))
            continue
        if not isinstance(content, list) or not content:
            raise ImageProtocolError(
                f"messages[{message_index}].content must be text or a non-empty array"
            )
        parts = []
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise ImageProtocolError(
                    f"messages[{message_index}].content[{part_index}] must be an object"
                )
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ImageProtocolError("text image-content parts require text")
                parts.append({"type": "text", "text": text})
                continue
            if part_type != "image_url":
                raise ImageProtocolError(f"unsupported content part {part_type!r}")
            if role == "system":
                raise ImageProtocolError("system messages cannot contain images")
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str) or not url:
                raise ImageProtocolError("image_url parts require a URL string")
            if len(images) >= active_limits.max_images:
                raise ImageProtocolError("request contains too many images")
            if url.startswith("data:"):
                data, mime = _decode_data_url(url, active_limits)
            else:
                _url_parts(url)
                data = active_fetcher(url, active_limits)
                mime = None
            images.append(
                _load_image(data, active_limits, declared_mime=mime)
            )
            parts.append({"type": "image"})
        copied = dict(message)
        copied["content"] = parts
        normalised.append(copied)
    return NormalisedContent(messages=normalised, images=images)
