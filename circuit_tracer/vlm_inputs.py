from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class VLMInput:
    """Container for a single multimodal user input."""

    prompt: str
    image: str | Path | Any
    image_url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class PreparedInput:
    """Normalized model inputs used by backend-specific attribution code."""

    trace_inputs: torch.Tensor | dict[str, torch.Tensor]
    input_ids: torch.Tensor
    prompt_text: str
    input_mode: Literal["text", "multimodal"] = "text"
    image_path: str | None = None
    image_url: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
