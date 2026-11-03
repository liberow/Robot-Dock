"""Explicit .env loading at the application boundary, never on library import."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: Path, *, required: bool = False) -> None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Environment file not found: {path}")
        return
    load_dotenv(path, override=False, interpolate=False, encoding="utf-8")


@dataclass(frozen=True)
class PokebotEnvironment:
    api_key: str = field(repr=False)
    base_url: str = "https://api.pokebot.com/v1"
    model: str = "poke-aurora-preview"

    @classmethod
    def read(cls) -> "PokebotEnvironment":
        # Official vendor variable remains accepted for existing deployments.
        return cls(
            api_key=os.environ.get("ROBOT_DOCK_POKEBOT_API_KEY")
            or os.environ.get("PLATFORM_API_KEY", ""),
            base_url=os.environ.get("ROBOT_DOCK_POKEBOT_BASE_URL") or "https://api.pokebot.com/v1",
            model=os.environ.get("ROBOT_DOCK_POKEBOT_MODEL") or "poke-aurora-preview",
        )


@dataclass(frozen=True)
class OpenAIEnvironment:
    api_key: str = field(repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    extra_body: dict = field(default_factory=dict, repr=False)

    @classmethod
    def read(cls) -> "OpenAIEnvironment":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1",
            model=os.environ.get("OPENAI_API_MODEL", ""),
            extra_body=json.loads(os.environ.get("OPENAI_API_EXTRA_BODY") or "{}"),
        )
