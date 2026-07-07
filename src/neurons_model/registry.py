"""Simple name-based registry for runtime components."""

from collections.abc import Callable
from typing import Any


_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str, factory: Callable[..., Any]) -> None:
    _REGISTRY[name] = factory


def get(name: str) -> Callable[..., Any]:
    return _REGISTRY[name]


def list_registered() -> list[str]:
    return sorted(_REGISTRY)
