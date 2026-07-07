"""
Registry for discovered neurons_model extensions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .error_codes import ExtensionNotFoundError, ExtensionRegistrationError


@dataclass(frozen=True, slots=True)
class ExtensionInfo:
    """Metadata captured for a registered extension."""

    package: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)


_REGISTERED_EXTENSIONS: dict[str, ExtensionInfo] = {}


def register_extension(*, package: str, name: str, version: str, description: str = "", author: str = "", 
                       functions: Mapping[str, Callable[..., Any]] | None = None,
                       variables: Mapping[str, Any] | None = None,
                       ) -> ExtensionInfo:
    """Register an extension and return its normalized metadata."""
    if not package:
        raise ExtensionRegistrationError("Extension package name must be provided.")
    if not name:
        raise ExtensionRegistrationError("Extension display name must be provided.")
    if not version:
        raise ExtensionRegistrationError("Extension version must be provided.")

    existing = _REGISTERED_EXTENSIONS.get(package)
    if existing is not None:
        return existing

    for info in _REGISTERED_EXTENSIONS.values():
        if info.name == name:
            raise ExtensionRegistrationError(
                f"Extension name '{name}' is already registered by package '{info.package}'."
            )

    metadata = ExtensionInfo(package=package, name=name, version=version, 
                             description=description, author=author,
                             functions=dict(functions or {}),
                             variables=dict(variables or {}),
                             )
    _REGISTERED_EXTENSIONS[package] = metadata
    return metadata


def get_registered_extension(extension: str) -> ExtensionInfo:
    """Return a registered extension by package or display name."""
    if extension in _REGISTERED_EXTENSIONS:
        return _REGISTERED_EXTENSIONS[extension]

    for info in _REGISTERED_EXTENSIONS.values():
        if info.name == extension:
            return info

    raise ExtensionNotFoundError(f"Extension '{extension}' is not registered.")


def is_registered(extension: str) -> bool:
    """Return True when an extension is already registered."""
    try:
        get_registered_extension(extension)
    except ExtensionNotFoundError:
        return False
    return True


def list_registered_extensions() -> list[ExtensionInfo]:
    """Return registered extensions sorted by display name."""
    return sorted(_REGISTERED_EXTENSIONS.values(), key=lambda info: (info.name, info.package))


def clear_registered_extensions() -> None:
    """Clear the registry. Intended for tests and local tooling."""
    _REGISTERED_EXTENSIONS.clear()


__all__ = [
    "ExtensionInfo",
    "register_extension",
    "get_registered_extension",
    "is_registered",
    "list_registered_extensions",
    "clear_registered_extensions",
]
