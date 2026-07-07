"""
Public API for the neurons_model extension system.
"""

from __future__ import annotations

from .discovery import discover_extensions, get_extension_path, import_extension, load_extension, load_extensions
from .registry import (
    ExtensionInfo,
    clear_registered_extensions,
    get_registered_extension,
    is_registered,
    list_registered_extensions,
    register_extension,
)

__all__ = [
    "ExtensionInfo",
    "clear_registered_extensions",
    "discover_extensions",
    "get_extension_path",
    "get_registered_extension",
    "import_extension",
    "is_registered",
    "list_registered_extensions",
    "load_extension",
    "load_extensions",
    "register_extension",
]
