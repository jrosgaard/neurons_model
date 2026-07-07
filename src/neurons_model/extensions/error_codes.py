"""
Extension-specific exceptions for the neurons_model package.
"""

from __future__ import annotations


class ExtensionError(RuntimeError):
    """Base class for extension-related failures."""

class ExtensionNotFoundError(ExtensionError):
    """Raised when an extension cannot be discovered or imported."""

class ExtensionRegistrationError(ExtensionError):
    """Raised when an extension fails validation or registration."""


__all__ = [
    "ExtensionError",
    "ExtensionNotFoundError",
    "ExtensionRegistrationError",
]
