"""
Utilities for discovering and loading neurons_model extensions.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from . import registry as extension_registry
from .error_codes import ExtensionNotFoundError, ExtensionRegistrationError
from .registry import ExtensionInfo, get_registered_extension, is_registered

_EXTENSION_PREFIX = "_EXT_"


def _default_search_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return [Path.cwd(), repo_root]


def _normalize_search_paths(search_paths: Iterable[str | Path] | None) -> list[Path]:
    if search_paths is None:
        raw_paths = _default_search_paths()
    elif isinstance(search_paths, (str, Path)):
        raw_paths = [Path(search_paths)]
    else:
        raw_paths = [Path(path) for path in search_paths]

    normalized: list[Path] = []
    seen: set[Path] = set()
    for path in raw_paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            normalized.append(resolved)

    return normalized


def _is_extension_package(path: Path) -> bool:
    return (path.is_dir()
            and path.name.startswith(_EXTENSION_PREFIX)
            and (path / "__init__.py").is_file()
            and (path / "plugin.py").is_file()
            )


def _discover_extension_paths(search_paths: Iterable[str | Path] | None = None) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for root in _normalize_search_paths(search_paths):
        if not root.exists():
            continue
        for entry in root.iterdir():
            if _is_extension_package(entry) and entry.name not in discovered:
                discovered[entry.name] = entry
    
    return dict(sorted(discovered.items()))


def discover_extensions(search_paths: Iterable[str | Path] | None = None) -> list[str]:
    """Return the names of discoverable extension packages."""
    return list(_discover_extension_paths(search_paths))


def get_extension_path(extension_name: str,search_paths: Iterable[str | Path] | None = None,
                       ) -> Path:
    """Return the filesystem path for a discovered extension package."""
    discovered = _discover_extension_paths(search_paths)
    try:
        return discovered[extension_name]
    except KeyError as exc:
        available = ", ".join(discovered) or "<none>"
        raise ExtensionNotFoundError(
            f"Extension '{extension_name}' was not found. Available extensions: {available}"
                                     ) from exc


def import_extension(extension_name: str,search_paths: Iterable[str | Path] | None = None,
                     ) -> ModuleType:
    """Import an extension package, adding its parent directory to sys.path if needed."""
    extension_path = get_extension_path(extension_name, search_paths=search_paths)
    parent = str(extension_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    importlib.invalidate_caches()
    try:
        return importlib.import_module(extension_name)
    except ModuleNotFoundError as exc:
        raise ExtensionNotFoundError(
            f"Extension '{extension_name}' could not be imported from '{extension_path}'."
        ) from exc


def _invoke_plugin_register(register_callable, extension_name: str) -> ExtensionInfo | dict | None:
    signature = inspect.signature(register_callable)
    required_parameters = [parameter for parameter in signature.parameters.values()
                           if parameter.default is inspect.Signature.empty
                           and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                           ]

    if not required_parameters:
        return register_callable()
    if len(required_parameters) == 1:
        return register_callable(extension_registry)

    raise ExtensionRegistrationError(
        f"Extension '{extension_name}' has an unsupported register() signature.")


def load_extension(extension_name: str, search_paths: Iterable[str | Path] | None = None,
                   ) -> ExtensionInfo:
    """Import and register a single extension package."""
    if is_registered(extension_name):
        return get_registered_extension(extension_name)

    import_extension(extension_name, search_paths=search_paths)
    plugin_module = importlib.import_module(f"{extension_name}.plugin")
    register_callable = getattr(plugin_module, "register", None)
    if register_callable is None or not callable(register_callable):
        raise ExtensionRegistrationError(
            f"Extension '{extension_name}' does not define a callable register() function."
        )

    result = _invoke_plugin_register(register_callable, extension_name)
    if isinstance(result, ExtensionInfo):
        return result
    if isinstance(result, dict):
        return extension_registry.register_extension(package=extension_name, **result)

    if is_registered(extension_name):
        return get_registered_extension(extension_name)

    raise ExtensionRegistrationError(
        f"Extension '{extension_name}' did not return metadata and did not register itself."
    )


def load_extensions(extensions: Iterable[str] | None = None, 
                    search_paths: Iterable[str | Path] | None = None,
                    ) -> list[ExtensionInfo]:
    """Load a sequence of extension packages, or all discoverable extensions."""
    if extensions is None:
        extension_names = discover_extensions(search_paths)
    elif isinstance(extensions, str):
        extension_names = [extensions]
    else:
        extension_names = list(extensions)
    return [load_extension(name, search_paths=search_paths) for name in extension_names]


__all__ = [
    "discover_extensions",
    "get_extension_path",
    "import_extension",
    "load_extension",
    "load_extensions",
]
