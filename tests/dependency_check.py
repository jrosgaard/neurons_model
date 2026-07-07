"""
tests/dependency_check.py
Checks that required dependencies are installed and compatible with the project.
"""

from __future__ import annotations

import sys
import importlib.metadata as metadata


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pytest",
    "requests",
    "PyYAML",
]


def check_package(package_name: str) -> None:
    try:
        version = metadata.version(package_name)
        print(f"{package_name}: {version}")
    except metadata.PackageNotFoundError:
        print(f"{package_name}: NOT INSTALLED")


def main() -> None:
    print("Checking Python version:\n")
    print(f"python: {sys.version}")

    print("\nChecking project dependencies:\n")
    for package in REQUIRED_PACKAGES:
        check_package(package)


if __name__ == "__main__":
    main()