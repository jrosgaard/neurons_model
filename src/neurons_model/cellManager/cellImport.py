"""
src/neurons_model/cellManager/cellImport.py

Imports cell type definitions from YAML or JSON files into a CellManager
registry. YAML loading delegates to the existing load_cell_type_specs()
machinery in cell_types.py so that glia-specific fields, surveillance
sub-parameters, and aliases are handled consistently with the rest of the
codebase. Individual JSON files (one cell type per file) map to a single
CellTypeSpec via CellTypeSpec.from_mapping().

Imported specs are stored as JSON in cellClasses/ for later reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..cell_types import CellTypeSpec, load_cell_type_specs
from .cellManager import CellManager

if TYPE_CHECKING:
    pass

CELL_CLASSES_FOLDER = Path(__file__).parent / "cellClasses"


class CellImport:
    """
    Loads cell type specs from YAML or JSON files and registers them with a
    CellManager. Optionally persists imported types to the cellClasses/ folder.
    """

    def __init__(self, cell_manager: CellManager) -> None:
        self.cell_manager = cell_manager

    def import_from_yaml(self, yaml_path: str | Path) -> list[CellTypeSpec]:
        """
        Load all cell type specs from a YAML file and register them.

        The YAML section is inferred automatically (e.g. ``neurons``,
        ``glial_populations``) using the same logic as the rest of the
        codebase. Raises ValueError if no recognisable section is found.
        """
        specs = load_cell_type_specs(yaml_path)
        for spec in specs:
            self.cell_manager.register(spec)
        return specs

    def import_from_json(self, json_path: str | Path) -> CellTypeSpec:
        """
        Load a single cell type spec from a JSON file and register it.

        The JSON file should contain one cell type definition as a flat
        mapping (the same shape as a single entry in a YAML populations list).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        spec = CellTypeSpec.from_mapping(data)
        self.cell_manager.register(spec)
        return spec

    def save_to_cellClasses(self, cell_class_data: dict) -> Path:
        """
        Persist a cell type definition dict to a JSON file in cellClasses/.

        The dict must contain a ``name`` key; the file will be written as
        ``cellClasses/<name><version>.json`` if a ``version`` key is present.
        Returns the path written.
        """
        if not cell_class_data.get("name"):
            raise ValueError("cell_class_data must contain a 'name' key.")
        CELL_CLASSES_FOLDER.mkdir(parents=True, exist_ok=True)
        json_path = CELL_CLASSES_FOLDER / f"{cell_class_data['name'] + cell_class_data.get('version', '')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cell_class_data, f, indent=4)
        return json_path
