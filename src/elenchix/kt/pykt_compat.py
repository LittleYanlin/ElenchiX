"""Narrow imports for the pinned upstream pyKT AKT backbone.

The upstream package eagerly imports every bundled model from ``pykt.__init__``.
Some unrelated models require optional packages that are not declared by
pyKT's installer and are unnecessary for AKT, SimpleKT, or SparseKT.  This
module creates normal namespace-package entries for the installed source and
loads only AKT, so unrelated optional pyKT models cannot break the experiment.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class PyKTComponents:
    model_classes: dict[str, type[Any]]
    set_seed: Callable[[int], None]
    source_root: Path


def _namespace(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def load_pykt_components() -> PyKTComponents:
    """Load the pinned installed pyKT modules without its eager umbrella import."""

    existing = sys.modules.get("pykt")
    if existing is not None and getattr(existing, "__file__", None):
        raise RuntimeError(
            "pykt was imported before the ElenchiX compatibility loader; "
            "start a fresh Python process and import through load_pykt_components()"
        )
    if existing is None:
        spec = importlib.util.find_spec("pykt")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if not locations:
            raise RuntimeError("pykt-toolkit is not installed; run `uv sync --group kt --locked`")
        root = Path(locations[0]).resolve()
        _namespace("pykt", root)
    else:
        locations = list(getattr(existing, "__path__", []))
        if not locations:
            raise RuntimeError("the existing pykt namespace has no source path")
        root = Path(locations[0]).resolve()

    if "pykt.models" not in sys.modules:
        _namespace("pykt.models", root / "models")

    akt_module = importlib.import_module("pykt.models.akt")
    utility_module = importlib.import_module("pykt.utils.utils")
    return PyKTComponents(
        model_classes={"akt": akt_module.AKT},
        set_seed=utility_module.set_seed,
        source_root=root,
    )
