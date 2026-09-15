import importlib.util

import pytest

from elenchix.kt.pykt_compat import load_pykt_components


def test_pinned_pykt_components_if_installed() -> None:
    if importlib.util.find_spec("pykt") is None:
        pytest.skip("install the locked kt dependency group to test pyKT")
    components = load_pykt_components()
    assert set(components.model_classes) == {"akt"}
    assert components.source_root.name == "pykt"
    assert callable(components.set_seed)
