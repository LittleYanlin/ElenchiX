import subprocess
import sys


def test_real_langgraph_import_suppresses_only_the_upstream_allowed_objects_notice():
    # A fresh process is needed: the warning only happens on the first import.
    script = """
import warnings
from elenchix.llm import _load_react_agent
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter('always')
    before = list(warnings.filters)
    assert callable(_load_react_agent())
    assert warnings.filters == before
    assert not any('allowed_objects' in str(item.message) for item in recorded), recorded
    warnings.warn('unrelated deprecation', LangChainPendingDeprecationWarning)
    assert any('unrelated deprecation' in str(item.message) for item in recorded)
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
