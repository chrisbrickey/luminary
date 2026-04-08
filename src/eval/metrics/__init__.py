"""Auto-register all metrics by importing metric modules.

This module automatically discovers and imports all Python files in the
metrics directory (except __init__ and base), which triggers their
register_metric() calls at import time.
"""

from pathlib import Path

# Import base first to make METRIC_REGISTRY available
from src.eval.metrics.base import METRIC_REGISTRY  # noqa: F401

# Auto-discover and import all metric modules
_METRICS_DIR = Path(__file__).parent
for _module_path in _METRICS_DIR.glob("*.py"):
    if _module_path.stem not in ("__init__", "base"):
        # Import the module to trigger its register_metric() calls
        __import__(f"src.eval.metrics.{_module_path.stem}")
