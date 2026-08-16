"""datafly: a data flywheel framework for robot manipulation."""

__version__ = "0.2.0"

from .loop import FlywheelConfig, run_flywheel, save_results

__all__ = ["FlywheelConfig", "run_flywheel", "save_results", "__version__"]
