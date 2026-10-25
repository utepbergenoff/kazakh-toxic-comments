"""Reproduce published metrics without API calls, model training or comment text."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "evaluation" / "reproduce.py"), run_name="__main__")
