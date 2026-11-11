"""Persistent experiment records, separate from disposable model caches."""
import os
from pathlib import Path

MODEL = 'st2nav'
ROLE = 'server'


def results_root(value=None):
    selected = value if value is not None else os.environ.get('NAV_RESULTS_DIR')
    if selected is None:
        return Path.home() / 'navigation_results' / MODEL
    if not str(selected).strip():
        raise ValueError('Results directory must not be empty')
    return Path(selected).expanduser().resolve()


def configure_results(value=None):
    """Export the resolved root so child processes use exactly the same location."""
    root = results_root(value)
    os.environ['NAV_RESULTS_DIR'] = str(root)
    return root


def results_dir():
    return results_root() / ROLE
