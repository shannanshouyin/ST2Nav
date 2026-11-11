"""Exercise real path expressions and CLI parsing without robot/model execution."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    checked = 0
    with tempfile.TemporaryDirectory(prefix='navigation-results-check-') as temporary:
        base = Path(temporary)
        for role in ('robot', 'server'):
            storage = load(ROOT / role / 'storage.py', 'storage_' + role)
            with patch.dict(os.environ):
                os.environ.pop('NAV_RESULTS_DIR', None)
                with patch.object(Path, 'home', return_value=base / 'home'):
                    assert storage.results_dir() == base / 'home/navigation_results' / storage.MODEL / role
                os.environ['NAV_RESULTS_DIR'] = str(base / 'environment output')
                assert storage.results_dir() == base / 'environment output' / role
                os.environ['NAV_RESULTS_DIR'] = ''
                try:
                    storage.results_root()
                except ValueError:
                    pass
                else:
                    raise AssertionError('Empty output path accepted')
                os.environ['NAV_RESULTS_DIR'] = str(base / 'environment output')
                # Run the packaged CLI parser and configure_results call only.
                # No hardware/model initialization or command dispatch is executed.
                entry = ROOT / role / ('navigate.py' if role == 'robot' else 'server.py')
                tree = ast.parse(entry.read_text(encoding='utf-8'))
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
                prefix = []
                for node in function.body:
                    prefix.append(node)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'configure_results':
                        break
                else:
                    raise AssertionError('Missing CLI output configuration')
                options = load(ROOT / role / 'model_options.py', 'options_' + role)
                context = {'argparse': argparse, 'Path': Path, '__doc__': '',
                           'MODELS': options.MODELS, 'model_id': options.model_id,
                           'configure_results': storage.configure_results}
                override = base / 'explicit output'
                with patch.object(sys, 'argv', [str(entry), '--results-dir', str(override)]):
                    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(entry), 'exec'), context)
                assert storage.results_dir() == override / role
                child = subprocess.check_output(
                    [sys.executable, '-B', '-c', 'import storage; print(storage.results_dir())'],
                    cwd=ROOT / role, text=True)
                assert Path(child.strip()) == override / role
                # Evaluate each actual record-path assignment, then write and read a marker.
                for path in (ROOT / role).glob('*.py'):
                    source = ast.parse(path.read_text(encoding='utf-8'))
                    for node in ast.walk(source):
                        if not isinstance(node, ast.Assign):
                            continue
                        if not any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'results_dir' for c in ast.walk(node.value)):
                            continue
                        destination = eval(compile(ast.Expression(node.value), str(path), 'eval'), {
                            'results_dir': storage.results_dir,
                            'time': types.SimpleNamespace(time_ns=lambda: 123456),
                            'args': types.SimpleNamespace(command='scan')})
                        assert destination == override / role or override / role in destination.parents
                        destination.mkdir(parents=True, exist_ok=True)
                        marker = destination / 'storage-check.json'
                        marker.write_text(json.dumps({'role': role}), encoding='utf-8')
                        assert json.loads(marker.read_text()) == {'role': role}
                        checked += 1
        assert checked == 9, checked
    print(f'PASS: defaults, environment/CLI precedence, child inheritance, and {checked} actual record destinations; no robot access.')


if __name__ == '__main__':
    main()
