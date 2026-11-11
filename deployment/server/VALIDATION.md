# Checks

From `deployment/server/`, run these checks in the deployment environment:

```bash
python tests/check_package.py
python tests/check_storage.py
```

`check_package.py` checks Python syntax, vendor source hashes, depth preservation and completion, view ordering, circular waypoint peaks and coordinate conversion. `check_storage.py` checks record directories, environment and CLI overrides, and child-process path inheritance.

To check model loading and replay a saved observation:

```bash
python replay.py --input /path/to/model-input.npz --instruction "Go forward." --output /path/to/result.json
```

For robot setup, follow the [calibration procedure](../robot/CALIBRATION.md) and the [deployment guide](../README.md).
