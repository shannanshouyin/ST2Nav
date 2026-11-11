# Checkpoint configuration

Configure a compatible ST2Nav checkpoint and the baseline 90-degree waypoint predictor. The model checkpoint must contain RGB, depth and navigation network weights together with the matching model configuration. Refer to the [ETP-R1 repository](https://github.com/Cepillar/ETP-R1.git) for the waypoint predictor originally named `check_cwp_bestdist_hfov90`.

Place the checkpoint files in this directory, or set `NAV_CHECKPOINT` and `NAV_WAYPOINT` to their local paths. Set the corresponding `filename` and `sha256` entries in `manifest.json` to match the selected files. Compute each checksum with `sha256sum /path/to/file`. Environment path overrides still enforce the manifest checksums.

Ensure that the checkpoint matches the model architecture and configuration before running inference. XLM-R tokenizer assets are included in the vendor source; the adapter reads RGB and depth encoder weights from the model checkpoint rather than separate CLIP or DD-PPO files.
