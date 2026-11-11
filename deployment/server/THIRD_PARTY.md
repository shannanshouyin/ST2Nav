# Third-party sources

The inference server uses model and graph code from ST2Nav and ETP-R1, XLM-R tokenizer assets, and selected Habitat-Lab modules. `source_manifest.json` records the hashes of the vendor files.

- [ETP-R1](https://github.com/Cepillar/ETP-R1), commit `1570c82c1a6acde9d7d55ad8b9d3f02682152b1f`: [MIT notice](vendor/ST2Nav/LICENSE) and [additional notices](vendor/ST2Nav/licenses/).
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), commit `d6ed1c0a0e786f16f261de2beafe347f4186d0d8`: geometry and depth-encoder modules, with the [MIT notice](vendor/habitat-lab/LICENSE).
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python), commit `814556d15970dd2ecf1c9984e845ca02ab07e206`: installed separately on the robot.

The deployment adapter provides tokenizer padding masks, ordered RGBD views, robot/Habitat coordinate conversion and graph execution feedback. ST2Nav also uses periodic waypoint peak extraction.
