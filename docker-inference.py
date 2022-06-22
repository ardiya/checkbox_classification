#! /usr/bin/env python
"""
Wrapper to docker-compose to run inference on single image
"""
from genericpath import exists
import yaml
import argparse
from pathlib import Path

from os import system

yaml_template = """
version: '3'
services:
  checkbox_classification:
    image: ardiya/checkbox_classification:latest
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image to be inferred")
    args = parser.parse_args()
    image_path = Path(args.image_path)
    docker_path = Path("/root/checkbox_classification/data") / image_path.name

    if not image_path.exists():
        print("Unable to find file ", image_path)
        from sys import exit
        exit(1)

    yaml_cfg = yaml.safe_load(yaml_template)
    yaml_cfg["services"]["checkbox_classification"]["volumes"] = [f"{image_path.absolute()}:{docker_path}"]
    yaml_cfg["services"]["checkbox_classification"]["command"] = f"python main_inference.py {docker_path}"
    with open("docker-compose-inference.yaml", "w") as fp:
        yaml.safe_dump(yaml_cfg, fp)

    system("docker-compose -f docker-compose-inference.yaml up")
