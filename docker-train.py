#! /usr/bin/env python
"""
Wrapper to docker-compose to run training
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
    parser.add_argument("--cfg", default="default.yaml", type=str, help="The fullname of the .yaml file in config foler")
    args = parser.parse_args()

    curr_dir = Path(__file__).parent.resolve()
    data_dir = curr_dir / "data"
    model_dir = curr_dir / "model"
    config_dir = curr_dir / "config"

    yaml_cfg = yaml.safe_load(yaml_template)
    yaml_cfg["services"]["checkbox_classification"]["volumes"] = [
        f"{data_dir.absolute()}:/root/checkbox_classification/data",
        f"{model_dir.absolute()}:/root/checkbox_classification/model",
        f"{config_dir.absolute()}:/root/checkbox_classification/config",
    ]
    yaml_cfg["services"]["checkbox_classification"][
        "command"
    ] = f"python main_train.py config/{ args.cfg }"

    with open("docker-compose-train.yaml", "w") as fp:
        yaml.safe_dump(yaml_cfg, fp)

    system("docker-compose -f docker-compose-train.yaml up")
