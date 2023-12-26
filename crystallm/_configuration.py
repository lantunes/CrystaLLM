from argparse import ArgumentParser
from typing import Type, List, Dict, Any
from dataclasses import dataclass, fields
from omegaconf import DictConfig, OmegaConf
import yaml


def _parse_cli_overrides(overrides: List[str], config_dataclass: Type[dataclass]) -> Dict[str, Any]:
    cli_args = {}
    allowed_options = {field.name: field.type for field in fields(config_dataclass)}
    for override in overrides:
        if "=" in override:
            key, value = override.split("=")
            if key not in allowed_options:
                raise KeyError(f"'{key}' is not a supported option")
            expected_type = allowed_options[key]
            try:
                # convert value to the expected type
                cli_args[key] = expected_type(value)
            except ValueError:
                raise ValueError(f"invalid type for '{key}': expected {expected_type}")
    return cli_args


def _load_config_from_yaml(file_path: str) -> DictConfig:
    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)
    return OmegaConf.create(yaml_content)


def parse_config(config_dataclass: Type[dataclass]) -> DictConfig:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to a config YAML file.")
    parser.add_argument("overrides", nargs="*")
    # parse the CLI args
    args = parser.parse_args()

    # default options
    opt = OmegaConf.structured(config_dataclass())

    if args.config:
        # override the default options with the yaml values
        yaml_options = _load_config_from_yaml(args.config)
        opt = OmegaConf.merge(opt, yaml_options)

    cli_options = _parse_cli_overrides(args.overrides, config_dataclass)
    opt = OmegaConf.merge(opt, cli_options)
    return opt
