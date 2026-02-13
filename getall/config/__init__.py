"""Configuration module for getall."""

from getall.config.loader import load_config, get_config_path
from getall.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
