import configparser
import os
from pathlib import Path


def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Set default values
    defaults = {
        "Paths": {
            "pretrained_model": "./pretrained_models/gpt2-chinese-poem",
            "model_save_dir": "./models",
            "data_dir": "./data",
            "log_dir": "./logs",
        },
        "Model": {"freeze_layers": "all,-2", "max_length": "64"},
        "Training": {
            "epochs": "10",
            "batch_size": "32",
            "learning_rate": "0.0001",
            "grad_clip": "1.0",
            "weight_clip": "0.5",
            "save_every": "1",
        },
        "Optimizer": {"betas": "0.9,0.999", "weight_decay": "0.01"},
        "Generation": {"num_poems": "5", "temperature": "1.0", "top_k": "5"},
    }

    # Merge config with defaults
    for section, options in defaults.items():
        if not config.has_section(section):
            config.add_section(section)
        for option, value in options.items():
            if not config.has_option(section, option):
                config.set(section, option, value)

    # Create required directories
    for path_key in ["pretrained_model", "model_save_dir", "data_dir", "log_dir"]:
        Path(config["Paths"][path_key]).mkdir(parents=True, exist_ok=True)

    return config


# Global config object
CONFIG = load_config()

