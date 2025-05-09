# config.py
import configparser
import os
from pathlib import Path


def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")

    # 设置默认值
    defaults = {
        "Training": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "save_dir": "models",
        },
        "Model": {"embed_dim": 128, "hidden_dim": 256, "num_layers": 2},
        "Generation": {"max_length": 64, "num_poems": 5, "temperature": 1.0},
    }

    # 合并配置
    for section, options in defaults.items():
        if not config.has_section(section):
            config.add_section(section)
        for option, value in options.items():
            if not config.has_option(section, option):
                config.set(section, option, str(value))

    # 确保保存目录存在
    save_dir = config["Training"]["save_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    return config


# 全局配置对象
CONFIG = load_config()
