# common.py
import logging
import os
from datetime import datetime

_logger = None


def setup_logger(name="App"):
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)

    # 创建logs目录
    os.makedirs("logs", exist_ok=True)

    # 文件handler
    log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    return _logger


def get_default_model_path():
    """自动获取最新模型路径"""
    model_files = []
    for f in os.listdir():
        if f.endswith(".pth") and not f.startswith("checkpoint"):
            model_files.append(f)

    if not model_files:
        return "poetry_model.pth"

    # 返回修改时间最新的模型
    return max(model_files, key=lambda x: os.path.getmtime(x))
