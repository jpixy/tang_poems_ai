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

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 文件日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # 控制台日志 - 针对Kaggle优化格式
    console_handler = logging.StreamHandler()
    if "KAGGLE_URL_BASE" in os.environ:
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    else:
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    # 针对Kaggle的额外设置
    if "KAGGLE_URL_BASE" in os.environ:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

    return _logger

