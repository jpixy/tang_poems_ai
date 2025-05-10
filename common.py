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

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger

