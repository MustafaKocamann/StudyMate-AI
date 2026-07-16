import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = os.path.abspath(LOG_FILE)
    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(handler.baseFilename) == log_path
        for handler in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    return logger
