import os
import logging


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'log/beta_class16_dim_768_512_256_epochs_50_split20_layers_3'))
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler, file_handler]

    return logger


logger = _setup_logger()
