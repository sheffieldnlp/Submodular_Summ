import logging


def get_logger(log_name: str, log_path: str):
    logger = logging.getLogger(log_name)
    formatter = logging.Formatter(logging.BASIC_FORMAT)

    if not len(logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler_info = logging.FileHandler(log_path, mode='w')
        file_handler_info.setFormatter(formatter)
        file_handler_info.setLevel(logging.DEBUG)
        logger.addHandler(file_handler_info)

        logger.setLevel(logging.DEBUG)
    return logger
