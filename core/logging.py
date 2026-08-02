import logging
from logging.handlers import RotatingFileHandler

import structlog


def setup_logging():
    # 5MB limit for log files
    max_bytes = 5 * 1024 * 1024

    info_handler = RotatingFileHandler(
        "info.log", maxBytes=max_bytes, backupCount=5
    )
    info_handler.setLevel(logging.INFO)

    error_handler = RotatingFileHandler(
        "error.log", maxBytes=max_bytes, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[info_handler, error_handler]
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    return structlog.get_logger(name)
