import logging
import sys
from functools import lru_cache

import structlog
from pythonjsonlogger import jsonlogger

from .config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.api_debug:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = jsonlogger.JsonFormatter(
            "%(timestamp)s %(levelname)s %(name)s %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Quiet some noisy loggers
    logging.getLogger("uvicorn.access").setLevel("WARNING")
    logging.getLogger("transformers").setLevel("WARNING")


@lru_cache()
def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to provide logging capabilities to other classes."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger instance for this class."""
        return get_logger(self.__class__.__name__)


configure_logging()