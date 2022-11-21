# -*- coding: utf-8 -*-
from loguru import logger
import logging
from dask.distributed import WorkerPlugin
import sys
from pathlib import Path
import time


class LoggerPlugin(WorkerPlugin):

    def __init__(self, *args, **kwargs):
        self.proxy_logger = kwargs['proxy_logger']

    def setup(self, worker):
        self.proxy_logger.configure()


class InterceptHandler(logging.Handler):
    """Intercept log messages are reroute them to the loguru logger."""
    def emit(self, record):
        # Retrieve context where the logging call occurred, this happens to be
        # in the 7th frame upward.
        logger_opt = logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


class ProxyLogger(object):

    def __init__(self, output_dir, birth=None, log_to_term=True):

        self.birth = birth or time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.log_to_term = log_to_term

    def __reduce__(self):
        return (ProxyLogger, (self.output_dir, self.birth, self.log_to_term))

    def configure(self):
        logging.basicConfig(handlers=[InterceptHandler()], level="WARNING")

        # Put together a formatting string for the logger.

        tim_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
        lvl_fmt = "<level>{level}</level>"
        src_fmt = "<cyan>{module}</cyan>:<cyan>{function}</cyan>"
        msg_fmt = "<level>{message}</level>"

        fmt = " | ".join([tim_fmt, lvl_fmt, src_fmt, msg_fmt])

        output_path = Path(self.output_dir)
        output_name = Path(f"{self.birth}.log.qc")

        logger.remove()  # Remove existing handlers.

        if self.log_to_term:
            logger.add(
                sys.stderr,
                level="INFO",
                format=fmt,
                enqueue=True,
                colorize=True
            )

        logger.add(
            str(output_path / output_name),
            level="DEBUG",
            format=fmt,
            enqueue=True,
            colorize=False
        )
