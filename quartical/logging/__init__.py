# -*- coding: utf-8 -*-
from loguru import logger
import logging
import sys
from pathlib import Path
import time


class InterceptHandler(logging.Handler):
    """Intercept log messages are reroute them to the loguru logger."""
    def emit(self, record):
        # Retrieve context where the logging call occurred, this happens to be
        # in the 7th frame upward.
        logger_opt = logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


class ProxyLogger(object):

    def __init__(self, output_dir):
        self.birth = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)

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

        logger.remove()

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
