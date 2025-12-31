import logging
from datetime import datetime
from pathlib import Path

LOG_NAME = "morrow_plots_simulation.log"  # or whatever you use
log_dir_defaults = './logs'


def get_logger(
        name: str = None,
        log_dir: Path | str = log_dir_defaults,
        level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or refresh) a logger that writes to a date-stamped file like:
        2025-10-21-morrow_plots_simulation.log
    Also logs to console. No rotation/backups.
    """
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    name = LOG_NAME or name
    # Build today's file name
    today_name = f"{datetime.now():%Y-%m-%d}-{LOG_NAME}"
    log_path = log_dir / f"{today_name}-"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # If handlers exist, clear them so we don't keep writing to yesterday's file.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    # File handler (date-stamped, no rotation)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger
