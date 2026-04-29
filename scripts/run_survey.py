"""Run open-source data survey."""
from src.utils.logger import setup_logging, get_logger

if __name__ == "__main__":
    setup_logging()
    log = get_logger(__name__)
    log.info("Starting open-source data survey")
