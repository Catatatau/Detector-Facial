import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from config.settings import config

def setup_logger(name: str = "face_platform") -> logging.Logger:
    """Configure and return a logger using rich."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    log_level_str = config.system.logging_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Format for file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(file_formatter)
    
    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
