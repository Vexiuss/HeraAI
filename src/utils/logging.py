"""
Logging utilities for HeraAI

Provides centralized logging configuration and utilities.
"""

import logging
import os
from datetime import datetime
from ..config.settings import SYSTEM_CONFIG


def setup_logger(name: str = "heraai", level: str = None) -> logging.Logger:
    """
    Set up a logger with the specified configuration
    
    Args:
        name: Logger name
        level: Logging level (defaults to config setting)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get configuration
    log_config = SYSTEM_CONFIG["logging"]
    log_level = level or log_config["level"]
    log_format = log_config["format"]
    log_file = log_config["file"]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")
    
    return logger


def log_performance(func):
    """
    Decorator to log function performance
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("heraai.performance")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper 