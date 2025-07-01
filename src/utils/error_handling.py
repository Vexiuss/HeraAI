"""
Error handling utilities for HeraAI

Provides centralized error handling and exception management.
"""

import logging
import traceback
import functools
from typing import Callable, Any


def handle_exceptions(default_return: Any = None, log_errors: bool = True):
    """
    Decorator to handle exceptions gracefully
    
    Args:
        default_return: Value to return on exception
        log_errors: Whether to log errors
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger("heraai.errors")
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                return default_return
        
        return wrapper
    return decorator


class HeraAIException(Exception):
    """Base exception for HeraAI"""
    pass


class MemoryError(HeraAIException):
    """Memory system related errors"""
    pass


class AudioError(HeraAIException):
    """Audio system related errors"""
    pass


class AIModelError(HeraAIException):
    """AI model related errors"""
    pass


class UserManagementError(HeraAIException):
    """User management related errors"""
    pass


def log_and_raise(exception_class: type, message: str, logger_name: str = "heraai"):
    """
    Log an error and raise an exception
    
    Args:
        exception_class: Exception class to raise
        message: Error message
        logger_name: Logger name to use
    """
    logger = logging.getLogger(logger_name)
    logger.error(message)
    raise exception_class(message) 