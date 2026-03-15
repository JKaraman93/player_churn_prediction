"""
Logging Module: Centralized Logging Configuration

Provides a consistent logging setup across the entire project with
proper formatting and log levels.
"""

import logging
import sys
from typing import Optional
from bet.utils.constants import LOG_LEVEL


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               defaults to LOG_LEVEL from constants
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Set log level
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


def log_dataframe_info(df, name: str = "DataFrame", logger: Optional[logging.Logger] = None) -> None:
    """
    Log shape and schema information for a Spark DataFrame for debugging.
    
    Args:
        df: Spark DataFrame to log info for
        name: Friendly name for the DataFrame in logs
        logger: Logger instance (creates one if not provided)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    try:
        count = df.count()
        cols = len(df.columns)
        logger.info(f"{name}: {count} rows × {cols} columns")
    except Exception as e:
        logger.error(f"Failed to log {name} info: {str(e)}")
