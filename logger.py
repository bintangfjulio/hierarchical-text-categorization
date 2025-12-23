"""
Centralized logging system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class Logger:
    """Centralized logger for the project."""
    
    _instance: Optional['Logger'] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = 'HierarchicalClassification',
        log_dir: Optional[Path] = None,
        level: int = logging.INFO
    ):
        """Initialize logger with file and console handlers."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'training_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
            self.info(f"Logging to file: {log_file}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """Log exception with traceback."""
        self.logger.exception(message)


def get_logger(
    name: str = 'HierarchicalClassification',
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> Logger:
    """Get or create logger instance."""
    return Logger(name=name, log_dir=log_dir, level=level)