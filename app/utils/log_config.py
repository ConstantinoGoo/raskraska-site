import logging
import logging.handlers
import os
from typing import Optional
from pathlib import Path

class LogConfig:
    def __init__(self, 
                 log_file: str,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 log_level: str = 'INFO'):
        self.log_file = Path(log_file)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = getattr(logging, log_level.upper())
        
        # Создаем директорию для логов если нужно
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._configure_logging()
    
    def _configure_logging(self):
        """Настраивает систему логирования с ротацией файлов"""
        # Создаем форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Настраиваем ротацию файлов
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        
        # Настраиваем вывод в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)
        
        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Удаляем существующие обработчики
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Добавляем новые обработчики
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging system initialized") 