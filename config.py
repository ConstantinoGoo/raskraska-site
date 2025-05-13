import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

@dataclass
class BaseConfig:
    """Базовая конфигурация с валидацией"""
    # Основные настройки приложения
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-production')
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB максимальный размер файла
    
    # Настройки сервера
    HOST: str = '127.0.0.1'
    PORT: int = 8080
    DEBUG: bool = False
    
    # Пути к директориям
    BASE_DIR: Path = Path(__file__).parent.absolute()
    UPLOAD_FOLDER: Path = BASE_DIR / 'uploads'
    RESULTS_FOLDER: Path = BASE_DIR / 'results'
    LOG_DIR: Path = BASE_DIR / 'logs'
    
    # Настройки обработки изображений
    ALLOWED_EXTENSIONS: set = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_IMAGE_SIZE: tuple = (1920, 1080)  # максимальное разрешение
    JPEG_QUALITY: int = 95  # качество сохранения JPEG
    
    # Настройки логирования
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: Path = LOG_DIR / 'app.log'
    LOG_LEVEL: str = 'INFO'
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    def validate(self) -> None:
        """Проверяет корректность конфигурации"""
        # Проверка секретного ключа
        if self.SECRET_KEY == 'dev-key-please-change-in-production' and not self.DEBUG:
            raise ValueError("Production environment requires a secure SECRET_KEY")
        
        # Проверка размера файла
        if self.MAX_CONTENT_LENGTH <= 0:
            raise ValueError("MAX_CONTENT_LENGTH must be positive")
        
        # Проверка разрешений изображений
        if not isinstance(self.MAX_IMAGE_SIZE, tuple) or len(self.MAX_IMAGE_SIZE) != 2:
            raise ValueError("MAX_IMAGE_SIZE must be a tuple of two integers")
        
        # Проверка качества JPEG
        if not 0 <= self.JPEG_QUALITY <= 100:
            raise ValueError("JPEG_QUALITY must be between 0 and 100")
        
        # Создание необходимых директорий
        for path in [self.UPLOAD_FOLDER, self.RESULTS_FOLDER, self.LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(BaseConfig):
    """Конфигурация для разработки"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(BaseConfig):
    """Конфигурация для тестирования"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    # Используем временные директории для тестов
    UPLOAD_FOLDER = Path('/tmp/test_uploads')
    RESULTS_FOLDER = Path('/tmp/test_results')
    LOG_DIR = Path('/tmp/test_logs')

class ProductionConfig(BaseConfig):
    """Конфигурация для продакшена"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    # В продакшене используем более строгие настройки
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB
    JPEG_QUALITY = 90  # Немного снижаем качество для оптимизации

# Словарь доступных конфигураций
config_dict: Dict[str, Any] = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> BaseConfig:
    """
    Возвращает конфигурацию на основе окружения
    Args:
        config_name: Имя конфигурации ('development', 'testing', 'production')
    Returns:
        BaseConfig: Объект конфигурации
    """
    if not config_name:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    config_class = config_dict.get(config_name, config_dict['default'])
    config = config_class()
    config.validate()
    return config

# Создаем объект конфигурации
Config = get_config() 