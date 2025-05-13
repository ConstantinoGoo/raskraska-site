from flask import Flask
from config import get_config
import os
import logging
from .utils.file_manager import FileManager
from .utils.log_config import LogConfig

def create_app(config_name='default'):
    """
    Создает экземпляр приложения Flask
    Args:
        config_name: имя конфигурации ('development', 'testing', 'production')
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Загружаем конфигурацию
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Инициализируем менеджер файлов
    file_manager = FileManager(
        upload_dir=str(config.UPLOAD_FOLDER),
        results_dir=str(config.RESULTS_FOLDER)
    )
    
    # Добавляем менеджер файлов в контекст приложения
    app.file_manager = file_manager
    
    # Настраиваем логирование
    log_config = LogConfig(
        log_file=str(config.LOG_FILE),
        max_bytes=config.LOG_MAX_BYTES,
        backup_count=config.LOG_BACKUP_COUNT,
        log_level=config.LOG_LEVEL
    )
    
    # Регистрируем маршруты
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    # Логируем успешную инициализацию
    logging.info(f"Application initialized in {config_name} mode")
    
    return app 