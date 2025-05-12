from flask import Flask
from config import Config
import os

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
    app.config.from_object(Config)
    
    # Create necessary directories
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    result_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    
    for folder in [upload_folder, result_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Регистрируем маршруты
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app 