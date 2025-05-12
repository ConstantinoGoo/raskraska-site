import os

class Config:
    # Основные настройки приложения
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB максимальный размер файла
    
    # Настройки сервера
    HOST = '127.0.0.1'
    PORT = 8080
    DEBUG = True
    
    # Пути к директориям
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    
    # Настройки обработки изображений
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_IMAGE_SIZE = (1920, 1080)  # максимальное разрешение
    JPEG_QUALITY = 95  # качество сохранения JPEG
    
    # Настройки логирования
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BASE_DIR, 'flask.log')
    LOG_LEVEL = 'INFO' 