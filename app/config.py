import os

# Определяем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Основные настройки приложения
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB максимальный размер файла
    
    # Пути к директориям
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    
    # Настройки обработки изображений
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_IMAGE_SIZE = (1920, 1080)  # максимальное разрешение
    JPEG_QUALITY = 95  # качество сохранения JPEG
    
    # Параметры анализа изображений
    IMAGE_ANALYSIS = {
        'edge_detection': {
            'low_threshold': 100,
            'high_threshold': 200,
        },
        'color_clustering': {
            'n_clusters': 5,
            'random_state': 42,
        },
        'texture_analysis': {
            'distances': [1],
            'angles': [0],
            'levels': 256,
        },
    }
    
    # Пороговые значения для классификации
    CLASSIFICATION_THRESHOLDS = {
        'color_diversity': 0.3,
        'edge_density': 0.1,
        'texture_complexity': 0.4,
        'gradient_strength': 0.2,
        'edge_continuity': 0.5,
    }
    
    # Настройки кэширования
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Настройки логирования
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BASE_DIR, 'flask.log')
    LOG_LEVEL = 'INFO'
    
    @staticmethod
    def init_app(app):
        """Инициализация приложения"""
        # Создаем необходимые директории
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
        
        # Настраиваем логирование
        import logging
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=10000,
            backupCount=3
        )
        handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        handler.setLevel(Config.LOG_LEVEL)
        
        app.logger.addHandler(handler)
        app.logger.setLevel(Config.LOG_LEVEL)
        
        return app

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    DEBUG = False
    TESTING = True
    # Используем временные директории для тестов
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'tests', 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'tests', 'results')

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # В продакшене используем более строгие настройки
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB
    JPEG_QUALITY = 90
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        # Дополнительные настройки для продакшена
        import logging
        from logging.handlers import SMTPHandler
        
        # Настраиваем отправку ошибок на почту
        mail_handler = SMTPHandler(
            mailhost=os.environ.get('MAIL_SERVER', 'localhost'),
            fromaddr=os.environ.get('MAIL_SENDER'),
            toaddrs=[os.environ.get('ADMIN_EMAIL')],
            subject='Application Error'
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 