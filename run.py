import argparse
import socket
import logging
from app import create_app
from config import get_config
from app.utils.log_config import LogConfig
from app.utils.file_manager import FileManager

def get_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from the specified one"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")

def cleanup_on_shutdown(file_manager: FileManager):
    """Выполняет очистку при завершении работы"""
    file_manager.cleanup_temp_files()
    file_manager.clean_old_files()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run coloring page creation server')
    parser.add_argument('--port', type=int, help='Port to run the server on')
    parser.add_argument('--env', type=str, default='development',
                       choices=['development', 'testing', 'production'],
                       help='Environment to run the server in')
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию для указанного окружения
        config = get_config(args.env)
        
        # Инициализируем систему логирования
        log_config = LogConfig(
            log_file=str(config.LOG_FILE),
            max_bytes=config.LOG_MAX_BYTES,
            backup_count=config.LOG_BACKUP_COUNT,
            log_level=config.LOG_LEVEL
        )
        
        # Инициализируем менеджер файлов
        file_manager = FileManager(
            upload_dir=str(config.UPLOAD_FOLDER),
            results_dir=str(config.RESULTS_FOLDER)
        )
        
        # Find available port
        port = get_available_port(args.port or config.PORT)
        logging.info(f"Starting server at http://{config.HOST}:{port}")
        
        # Create and run app
        app = create_app(config_name=args.env)
        
        # Регистрируем функцию очистки при завершении
        import atexit
        atexit.register(lambda: cleanup_on_shutdown(file_manager))
        
        app.run(host=config.HOST, port=port, debug=config.DEBUG)
        
    except Exception as e:
        logging.error(f"Error starting server: {str(e)}", exc_info=True) 