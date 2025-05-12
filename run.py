import argparse
import socket
import logging
from app import create_app
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run coloring page creation server')
    parser.add_argument('--port', type=int, default=Config.PORT, help=f'Port to run the server on (default: {Config.PORT})')
    args = parser.parse_args()
    
    try:
        # Find available port and start server
        port = get_available_port(args.port)
        logger.info(f"Starting server at http://{Config.HOST}:{port}")
        
        app = create_app()
        app.run(host=Config.HOST, port=port, debug=Config.DEBUG)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True) 