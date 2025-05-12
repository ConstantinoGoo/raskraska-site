from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from ai_processor import process_image
from image_processor import process_image_locally  # Добавляем импорт локальной обработки
from dotenv import load_dotenv
import logging
import argparse

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Создаем директории для файлов
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # Максимальный размер файла 4MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не был загружен'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Файл не был выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Недопустимый формат файла. Разрешены только PNG и JPEG'}), 400
        
        # Сохраняем оригинальное изображение
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        logger.debug(f"Файл сохранен: {input_path}")
        
        # Получаем режим обработки
        mode = request.form.get('mode', 'light')
        
        # Обрабатываем изображение в зависимости от режима
        if mode == 'pro':
            result_filename = process_image(input_path, app.config['RESULT_FOLDER'])
        else:  # light mode
            result_filename = process_image_locally(input_path, app.config['RESULT_FOLDER'])
        
        if result_filename:
            result_url = f'/results/{result_filename}'
            return jsonify({
                'message': 'Изображение успешно обработано!',
                'result_url': result_url
            })
        else:
            error_msg = 'Ошибка при обработке изображения'
            if mode == 'pro':
                error_msg = 'Ошибка при обработке изображения через AI. Попробуйте использовать Light версию.'
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_result(filename):
    try:
        return send_from_directory(app.config['RESULT_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Ошибка при получении результата: {str(e)}", exc_info=True)
        return jsonify({'error': 'Файл не найден'}), 404

def get_available_port(start_port, max_attempts=10):
    """Находит свободный порт, начиная с указанного"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError(f"Не удалось найти свободный порт в диапазоне {start_port}-{start_port + max_attempts - 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Запуск сервера для создания раскрасок')
    parser.add_argument('--port', type=int, default=8080, help='Порт для запуска сервера (по умолчанию: 8080)')
    args = parser.parse_args()
    
    try:
        port = get_available_port(args.port)
        logger.info(f"Starting server at http://127.0.0.1:{port}")
        app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}", exc_info=True) 