from flask import Blueprint, render_template, request, send_from_directory, jsonify, current_app, url_for
from werkzeug.utils import secure_filename
import os
import logging
from app.processors.ai_processor import process_image
from app.processors.image_processor import process_image_locally
from typing import Union, Tuple
import base64
from PIL import Image
import io
import cv2
import numpy as np
from .processors import create_coloring_page, ImageType, ImageAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
main = Blueprint('main', __name__)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def create_thumbnail(image_path: str, max_size: tuple = (300, 300)) -> str:
    """Create a base64 thumbnail of an image"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or 'JPEG')
            return f"data:image/{img.format.lower() if img.format else 'jpeg'};base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        return ""

@main.route('/')
def index() -> str:
    """Render the main page"""
    return render_template('index.html')

@main.route('/process', methods=['POST'])
def process():
    """Обработка загруженного изображения"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Читаем изображение
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
            
        # Изменяем размер если нужно
        height, width = image.shape[:2]
        max_width, max_height = current_app.config['MAX_IMAGE_SIZE']
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Анализируем изображение
        analyzer = ImageAnalyzer()
        metrics = analyzer.analyze_image(image)
        complexity_score = analyzer.get_image_complexity_score()
        
        # Создаем раскраску
        result = create_coloring_page(image)
        if result is None:
            return jsonify({'error': 'Error creating coloring page'}), 500
            
        # Сохраняем результат
        filename = os.path.splitext(file.filename)[0]
        output_filename = f"{filename}_coloring.jpg"
        output_path = os.path.join(current_app.config['RESULTS_FOLDER'], output_filename)
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, current_app.config['JPEG_QUALITY']])
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'metrics': metrics,
            'complexity_score': complexity_score
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@main.route('/results/<filename>')
def results(filename):
    """Отдает обработанное изображение"""
    return send_from_directory(current_app.config['RESULTS_FOLDER'], filename)

@main.route('/upload', methods=['POST'])
def upload_image() -> Tuple[str, int]:
    """Handle image upload and processing"""
    try:
        # Validate file presence
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не загружен'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Неверный формат файла. Разрешены только PNG и JPEG'}), 400
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        logger.debug(f"File saved: {input_path}")
        
        # Create thumbnail of original image
        original_thumbnail = create_thumbnail(input_path)
        
        # Get processing mode and process image
        mode = request.form.get('mode', 'light')
        result_filename = (
            process_image(input_path, current_app.config['RESULTS_FOLDER'])
            if mode == 'pro'
            else process_image_locally(input_path, current_app.config['RESULTS_FOLDER'])
        )
        
        if result_filename:
            result_path = os.path.join(current_app.config['RESULTS_FOLDER'], result_filename)
            result_thumbnail = create_thumbnail(result_path)
            
            return jsonify({
                'message': 'Изображение успешно обработано!',
                'result_url': url_for('main.get_result', filename=result_filename),
                'original_thumbnail': original_thumbnail,
                'result_thumbnail': result_thumbnail
            }), 200
        else:
            error_msg = (
                'Ошибка обработки изображения через ИИ. Попробуйте быстрый режим.'
                if mode == 'pro'
                else 'Ошибка обработки изображения'
            )
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@main.route('/results/<filename>')
def get_result(filename: str) -> Union[str, Tuple[str, int]]:
    """Serve processed image results"""
    try:
        return send_from_directory(
            os.path.abspath(current_app.config['RESULTS_FOLDER']), 
            filename,
            as_attachment=True
        )
    except Exception as e:
        logger.error(f"Error serving result file: {str(e)}", exc_info=True)
        return jsonify({'error': 'Файл не найден'}), 404 