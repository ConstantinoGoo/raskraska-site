from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from ai_processor import process_image
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

app = Flask(__name__)

# Создаем директории для файлов
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла 16MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Файл не был загружен'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не был выбран'}), 400
    
    if file and allowed_file(file.filename):
        # Сохраняем оригинальное изображение
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(input_path)
        
        # Создаем раскраску используя AI
        result_filename = process_image(input_path, app.config['RESULT_FOLDER'])
        
        if result_filename:
            result_url = f'/results/{result_filename}'
            return jsonify({
                'message': 'Изображение успешно обработано!',
                'result_url': result_url
            })
        else:
            return jsonify({'error': 'Ошибка при обработке изображения'}), 500
    
    return jsonify({'error': 'Недопустимый формат файла'}), 400

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Не указан текст для генерации'}), 400
    
    # TODO: Здесь будет логика генерации раскраски
    return jsonify({'message': 'Генерация раскраски по запросу завершена!'})

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True) 