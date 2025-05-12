from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)

# Создаем директорию для загруженных файлов
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Максимальный размер файла 16MB

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'Файл не был загружен', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'Файл не был выбран', 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # TODO: Здесь будет логика обработки изображения
        return "Изображение успешно загружено и обработано!"

@app.route('/generate', methods=['POST'])
def generate_image():
    prompt = request.form.get('prompt', '')
    if not prompt:
        return 'Не указан текст для генерации', 400
    
    # TODO: Здесь будет логика генерации раскраски
    return "Генерация раскраски по запросу завершена!"

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8080")
    app.run(host='127.0.0.1', port=8080, debug=True) 