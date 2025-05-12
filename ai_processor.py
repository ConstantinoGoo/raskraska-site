import os
from openai import OpenAI
import base64
from PIL import Image
import io

class AIImageProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def create_coloring_page(self, image_path, output_path):
        """
        Создает раскраску из изображения используя OpenAI DALL-E.
        
        Args:
            image_path (str): Путь к исходному изображению
            output_path (str): Путь для сохранения раскраски
        
        Returns:
            bool: True если преобразование успешно, False в случае ошибки
        """
        try:
            # Читаем и кодируем изображение в base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Создаем запрос к DALL-E
            response = self.client.images.edit(
                image=open(image_path, "rb"),
                prompt="Convert this image into a black and white coloring book page. Make clear, bold outlines suitable for coloring. Keep important details but remove unnecessary elements. Make it look like a professional coloring book illustration.",
                n=1,
                size="1024x1024"
            )

            # Получаем URL сгенерированного изображения
            image_url = response.data[0].url

            # Скачиваем изображение
            import requests
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            
            # Сохраняем результат
            img.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Ошибка при обработке изображения через AI: {e}")
            return False

def process_image(input_path, output_dir):
    """
    Обрабатывает изображение и создает раскраску.
    
    Args:
        input_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения результата
    
    Returns:
        str: Имя файла созданной раскраски или None в случае ошибки
    """
    try:
        # Создаем имя файла для раскраски
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Создаем раскраску
        processor = AIImageProcessor()
        if processor.create_coloring_page(input_path, output_path):
            return output_filename
        return None
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None 