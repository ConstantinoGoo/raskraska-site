import os
from openai import OpenAI, OpenAIError
import base64
from PIL import Image
import io
import logging

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

class AIImageProcessor:
    def __init__(self):
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("API ключ OpenAI не найден в переменных окружения")
            if not api_key.startswith('sk-'):
                raise ValueError("Некорректный формат API ключа OpenAI")
            self.client = OpenAI(api_key=api_key)
            logger.debug("OpenAI клиент успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка при инициализации OpenAI клиента: {e}")
            raise

    def validate_image(self, image_path):
        """Проверяет изображение на соответствие требованиям API"""
        try:
            with Image.open(image_path) as img:
                # Проверяем формат
                if img.format not in ['PNG', 'JPEG']:
                    raise ValueError(f"Неподдерживаемый формат изображения: {img.format}")
                
                # Проверяем размер файла
                file_size = os.path.getsize(image_path)
                if file_size > 4 * 1024 * 1024:  # 4MB
                    raise ValueError("Размер файла превышает 4MB")
                
                # Проверяем размеры изображения
                width, height = img.size
                if width > 2048 or height > 2048:
                    raise ValueError("Размеры изображения превышают 2048x2048")
                
                return True
        except Exception as e:
            logger.error(f"Ошибка при валидации изображения: {e}")
            raise

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
            logger.debug(f"Начинаю обработку изображения: {image_path}")
            
            # Проверяем существование файла
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            
            # Валидируем изображение
            self.validate_image(image_path)
            
            # Подготавливаем изображение
            with Image.open(image_path) as img:
                # Конвертируем в RGB если нужно
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Сохраняем во временный буфер
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
            
            # Создаем запрос к DALL-E
            logger.debug("Отправляю запрос к DALL-E API")
            try:
                response = self.client.images.create_variation(
                    image=img_byte_arr,
                    n=1,
                    size="1024x1024",
                    response_format="url"
                )
                
                logger.debug("Получен ответ от DALL-E API")
                
                # Получаем URL сгенерированного изображения
                image_url = response.data[0].url
                logger.debug(f"Получен URL результата: {image_url}")
                
                # Скачиваем изображение
                import requests
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Сохраняем результат
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.debug(f"Результат сохранен в: {output_path}")
                
                return True
                
            except OpenAIError as e:
                logger.error(f"Ошибка API OpenAI: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения через AI: {str(e)}", exc_info=True)
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
        logger.debug(f"Начало обработки изображения: {input_path}")
        
        # Создаем имя файла для раскраски
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.debug(f"Путь для сохранения результата: {output_path}")
        
        # Создаем раскраску
        processor = AIImageProcessor()
        if processor.create_coloring_page(input_path, output_path):
            logger.debug("Обработка успешно завершена")
            return output_filename
            
        logger.error("Не удалось создать раскраску")
        return None
        
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}", exc_info=True)
        return None 