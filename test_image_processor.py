import os
from image_processor import create_coloring_page
import unittest
import cv2
import numpy as np
from image_processor import classify_image, ImageType, get_processing_params, process_image_locally

def test_image_processing():
    # Путь к тестовому изображению
    test_image = "uploads/photo_2025-05-05 14.04.48.jpeg"
    output_image = "test_output.jpg"
    
    # Проверяем существование тестового изображения
    if not os.path.exists(test_image):
        print(f"Ошибка: файл {test_image} не найден")
        print("Пожалуйста, создайте тестовое изображение перед запуском теста")
        return False
    
    # Пробуем создать раскраску
    success = create_coloring_page(test_image, output_image)
    
    if success:
        print(f"✅ Раскраска успешно создана и сохранена как {output_image}")
        if os.path.exists(output_image):
            print(f"✅ Размер выходного файла: {os.path.getsize(output_image)} байт")
    else:
        print("❌ Ошибка при создании раскраски")
    
    return success

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Создаем тестовые изображения разных типов
        self.test_size = (100, 100, 3)
        
        # Создаем мультяшное изображение (яркие цвета, четкие контуры)
        self.cartoon_image = np.ones(self.test_size, dtype=np.uint8) * 255
        cv2.rectangle(self.cartoon_image, (20, 20), (80, 80), (0, 0, 255), -1)
        cv2.circle(self.cartoon_image, (50, 50), 20, (255, 0, 0), -1)
        
        # Создаем фото-подобное изображение (плавные переходы)
        self.photo_image = np.zeros(self.test_size, dtype=np.uint8)
        for i in range(100):
            for j in range(100):
                self.photo_image[i,j] = [(i+j)//2, (i+j)//2, (i+j)//2]
        
        # Создаем скетч (черно-белый, тонкие линии)
        self.sketch_image = np.ones(self.test_size, dtype=np.uint8) * 255
        cv2.line(self.sketch_image, (20, 20), (80, 80), (0, 0, 0), 1)
        cv2.line(self.sketch_image, (20, 80), (80, 20), (0, 0, 0), 1)
        
        # Создаем логотип (минимум деталей)
        self.logo_image = np.ones(self.test_size, dtype=np.uint8) * 255
        cv2.rectangle(self.logo_image, (30, 30), (70, 70), (0, 0, 0), -1)

    def test_classify_image(self):
        """Проверяем корректность классификации разных типов изображений"""
        self.assertEqual(classify_image(self.cartoon_image), ImageType.CARTOON)
        self.assertEqual(classify_image(self.photo_image), ImageType.PHOTO)
        self.assertEqual(classify_image(self.sketch_image), ImageType.SKETCH)
        self.assertEqual(classify_image(self.logo_image), ImageType.LOGO)

    def test_get_processing_params(self):
        """Проверяем корректность параметров для разных типов изображений"""
        for image_type in ImageType:
            params = get_processing_params(image_type)
            self.assertIsInstance(params, dict)
            self.assertTrue(all(key in params for key in [
                'blur_kernel',
                'clahe_clip_limit',
                'canny_low',
                'canny_high',
                'min_contour_area',
                'contour_simplification'
            ]))

    def test_process_image(self):
        """Проверяем полный процесс обработки изображения"""
        # Создаем временные директории для теста
        os.makedirs("test_uploads", exist_ok=True)
        os.makedirs("test_results", exist_ok=True)
        
        # Сохраняем тестовое изображение
        test_input = "test_uploads/test_cartoon.jpg"
        cv2.imwrite(test_input, self.cartoon_image)
        
        # Обрабатываем изображение
        result = process_image_locally(test_input, "test_results")
        
        # Проверяем результат
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(os.path.join("test_results", result)))
        
        # Очищаем тестовые файлы
        os.remove(test_input)
        os.remove(os.path.join("test_results", result))
        os.rmdir("test_uploads")
        os.rmdir("test_results")

if __name__ == "__main__":
    print("Начинаем тестирование обработки изображений...")
    test_image_processing()
    unittest.main() 