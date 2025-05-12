import os
import cv2
import numpy as np

def generate_test_samples():
    """Генерация тестовых изображений разных типов"""
    samples_dir = os.path.join('tests', 'data', 'test_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Создаем мультяшное изображение
    cartoon = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(cartoon, (100, 100), 50, (0, 255, 255), -1)
    cv2.circle(cartoon, (80, 80), 10, (0, 0, 0), -1)
    cv2.circle(cartoon, (120, 80), 10, (0, 0, 0), -1)
    cv2.ellipse(cartoon, (100, 120), (30, 20), 0, 0, 180, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(samples_dir, 'cartoon1.jpg'), cartoon)
    
    # Создаем фотографию
    photo = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    cv2.GaussianBlur(photo, (15, 15), 0, photo)
    cv2.imwrite(os.path.join(samples_dir, 'photo1.jpg'), photo)
    
    # Создаем скетч
    sketch = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.line(sketch, (50, 50), (150, 150), (0, 0, 0), 2)
    cv2.line(sketch, (150, 50), (50, 150), (0, 0, 0), 2)
    cv2.rectangle(sketch, (25, 25), (175, 175), (0, 0, 0), 2)
    cv2.imwrite(os.path.join(samples_dir, 'sketch1.jpg'), sketch)
    
    # Создаем логотип
    logo = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.putText(logo, 'LOGO', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.rectangle(logo, (30, 30), (170, 170), (0, 0, 0), 3)
    cv2.imwrite(os.path.join(samples_dir, 'logo1.png'), logo)

def generate_test_images():
    """Генерация тестовых изображений для обработки"""
    images_dir = os.path.join('tests', 'data', 'test_images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Создаем изображение с градиентом и фигурами
    gradient = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        gradient[:, i] = [i * 255 // 300, i * 255 // 300, i * 255 // 300]
    
    # Добавляем фигуры
    cv2.circle(gradient, (150, 150), 100, (255, 0, 0), 3)
    cv2.rectangle(gradient, (50, 50), (250, 250), (0, 255, 0), 2)
    cv2.line(gradient, (0, 0), (300, 300), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(images_dir, 'jessica.jpeg'), gradient)
    
    # Создаем изображение с текстурой
    texture = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(0, 300, 20):
        for j in range(0, 300, 20):
            color = np.random.randint(0, 255, 3)
            cv2.rectangle(texture, (i, j), (i+10, j+10), color.tolist(), -1)
    cv2.imwrite(os.path.join(images_dir, 'spider-man.jpeg'), texture)

if __name__ == '__main__':
    print("Generating test data...")
    generate_test_samples()
    generate_test_images()
    print("Test data generated successfully!") 