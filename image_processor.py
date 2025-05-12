import cv2
import numpy as np
from PIL import Image
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ImageType(Enum):
    CARTOON = "cartoon"
    PHOTO = "photo"
    SKETCH = "sketch"
    LOGO = "logo"

def classify_image(image):
    """
    Определяет тип изображения для выбора оптимальных параметров обработки.
    
    Args:
        image: numpy.ndarray - входное изображение в формате BGR
    
    Returns:
        ImageType: тип изображения
    """
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Анализ цветового разнообразия
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    color_variety = np.std(hist_h) + np.std(hist_s)
    
    # 2. Анализ краев
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # 3. Анализ текстуры
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(texture)
    
    # 4. Анализ градиентов
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_magnitude)
    
    logger.debug(f"Image analysis: color_variety={color_variety:.2f}, edge_density={edge_density:.2f}, "
                f"texture_variance={texture_variance:.2f}, gradient_mean={gradient_mean:.2f}")
    
    # Классификация на основе характеристик
    if edge_density < 0.01 and texture_variance < 100:
        return ImageType.LOGO
    elif color_variety > 1000 and edge_density < 0.1:
        return ImageType.PHOTO
    elif texture_variance < 500 and color_variety < 500:
        return ImageType.SKETCH
    else:
        return ImageType.CARTOON

def get_processing_params(image_type):
    """
    Возвращает оптимальные параметры обработки для каждого типа изображения.
    
    Args:
        image_type: ImageType - тип изображения
    
    Returns:
        dict: параметры обработки
    """
    params = {
        ImageType.CARTOON: {
            'blur_kernel': 5,
            'clahe_clip_limit': 2.5,
            'canny_low': 40,  # Уменьшили для лучшего захвата деталей
            'canny_high': 180,  # Увеличили для более четкого выделения основных контуров
            'min_contour_area': 20,  # Уменьшили для сохранения мелких деталей
            'contour_simplification': 0.0015,  # Уточнили для лучшей детализации
            'detail_threshold': 100,  # Порог для определения важных деталей
            'face_weight': 1.2,  # Коэффициент важности для области лица
            'edge_weight': 0.8  # Коэффициент важности для краев одежды
        },
        ImageType.PHOTO: {
            'blur_kernel': 7,
            'clahe_clip_limit': 3.0,
            'canny_low': 30,
            'canny_high': 200,
            'min_contour_area': 50,
            'contour_simplification': 0.001,
            'detail_threshold': 150,
            'face_weight': 1.0,
            'edge_weight': 1.0
        },
        ImageType.SKETCH: {
            'blur_kernel': 3,
            'clahe_clip_limit': 2.0,
            'canny_low': 70,
            'canny_high': 140,
            'min_contour_area': 20,
            'contour_simplification': 0.003,
            'detail_threshold': 80,
            'face_weight': 1.0,
            'edge_weight': 1.0
        },
        ImageType.LOGO: {
            'blur_kernel': 3,
            'clahe_clip_limit': 1.5,
            'canny_low': 100,
            'canny_high': 120,
            'min_contour_area': 10,
            'contour_simplification': 0.004,
            'detail_threshold': 200,
            'face_weight': 1.0,
            'edge_weight': 1.0
        }
    }
    return params[image_type]

def enhance_image(image):
    """
    Улучшает изображение перед обработкой.
    """
    # Преобразование в Lab цветовое пространство для лучшей обработки контраста
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Объединяем каналы обратно
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Увеличиваем резкость
    kernel_sharpening = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpening)
    
    return enhanced

def post_process_lines(image):
    """
    Улучшает качество линий после основной обработки.
    """
    # Убираем мелкие детали
    kernel = np.ones((2,2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Соединяем близкие линии
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Утончаем линии с помощью эрозии и дилатации
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    
    # Делаем линии более заметными
    image = cv2.dilate(image, np.ones((2,2), np.uint8), iterations=1)
    
    return image

def create_coloring_page(image_path, output_path):
    """
    Преобразует изображение в раскраску используя улучшенный алгоритм.
    
    Args:
        image_path (str): Путь к исходному изображению
        output_path (str): Путь для сохранения раскраски
    
    Returns:
        bool: True если преобразование успешно, False в случае ошибки
    """
    try:
        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Изменяем размер изображения для лучшей обработки
        height, width = image.shape[:2]
        max_dimension = 1200
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Предварительная обработка
        enhanced = enhance_image(image)
        
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Применяем несколько методов выделения краев и комбинируем их
        # 1. Canny с автоматическим определением порогов
        median = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges_canny = cv2.Canny(gray, lower, upper)
        
        # 2. Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.uint8(np.absolute(sobelx) + np.absolute(sobely))
        
        # 3. Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges_laplacian = np.uint8(np.absolute(laplacian))
        
        # Комбинируем все края с разными весами
        edges = cv2.addWeighted(edges_canny, 0.5, edges_sobel, 0.3, 0)
        edges = cv2.addWeighted(edges, 1.0, edges_laplacian, 0.2, 0)
        
        # Применяем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Инвертируем изображение и применяем постобработку
        result = cv2.bitwise_not(binary)
        result = post_process_lines(result)
        
        # Находим и фильтруем контуры
        contours, _ = cv2.findContours(cv2.bitwise_not(result), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем чистое изображение для финального результата
        final_result = np.ones_like(result) * 255
        
        # Отрисовываем контуры с разной толщиной в зависимости от размера
        min_area = 20  # Минимальная площадь контура
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Определяем толщину линии в зависимости от размера контура
                thickness = 2 if area > 100 else 1
                cv2.drawContours(final_result, [contour], -1, (0, 0, 0), thickness)
        
        # Финальное сглаживание
        final_result = cv2.medianBlur(final_result, 3)
        
        # Сохраняем результат
        cv2.imwrite(output_path, final_result)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return False

def process_image(input_path, output_dir):
    """
    Обрабатывает изображение и создает раскраску.
    
    Args:
        input_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения результата
    
    Returns:
        str: Путь к созданной раскраске или None в случае ошибки
    """
    try:
        # Создаем имя файла для раскраски
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Создаем раскраску
        if create_coloring_page(input_path, output_path):
            return output_filename
        return None
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

def enhance_details(image, params):
    """
    Улучшает детали в важных областях изображения.
    
    Args:
        image: numpy.ndarray - входное изображение
        params: dict - параметры обработки
    
    Returns:
        numpy.ndarray: улучшенное изображение
    """
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Находим области с высокой детализацией
    detail_map = cv2.Laplacian(gray, cv2.CV_64F)
    detail_map = np.uint8(np.absolute(detail_map))
    
    # Выделяем области с деталями выше порога
    _, detail_mask = cv2.threshold(
        detail_map,
        params['detail_threshold'],
        255,
        cv2.THRESH_BINARY
    )
    
    # Расширяем маску для захвата окружающих областей
    detail_mask = cv2.dilate(detail_mask, np.ones((3,3), np.uint8), iterations=2)
    
    return detail_mask

def process_image_locally(input_path, output_dir):
    """
    Создает раскраску из изображения используя OpenCV.
    
    Args:
        input_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения результата
    
    Returns:
        str: Имя файла созданной раскраски или None в случае ошибки
    """
    try:
        logger.debug(f"Начало локальной обработки изображения: {input_path}")
        
        # Читаем изображение
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Не удалось прочитать изображение")
            
        # Изменяем размер изображения для лучшей обработки
        height, width = image.shape[:2]
        max_dimension = 1500
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Определяем тип изображения и получаем параметры
        image_type = classify_image(image)
        params = get_processing_params(image_type)
        logger.debug(f"Определен тип изображения: {image_type.value}")
        
        # Находим области с важными деталями
        detail_mask = enhance_details(image, params)
        
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Улучшаем контраст с учетом типа изображения
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'], tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Применяем комбинацию фильтров для лучшего удаления шума
        # Сначала медианный фильтр для удаления пятен
        median = cv2.medianBlur(gray, params['blur_kernel'])
        
        # Затем билатеральный для сохранения краев
        denoised = cv2.bilateralFilter(median, 9, 75, 75)
        
        # Находим края с параметрами в зависимости от типа изображения
        edges = cv2.Canny(denoised, params['canny_low'], params['canny_high'])
        
        # Усиливаем края в областях с важными деталями
        edges = cv2.addWeighted(
            edges, 1.0,
            cv2.bitwise_and(edges, detail_mask),
            params['face_weight'] - 1.0,
            0
        )
        
        # Улучшаем связность линий
        kernel_line = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel_line, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
        
        # Находим контуры с иерархией
        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        # Создаем чистое изображение для финального результата
        final_result = np.ones_like(edges) * 255
        
        if len(contours) > 0 and hierarchy is not None:
            hierarchy = hierarchy[0]
            
            # Предварительно анализируем все контуры
            contour_info = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > params['min_contour_area']:
                    perimeter = cv2.arcLength(contour, True)
                    complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
                    
                    # Проверяем, находится ли контур в области важных деталей
                    mask = np.zeros_like(detail_mask)
                    cv2.drawContours(mask, [contour], -1, (255,255,255), 1)
                    detail_overlap = cv2.bitwise_and(mask, detail_mask)
                    is_detail = np.count_nonzero(detail_overlap) > 0
                    
                    contour_info.append({
                        'index': i,
                        'area': area,
                        'perimeter': perimeter,
                        'complexity': complexity,
                        'hierarchy': hierarchy[i],
                        'is_detail': is_detail
                    })
            
            # Сортируем контуры по площади
            contour_info.sort(key=lambda x: x['area'], reverse=True)
            
            # Сначала рисуем основные контуры
            for info in contour_info:
                if info['area'] > 200:  # Основные контуры
                    contour = contours[info['index']]
                    # Настраиваем упрощение в зависимости от того, является ли контур деталью
                    simplification = params['contour_simplification']
                    if info['is_detail']:
                        simplification *= 0.5  # Уменьшаем упрощение для деталей
                    epsilon = simplification * info['perimeter']
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    cv2.drawContours(final_result, [approx], -1, (0, 0, 0), 2)
            
            # Затем добавляем важные детали
            for info in contour_info:
                if params['min_contour_area'] < info['area'] <= 200 and info['hierarchy'][3] != -1:
                    if info['is_detail']:  # Рисуем только если это важная деталь
                        contour = contours[info['index']]
                        epsilon = params['contour_simplification'] * 0.5 * info['perimeter']
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.drawContours(final_result, [approx], -1, (0, 0, 0), 1)
        
        # Финальная обработка
        kernel_close = np.ones((3,3), np.uint8)
        final_result = cv2.morphologyEx(final_result, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_clean = np.ones((2,2), np.uint8)
        final_result = cv2.morphologyEx(final_result, cv2.MORPH_OPEN, kernel_clean)
        
        # Создаем имя файла для раскраски
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Сохраняем результат
        cv2.imwrite(output_path, final_result)
        logger.debug(f"Раскраска успешно создана: {output_path}")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Ошибка при локальной обработке изображения: {str(e)}", exc_info=True)
        return None 