import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from .image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)

class ImageType(Enum):
    """Enum for different types of images that require specific processing parameters"""
    CARTOON = "cartoon"
    PHOTO = "photo"
    SKETCH = "sketch"
    LOGO = "logo"

def classify_image(image: np.ndarray) -> ImageType:
    """
    Determines image type for optimal processing parameters selection.
    
    Uses multiple image analysis techniques:
    1. Color variety analysis using HSV color space
    2. Edge density analysis using Canny edge detection
    3. Texture analysis using Laplacian
    4. Gradient analysis using Sobel operators
    5. Color count analysis
    6. Edge continuity analysis
    
    Args:
        image: Input image in BGR format
    
    Returns:
        ImageType: Classified image type
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Color variety analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    color_variety = np.std(hist_h) + np.std(hist_s)
    
    # 2. Edge density analysis
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # 3. Texture analysis
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(texture)
    
    # 4. Gradient analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_magnitude)
    
    # 5. Color count analysis
    unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
    color_count_ratio = unique_colors / (image.shape[0] * image.shape[1])
    
    # 6. Edge continuity analysis
    edge_continuity = cv2.connectedComponents(edges)[0] / np.count_nonzero(edges) if np.count_nonzero(edges) > 0 else 0
    
    # 7. Additional metrics
    # Calculate color clusters for better cartoon detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(image.reshape(-1, 3).astype(np.float32), 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    color_clusters = len(centers)
    
    # Calculate edge sharpness for logo detection
    edge_sharpness = np.mean(gradient_magnitude[edges > 0]) if np.count_nonzero(edges) > 0 else 0
    
    logger.debug(f"Image analysis: color_variety={color_variety:.2f}, edge_density={edge_density:.2f}, "
                f"texture_variance={texture_variance:.2f}, gradient_mean={gradient_mean:.2f}, "
                f"color_count_ratio={color_count_ratio:.4f}, edge_continuity={edge_continuity:.4f}, "
                f"color_clusters={color_clusters}, edge_sharpness={edge_sharpness:.2f}")
    
    # Classification based on characteristics
    # LOGO: Very few color clusters, sharp edges, high edge sharpness
    if (color_clusters < 4 and edge_density < 0.05 and 
        edge_sharpness > 100 and color_count_ratio < 0.01):
        return ImageType.LOGO
    
    # SKETCH: High edge density, low color variety, high edge continuity
    elif (edge_density > 0.08 and color_variety < 5000 and 
          edge_continuity < 0.02 and color_clusters < 3):
        return ImageType.SKETCH
    
    # CARTOON: Distinct color clusters, medium edge density
    elif (4 <= color_clusters <= 8 and 0.02 <= edge_density <= 0.08 and 
          color_variety > 8000):
        return ImageType.CARTOON
    
    # PHOTO: High color variety, smooth gradients, many unique colors
    else:
        return ImageType.PHOTO

def get_processing_params(image_type: ImageType) -> Dict[str, Any]:
    """
    Returns optimal processing parameters for each image type.
    
    Parameters are tuned for:
    - Edge detection sensitivity
    - Contour processing
    - Detail preservation
    - Face and edge emphasis
    
    Args:
        image_type: Type of the image to process
    
    Returns:
        Dictionary with processing parameters
    """
    params = {
        ImageType.CARTOON: {
            'blur_kernel': 3,
            'clahe_clip_limit': 2.0,
            'canny_low': 100,
            'canny_high': 200,
            'min_contour_area': 50,
            'contour_simplification': 0.01,
            'detail_threshold': 30,
            'face_weight': 1.5
        },
        ImageType.PHOTO: {
            'blur_kernel': 5,
            'clahe_clip_limit': 3.0,
            'canny_low': 50,
            'canny_high': 150,
            'min_contour_area': 100,
            'contour_simplification': 0.02,
            'detail_threshold': 40,
            'face_weight': 1.8
        },
        ImageType.SKETCH: {
            'blur_kernel': 3,
            'clahe_clip_limit': 2.0,
            'canny_low': 30,
            'canny_high': 100,
            'min_contour_area': 20,
            'contour_simplification': 0.005,
            'detail_threshold': 20,
            'face_weight': 1.3
        },
        ImageType.LOGO: {
            'blur_kernel': 3,
            'clahe_clip_limit': 1.0,
            'canny_low': 100,
            'canny_high': 200,
            'min_contour_area': 200,
            'contour_simplification': 0.05,
            'detail_threshold': 50,
            'face_weight': 1.2
        }
    }
    return params[image_type]

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhances image before processing using multiple techniques:
    1. LAB color space conversion for better contrast processing
    2. CLAHE for adaptive histogram equalization
    3. Sharpening filter application
    
    Args:
        image: Input image in BGR format
    
    Returns:
        Enhanced image
    """
    # Convert to Lab color space for better contrast processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Increase sharpness
    kernel_sharpening = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpening)
    
    return enhanced

def post_process_lines(image: np.ndarray) -> np.ndarray:
    """
    Improves line quality after main processing:
    1. Removes small artifacts
    2. Connects nearby lines
    3. Thins and strengthens lines
    
    Args:
        image: Binary image with lines
    
    Returns:
        Processed image with improved lines
    """
    # Remove small details
    kernel = np.ones((2,2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Connect nearby lines
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Thin lines using erosion and dilation
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.dilate(image, kernel, iterations=1)
    
    # Make lines more prominent
    image = cv2.dilate(image, np.ones((2,2), np.uint8), iterations=1)
    
    return image

def create_coloring_page(image):
    """
    Создает раскраску из изображения
    Args:
        image: numpy.ndarray - входное изображение в формате BGR
    Returns:
        numpy.ndarray - изображение раскраски в формате BGR
    """
    try:
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Применяем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Находим края
        edges = cv2.Canny(binary, 100, 200)
        
        # Расширяем края для лучшей видимости
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Инвертируем изображение (черные линии на белом фоне)
        result = cv2.bitwise_not(dilated)
        
        # Преобразуем в трехканальное изображение
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating coloring page: {str(e)}")
        return None

def process_image(image, image_type):
    """
    Обрабатывает изображение в зависимости от его типа
    Args:
        image: numpy.ndarray - входное изображение в формате BGR
        image_type: ImageType - тип изображения
    Returns:
        numpy.ndarray - обработанное изображение в формате BGR
    """
    try:
        # Получаем параметры обработки
        params = get_processing_params(image_type)
        
        # Применяем размытие
        blurred = cv2.GaussianBlur(image, params['blur_kernel'], 0)
        
        # Применяем CLAHE для улучшения контраста
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'])
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Находим края
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(
            gray,
            params['canny_low'],
            params['canny_high']
        )
        
        # Находим и фильтруем контуры
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        filtered_contours = []
        for contour in contours:
            # Фильтруем по площади
            if cv2.contourArea(contour) > params['min_contour_area']:
                # Упрощаем контур
                epsilon = params['contour_simplification'] * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                filtered_contours.append(approx)
        
        # Создаем пустое изображение для результата
        result = np.ones_like(image) * 255
        
        # Рисуем контуры
        cv2.drawContours(result, filtered_contours, -1, (0, 0, 0), 2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
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
    Создает раскраску из изображения используя комбинацию Pillow и OpenCV
    с адаптивными параметрами обработки и улучшенным сохранением деталей.
    
    Args:
        input_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения результата
    
    Returns:
        str: Имя файла созданной раскраски или None в случае ошибки
    """
    try:
        logger.debug(f"Начало локальной обработки изображения: {input_path}")
        
        # Открываем изображение с помощью Pillow
        pil_image = Image.open(input_path)
        
        # Изменяем размер изображения для лучшей обработки
        max_dimension = 1500
        if max(pil_image.size) > max_dimension:
            ratio = max_dimension / max(pil_image.size)
            new_size = tuple(int(dim * ratio) for dim in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Конвертируем в массив numpy для OpenCV
        cv_image = np.array(pil_image)
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Анализируем изображение для определения оптимальных параметров
        analyzer = ImageAnalyzer()
        analyzer.analyze_image(cv_image)
        params = analyzer.get_processing_params()
        
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        
        # 1. Предварительная обработка с адаптивными параметрами
        # Билатеральная фильтрация с параметрами на основе уровня шума
        denoised = cv2.bilateralFilter(
            gray,
            params['bilateral_d'],
            params['bilateral_sigma'],
            params['bilateral_sigma']
        )
        
        # 2. Многомасштабное выделение деталей
        details_masks = []
        scales = [0.5, 1.0, 2.0]  # Разные масштабы для анализа деталей
        
        for scale in scales:
            # Масштабируем изображение
            if scale != 1.0:
                scaled_size = tuple(int(dim * scale) for dim in denoised.shape[:2][::-1])
                scaled = cv2.resize(denoised, scaled_size, interpolation=cv2.INTER_LINEAR)
            else:
                scaled = denoised
            
            # Находим детали на текущем масштабе
            laplacian = cv2.Laplacian(scaled, cv2.CV_64F)
            abs_laplacian = np.uint8(np.absolute(laplacian))
            
            # Адаптивная бинаризация для выделения деталей
            local_threshold = cv2.adaptiveThreshold(
                abs_laplacian,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Возвращаем к исходному размеру
            if scale != 1.0:
                local_threshold = cv2.resize(
                    local_threshold,
                    (denoised.shape[1], denoised.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            details_masks.append(local_threshold)
        
        # Объединяем маски деталей
        combined_details = np.zeros_like(denoised)
        for mask in details_masks:
            combined_details = cv2.bitwise_or(combined_details, mask)
        
        # 3. Адаптивное выравнивание гистограммы
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Улучшение краев с адаптивными порогами
        edges = cv2.Canny(
            enhanced,
            params['canny_low'],
            params['canny_high']
        )
        
        # Комбинируем края и детали
        combined_edges = cv2.bitwise_or(edges, combined_details)
        
        # 5. Морфологические операции для улучшения линий
        kernel_size = params['morph_kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(combined_edges, kernel, iterations=1)
        
        # 6. Находим и фильтруем контуры с сохранением деталей
        contours, hierarchy = cv2.findContours(
            dilated,
            cv2.RETR_TREE,  # Используем иерархию для лучшего сохранения деталей
            cv2.CHAIN_APPROX_TC89_KCOS  # Более точное приближение контуров
        )
        
        # Создаем маску для рисования контуров
        result = np.ones_like(gray) * 255
        
        # Рисуем контуры с адаптивными параметрами
        min_contour_area = params['min_contour_area'] * 0.5  # Уменьшаем порог для сохранения деталей
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                # Адаптивное сглаживание контура
                epsilon = params['contour_epsilon'] * cv2.arcLength(contour, True)
                if hierarchy[0][i][3] != -1:  # Если это внутренний контур
                    epsilon *= 0.5  # Уменьшаем сглаживание для деталей
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(result, [approx], -1, (0), 1)  # Уменьшаем толщину линий
        
        # 7. Добавляем мелкие детали
        # Находим очень мелкие детали, которые могли быть пропущены
        fine_details = cv2.Laplacian(enhanced, cv2.CV_64F)
        fine_details = np.uint8(np.absolute(fine_details))
        _, details_thresh = cv2.threshold(
            fine_details,
            params['detail_threshold'] * 0.7,  # Уменьшаем порог для деталей
            255,
            cv2.THRESH_BINARY
        )
        
        # Добавляем мелкие детали к результату
        result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(details_thresh))
        
        # 8. Финальная обработка
        # Аккуратное удаление шума с сохранением деталей
        result = cv2.medianBlur(result, params['final_blur_size'])
        
        # Улучшаем четкость линий
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel)
        
        # Создаем имя файла для раскраски
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Сохраняем результат
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.debug(f"Раскраска успешно создана: {output_path}")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Ошибка при локальной обработке изображения: {str(e)}", exc_info=True)
        return None 