import cv2
import numpy as np
from skimage import feature, color, exposure
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.processing_params = {}
    
    def analyze_image(self, image):
        """
        Комплексный анализ изображения с использованием различных метрик
        и определение оптимальных параметров обработки
        """
        try:
            # Конвертируем в различные цветовые пространства
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Базовые метрики
            self.metrics['color_diversity'] = self._analyze_color_diversity(image)
            self.metrics['edge_density'] = self._analyze_edge_density(gray)
            self.metrics['texture_complexity'] = self._analyze_texture(gray)
            self.metrics['gradient_strength'] = self._analyze_gradients(lab)
            self.metrics['color_clusters'] = self._normalize_clusters(self._analyze_color_clusters(image))
            self.metrics['edge_continuity'] = self._analyze_edge_continuity(gray)
            
            # Дополнительные метрики
            self.metrics['detail_density'] = self._analyze_detail_density(gray)
            self.metrics['contrast_level'] = self._analyze_contrast(lab)
            self.metrics['noise_level'] = self._analyze_noise(gray)
            self.metrics['saturation_variance'] = self._analyze_saturation(hsv)
            
            # Определяем оптимальные параметры обработки
            self._determine_processing_params()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error during image analysis: {str(e)}")
            return None
    
    def _analyze_color_diversity(self, image):
        """Анализ разнообразия цветов"""
        # Квантуем цвета для уменьшения влияния шума
        quantized = image // 32 * 32
        pixels = quantized.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        max_colors = 256 * 256 * 256 / (32 * 32 * 32)  # максимальное количество цветов после квантования
        return len(unique_colors) / max_colors
    
    def _analyze_edge_density(self, gray):
        """Анализ плотности краев"""
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    def _analyze_texture(self, gray):
        """Анализ текстуры с использованием GLCM"""
        glcm = feature.graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        # Нормализуем значения
        max_contrast = 100  # эмпирическое максимальное значение
        normalized_contrast = min(contrast / max_contrast, 1.0)
        return (normalized_contrast + homogeneity) / 2
    
    def _analyze_gradients(self, lab):
        """Анализ градиентов в пространстве LAB"""
        l_channel = lab[:,:,0]
        gradient_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude) / 255.0  # нормализуем
    
    def _analyze_color_clusters(self, image, n_clusters=5):
        """Кластеризация цветов"""
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pixels)
        return kmeans.inertia_
    
    def _normalize_clusters(self, inertia):
        """Нормализация значения кластеризации"""
        # Эмпирически подобранные значения для нормализации
        max_inertia = 1e8
        return 1 - min(inertia / max_inertia, 1.0)
    
    def _analyze_edge_continuity(self, gray):
        """Анализ непрерывности краев"""
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        if np.sum(dilated) == 0:
            return 0
        return np.sum(edges) / np.sum(dilated)
    
    def _analyze_detail_density(self, gray):
        """Анализ плотности мелких деталей"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        detail_mask = np.absolute(laplacian) > 20
        return np.sum(detail_mask) / (gray.shape[0] * gray.shape[1])
    
    def _analyze_contrast(self, lab):
        """Анализ контраста в пространстве LAB"""
        l_channel = lab[:,:,0]
        return (np.percentile(l_channel, 95) - np.percentile(l_channel, 5)) / 255.0
    
    def _analyze_noise(self, gray):
        """Оценка уровня шума"""
        # Используем разницу между оригиналом и сглаженным изображением
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        noise = cv2.absdiff(gray, blurred)
        return np.mean(noise) / 255.0
    
    def _analyze_saturation(self, hsv):
        """Анализ вариации насыщенности"""
        s_channel = hsv[:,:,1]
        return np.std(s_channel) / 255.0
    
    def _determine_processing_params(self):
        """
        Определение оптимальных параметров обработки на основе метрик
        """
        # Параметры размытия
        self.processing_params['bilateral_d'] = int(9 + self.metrics['noise_level'] * 4)
        self.processing_params['bilateral_sigma'] = int(50 + self.metrics['noise_level'] * 50)
        
        # Параметры выделения краев
        base_threshold = 100
        self.processing_params['canny_low'] = int(base_threshold * 
            (0.3 + self.metrics['contrast_level'] * 0.4))
        self.processing_params['canny_high'] = int(base_threshold * 
            (0.6 + self.metrics['contrast_level'] * 0.8))
        
        # Параметры морфологических операций
        self.processing_params['morph_kernel_size'] = 2
        if self.metrics['detail_density'] < 0.1:
            self.processing_params['morph_kernel_size'] = 3
        
        # Параметры контуров
        self.processing_params['min_contour_area'] = int(20 * 
            (1 + self.metrics['detail_density'] * 2))
        self.processing_params['contour_epsilon'] = 0.002
        if self.metrics['edge_continuity'] < 0.3:
            self.processing_params['contour_epsilon'] = 0.001
        
        # Параметры улучшения деталей
        self.processing_params['detail_threshold'] = int(20 + 
            self.metrics['texture_complexity'] * 20)
        
        # Параметры финальной обработки
        self.processing_params['final_blur_size'] = 3
        if self.metrics['noise_level'] > 0.1:
            self.processing_params['final_blur_size'] = 5
    
    def get_processing_params(self):
        """Возвращает оптимальные параметры обработки"""
        return self.processing_params
    
    def get_image_complexity_score(self):
        """
        Вычисление общего показателя сложности изображения
        """
        if not self.metrics:
            return 0
            
        weights = {
            'color_diversity': 0.15,
            'edge_density': 0.15,
            'texture_complexity': 0.15,
            'gradient_strength': 0.1,
            'color_clusters': 0.1,
            'edge_continuity': 0.1,
            'detail_density': 0.1,
            'contrast_level': 0.05,
            'noise_level': 0.05,
            'saturation_variance': 0.05
        }
        
        score = sum(self.metrics[metric] * weight 
                   for metric, weight in weights.items() 
                   if metric in self.metrics)
        
        return min(score * 100, 100)  # нормализация до 100% 