import pytest
import cv2
import numpy as np
from app.processors.image_analyzer import ImageAnalyzer

@pytest.fixture
def sample_image():
    """Создание тестового изображения с разнообразными элементами"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Добавляем градиент
    for i in range(100):
        for j in range(100):
            value = (i + j) // 2
            image[i, j] = [value, value, value]
    
    # Добавляем цветные области
    cv2.rectangle(image, (20, 20), (40, 40), (255, 0, 0), -1)
    cv2.circle(image, (70, 70), 15, (0, 255, 0), -1)
    cv2.line(image, (10, 90), (90, 10), (0, 0, 255), 2)
    
    return image

@pytest.fixture
def analyzer():
    """Создание экземпляра анализатора"""
    return ImageAnalyzer()

def test_color_diversity(analyzer, sample_image):
    """Тест анализа цветового разнообразия"""
    diversity = analyzer._analyze_color_diversity(sample_image)
    assert 0 <= diversity <= 1
    assert diversity > 0.01  # Должно быть некоторое разнообразие

def test_edge_density(analyzer, sample_image):
    """Тест анализа плотности краев"""
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    density = analyzer._analyze_edge_density(gray)
    assert 0 <= density <= 1
    assert density > 0  # Должны быть обнаружены края

def test_texture_analysis(analyzer, sample_image):
    """Тест текстурного анализа"""
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    texture = analyzer._analyze_texture(gray)
    assert texture > 0  # Должна быть некоторая текстура

def test_gradient_analysis(analyzer, sample_image):
    """Тест анализа градиентов"""
    lab = cv2.cvtColor(sample_image, cv2.COLOR_BGR2LAB)
    gradient = analyzer._analyze_gradients(lab)
    assert 0 <= gradient <= 1  # Нормализованное значение

def test_color_clusters(analyzer, sample_image):
    """Тест кластеризации цветов"""
    clusters = analyzer._analyze_color_clusters(sample_image)
    assert clusters > 0  # Должна быть некоторая вариация цветов
    
    normalized = analyzer._normalize_clusters(clusters)
    assert 0 <= normalized <= 1  # Проверяем нормализацию

def test_edge_continuity(analyzer, sample_image):
    """Тест непрерывности краев"""
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    continuity = analyzer._analyze_edge_continuity(gray)
    assert 0 <= continuity <= 1

def test_full_analysis(analyzer, sample_image):
    """Тест полного анализа изображения"""
    metrics = analyzer.analyze_image(sample_image)
    assert metrics is not None
    assert len(metrics) == 6  # Проверяем наличие всех метрик
    
    # Проверяем наличие всех ключей
    expected_keys = {
        'color_diversity', 'edge_density', 'texture_complexity',
        'gradient_strength', 'color_clusters', 'edge_continuity'
    }
    assert set(metrics.keys()) == expected_keys
    
    # Проверяем диапазоны значений
    for key, value in metrics.items():
        assert isinstance(value, (int, float))
        assert 0 <= value <= 1  # Все значения должны быть нормализованы

def test_complexity_score(analyzer, sample_image):
    """Тест вычисления общего показателя сложности"""
    analyzer.analyze_image(sample_image)
    score = analyzer.get_image_complexity_score()
    assert 0 <= score <= 100  # Проверяем нормализацию
    assert score > 0  # Должна быть некоторая сложность 