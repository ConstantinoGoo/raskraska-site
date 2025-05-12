import os
import cv2
import numpy as np
import pytest
from app.processors.image_processor import create_coloring_page, process_image, ImageType

@pytest.fixture
def test_image():
    """Создание тестового изображения"""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Добавляем простые фигуры
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), 2)
    cv2.circle(image, (50, 50), 20, (128, 128, 128), -1)
    cv2.line(image, (10, 90), (90, 10), (200, 200, 200), 2)
    return image

def test_create_coloring_page(test_image):
    """Тест создания раскраски"""
    result = create_coloring_page(test_image)
    assert result is not None
    assert result.shape == test_image.shape
    
    # Проверяем, что результат содержит черные линии на белом фоне
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    assert np.any(gray_result == 0)  # должны быть черные пиксели
    assert np.any(gray_result == 255)  # должны быть белые пиксели

def test_process_image(test_image):
    """Тест обработки изображения"""
    result = process_image(test_image, ImageType.SKETCH)
    assert result is not None
    assert result.shape == test_image.shape
    
    # Проверяем, что результат отличается от оригинала
    assert not np.array_equal(result, test_image)
    
    # Проверяем наличие контуров
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_result, 100, 200)
    assert np.sum(edges > 0) > 0  # должны быть обнаружены края

def test_process_image_with_different_types(test_image):
    """Тест обработки изображения с разными типами"""
    for image_type in ImageType:
        result = process_image(test_image, image_type)
        assert result is not None
        assert result.shape == test_image.shape
        
        # Проверяем базовые характеристики результата
        assert np.any(result < 255)  # должны быть не только белые пиксели
        assert np.any(result == 255)  # должны быть белые пиксели

def test_edge_detection(test_image):
    """Тест определения краев"""
    result = create_coloring_page(test_image)
    assert result is not None
    
    # Проверяем наличие краев
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_result, 100, 200)
    assert np.sum(edges > 0) > 0  # должны быть обнаружены края

def test_image_preprocessing(test_image):
    """Тест предварительной обработки изображения"""
    # Добавляем шум
    noise = np.random.normal(0, 25, test_image.shape).astype(np.uint8)
    noisy_image = cv2.add(test_image, noise)
    
    result = create_coloring_page(noisy_image)
    assert result is not None
    
    # Проверяем, что результат содержит меньше шума
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges_result = cv2.Canny(gray_result, 100, 200)
    edges_original = cv2.Canny(test_image, 100, 200)
    
    # Проверяем, что основные края сохранились
    intersection = np.logical_and(edges_result > 0, edges_original > 0)
    union = np.logical_or(edges_result > 0, edges_original > 0)
    if np.sum(union) > 0:
        similarity = np.sum(intersection) / np.sum(union)
        assert similarity > 0.1  # хотя бы 10% краев должно совпадать 