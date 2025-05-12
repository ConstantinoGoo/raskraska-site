import os
import requests
import logging
from PIL import Image
import numpy as np
import cv2
from image_processor import ImageType, process_image_locally, classify_image
from concurrent.futures import ThreadPoolExecutor
import json
import time
from datetime import datetime
from googlesearch import search
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import sys
import random

# Настройка логирования
log_file = 'auto_tester.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutoTester:
    def __init__(self):
        self.test_dir = "test_images"
        self.results_dir = "test_results"
        self.samples_dir = "test_samples"  # Директория с исходными тестовыми изображениями
        self.categories = {
            ImageType.CARTOON: "cartoon",
            ImageType.PHOTO: "photo",
            ImageType.SKETCH: "sketch",
            ImageType.LOGO: "logo"
        }
        self.setup_directories()
        logger.info("AutoTester инициализирован")

    def setup_directories(self):
        """Создает необходимые директории для тестирования"""
        try:
            # Создаем основные директории
            os.makedirs(self.test_dir, exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.samples_dir, exist_ok=True)
            
            # Создаем поддиректории для каждой категории
            for image_type in ImageType:
                os.makedirs(os.path.join(self.test_dir, image_type.value), exist_ok=True)
                os.makedirs(os.path.join(self.results_dir, image_type.value), exist_ok=True)
                os.makedirs(os.path.join(self.samples_dir, image_type.value), exist_ok=True)
            
            logger.info("Директории успешно созданы")
        except Exception as e:
            logger.error(f"Ошибка при создании директорий: {str(e)}")
            raise

    def prepare_test_images(self):
        """
        Подготавливает тестовые изображения из папки samples
        Копирует их в тестовую директорию для обработки
        """
        test_files = {}
        
        try:
            for image_type in ImageType:
                category_dir = os.path.join(self.samples_dir, image_type.value)
                if not os.path.exists(category_dir):
                    logger.warning(f"Директория {category_dir} не существует")
                    continue
                
                # Получаем список файлов изображений
                image_files = [f for f in os.listdir(category_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                
                if not image_files:
                    logger.warning(f"Нет изображений в директории {category_dir}")
                    continue
                
                test_files[image_type] = []
                
                # Копируем каждое изображение в тестовую директорию
                for img_file in image_files:
                    src_path = os.path.join(category_dir, img_file)
                    dst_path = os.path.join(self.test_dir, image_type.value, img_file)
                    
                    try:
                        # Проверяем, что это действительно изображение
                        with Image.open(src_path) as img:
                            # Копируем файл
                            import shutil
                            shutil.copy2(src_path, dst_path)
                            test_files[image_type].append(dst_path)
                            logger.debug(f"Скопировано изображение: {dst_path}")
                    except Exception as e:
                        logger.warning(f"Ошибка при копировании {src_path}: {str(e)}")
                        continue
                
                logger.info(f"Подготовлено {len(test_files[image_type])} изображений для категории {image_type.value}")
            
            return test_files
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке тестовых изображений: {str(e)}")
            return {}

    def run_tests(self):
        """
        Запускает полный цикл тестирования на локальных изображениях
        """
        all_results = []
        
        try:
            # Подготавливаем тестовые изображения
            test_files = self.prepare_test_images()
            if not test_files:
                logger.error("Нет тестовых изображений для обработки")
                return all_results
            
            # Фильтруем категории, в которых есть изображения
            active_categories = {k: v for k, v in test_files.items() if v}
            if not active_categories:
                logger.error("Нет активных категорий для тестирования")
                return all_results
            
            total_images = sum(len(files) for files in active_categories.values())
            logger.info(f"Начало тестирования. Всего изображений: {total_images}")
            
            # Создаем основной прогресс-бар для категорий
            with tqdm(total=len(active_categories), 
                     desc="Общий прогресс", 
                     position=0) as pbar_categories:
                
                # Для каждой категории
                for image_type, image_paths in active_categories.items():
                    pbar_categories.set_description(f"Категория: {image_type.value:<10}")
                    logger.info(f"Обработка категории: {image_type.value}")
                    
                    # Прогресс-бар для обработки изображений текущей категории
                    with tqdm(total=len(image_paths), 
                            desc=f"Обработка {image_type.value}", 
                            position=1, 
                            leave=False) as pbar_images:
                        
                        success_count = 0
                        total_count = 0
                        
                        for image_path in image_paths:
                            try:
                                # Обрабатываем изображение
                                output_filename = process_image_locally(
                                    image_path,
                                    os.path.join(self.results_dir, image_type.value)
                                )
                                
                                if output_filename:
                                    # Анализируем результат
                                    result = self.analyze_result(
                                        image_path,
                                        os.path.join(self.results_dir, image_type.value, output_filename),
                                        image_type
                                    )
                                    all_results.append(result)
                                    
                                    if result["success"]:
                                        success_count += 1
                                    
                                    logger.info(f"Успешно обработано: {os.path.basename(image_path)}")
                                else:
                                    logger.warning(f"Не удалось обработать: {os.path.basename(image_path)}")
                                
                                total_count += 1
                                
                                # Обновляем прогресс-бар
                                pbar_images.set_description(f"Обработка {image_type.value} [Успешно: {success_count}/{total_count}]")
                                pbar_images.update(1)
                                
                            except Exception as e:
                                logger.error(f"Ошибка при обработке {image_path}: {str(e)}")
                                total_count += 1
                                pbar_images.set_description(f"Обработка {image_type.value} [Успешно: {success_count}/{total_count}]")
                                pbar_images.update(1)
                                continue
                    
                    # Обновляем общий прогресс с информацией о категории
                    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
                    pbar_categories.set_description(f"Категория: {image_type.value} [Успешность: {success_rate:.1f}%]")
                    pbar_categories.update(1)
            
            # Генерируем отчет
            logger.info("Генерация отчета...")
            self.generate_report(all_results, "test_report.html")
            
            # Выводим итоговую статистику
            print("\nИтоговая статистика по категориям:")
            for image_type, image_paths in active_categories.items():
                results = [r for r in all_results if r["expected_type"] == image_type.value]
                success_count = len([r for r in results if r["success"]])
                success_rate = (success_count / len(image_paths) * 100) if image_paths else 0
                print(f"{image_type.value:<10}: {success_count}/{len(image_paths)} (успешность: {success_rate:.1f}%)")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Критическая ошибка при тестировании: {str(e)}")
            return all_results

    def analyze_result(self, input_path, output_path, expected_type):
        """
        Анализирует результат обработки изображения
        """
        try:
            # Проверяем существование файлов
            if not os.path.exists(input_path) or not os.path.exists(output_path):
                logger.error(f"Файл не найден: {input_path if not os.path.exists(input_path) else output_path}")
                return {
                    "input_file": os.path.basename(input_path),
                    "output_file": os.path.basename(output_path),
                    "expected_type": expected_type.value,
                    "success": False,
                    "error": "Файл не найден"
                }
            
            # Загружаем изображения
            input_img = cv2.imread(input_path)
            output_img = cv2.imread(output_path)
            
            if input_img is None or output_img is None:
                logger.error(f"Не удалось загрузить изображение: {input_path if input_img is None else output_path}")
                return {
                    "input_file": os.path.basename(input_path),
                    "output_file": os.path.basename(output_path),
                    "expected_type": expected_type.value,
                    "success": False,
                    "error": "Ошибка загрузки изображения"
                }
            
            # Проверяем размеры
            if input_img.shape[:2] != output_img.shape[:2]:
                logger.error(f"Несоответствие размеров: {input_path} vs {output_path}")
                return {
                    "input_file": os.path.basename(input_path),
                    "output_file": os.path.basename(output_path),
                    "expected_type": expected_type.value,
                    "success": False,
                    "error": "Несоответствие размеров"
                }
            
            # Анализируем качество раскраски
            quality_metrics = {
                'edge_preservation': 0.0,
                'line_clarity': 0.0,
                'fill_uniformity': 0.0,
                'artifacts': 0.0
            }
            
            # Анализ сохранения контуров
            input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            
            edge_scores = []
            for threshold in [100, 127, 150]:
                _, input_edges = cv2.threshold(input_gray, threshold, 255, cv2.THRESH_BINARY)
                _, output_edges = cv2.threshold(output_gray, threshold, 255, cv2.THRESH_BINARY)
                contours_input, _ = cv2.findContours(input_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_output, _ = cv2.findContours(output_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_input) > 0 and len(contours_output) > 0:
                    similarity = cv2.matchShapes(contours_input[0], contours_output[0], cv2.CONTOURS_MATCH_I1, 0.0)
                    edge_scores.append(max(0, min(100, 100 * (1 - similarity))))
            quality_metrics['edge_preservation'] = np.mean(edge_scores) if edge_scores else 0
            
            # Анализ четкости линий
            def analyze_line_clarity(edges):
                sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                return np.mean(gradient_magnitude)
            
            input_clarity = analyze_line_clarity(input_edges)
            output_clarity = analyze_line_clarity(output_edges)
            quality_metrics['line_clarity'] = max(0, min(100, 100 * (output_clarity / input_clarity))) if input_clarity > 0 else 0
            
            # Анализ равномерности заливки
            def analyze_fill_uniformity(img):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                regions = []
                for i in range(0, img.shape[0], 50):
                    for j in range(0, img.shape[1], 50):
                        region = hsv[i:min(i+50, img.shape[0]), j:min(j+50, img.shape[1])]
                        if region.size > 0:
                            std_dev = np.std(region)
                            regions.append(std_dev)
                return np.mean(regions) if regions else 0
            
            input_uniformity = analyze_fill_uniformity(input_img)
            output_uniformity = analyze_fill_uniformity(output_img)
            quality_metrics['fill_uniformity'] = max(0, min(100, 100 * (1 - abs(output_uniformity - input_uniformity) / input_uniformity))) if input_uniformity > 0 else 0
            
            # Анализ артефактов
            def detect_artifacts(img):
                blur = cv2.GaussianBlur(img, (5,5), 0)
                diff = cv2.absdiff(img, blur)
                _, artifacts = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                return np.sum(artifacts) / (img.shape[0] * img.shape[1] * 3)
            
            artifact_level = detect_artifacts(output_img)
            quality_metrics['artifacts'] = max(0, min(100, 100 * (1 - artifact_level)))
            
            # Вычисляем итоговую оценку с весами
            weights = {
                'edge_preservation': 0.4,
                'line_clarity': 0.3,
                'fill_uniformity': 0.2,
                'artifacts': 0.1
            }
            
            quality_score = sum(weights[metric] * quality_metrics[metric] for metric in weights)
            
            # Логируем детальную информацию
            logger.debug(f"Анализ качества раскраски для {os.path.basename(input_path)}:")
            for metric, score in quality_metrics.items():
                logger.debug(f"- {metric}: {score:.1f}%")
            logger.debug(f"Итоговая оценка: {quality_score:.1f}%")
            
            return {
                "input_file": os.path.basename(input_path),
                "output_file": os.path.basename(output_path),
                "expected_type": expected_type.value,
                "success": True,
                "quality_score": quality_score,
                "quality_metrics": quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе результата: {str(e)}")
            return {
                "input_file": os.path.basename(input_path),
                "output_file": os.path.basename(output_path),
                "expected_type": expected_type.value,
                "success": False,
                "error": str(e)
            }

    def analyze_coloring_quality(self, input_img, output_img):
        """
        Анализирует качество раскраски по нескольким метрикам:
        1. Сохранение контуров (edge preservation)
        2. Четкость линий (line clarity)
        3. Равномерность заливки (fill uniformity)
        4. Отсутствие артефактов (artifact detection)
        """
        try:
            # 1. Анализ сохранения контуров
            input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
            
            # Находим контуры с разными порогами для лучшего определения
            edge_scores = []
            for threshold in [100, 127, 150]:
                _, input_edges = cv2.threshold(input_gray, threshold, 255, cv2.THRESH_BINARY)
                _, output_edges = cv2.threshold(output_gray, threshold, 255, cv2.THRESH_BINARY)
                
                # Используем более точное сравнение контуров
                contours_input, _ = cv2.findContours(input_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_output, _ = cv2.findContours(output_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Сравниваем количество и форму контуров
                if len(contours_input) > 0 and len(contours_output) > 0:
                    similarity = cv2.matchShapes(contours_input[0], contours_output[0], cv2.CONTOURS_MATCH_I1, 0.0)
                    edge_scores.append(max(0, min(100, 100 * (1 - similarity))))
            
            edge_preservation_score = np.mean(edge_scores) if edge_scores else 0
            
            # 2. Анализ четкости линий
            def analyze_line_clarity(edges):
                # Используем оператор Собеля для определения градиентов
                sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                return np.mean(gradient_magnitude)
            
            input_clarity = analyze_line_clarity(input_edges)
            output_clarity = analyze_line_clarity(output_edges)
            line_clarity_score = max(0, min(100, 100 * (output_clarity / input_clarity))) if input_clarity > 0 else 0
            
            # 3. Анализ равномерности заливки
            def analyze_fill_uniformity(img):
                # Конвертируем в HSV для лучшего анализа цветов
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # Анализируем стандартное отклонение в каждой области
                regions = []
                for i in range(0, img.shape[0], 50):
                    for j in range(0, img.shape[1], 50):
                        region = hsv[i:min(i+50, img.shape[0]), j:min(j+50, img.shape[1])]
                        if region.size > 0:
                            std_dev = np.std(region)
                            regions.append(std_dev)
                return np.mean(regions) if regions else 0
            
            input_uniformity = analyze_fill_uniformity(input_img)
            output_uniformity = analyze_fill_uniformity(output_img)
            fill_uniformity_score = max(0, min(100, 100 * (1 - abs(output_uniformity - input_uniformity) / input_uniformity))) if input_uniformity > 0 else 0
            
            # 4. Анализ артефактов
            def detect_artifacts(img):
                # Ищем резкие изменения цвета и шум
                blur = cv2.GaussianBlur(img, (5,5), 0)
                diff = cv2.absdiff(img, blur)
                _, artifacts = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                return np.sum(artifacts) / (img.shape[0] * img.shape[1] * 3)
            
            artifact_level = detect_artifacts(output_img)
            artifact_score = max(0, min(100, 100 * (1 - artifact_level)))
            
            # Вычисляем итоговую оценку с весами
            weights = {
                'edge_preservation': 0.4,
                'line_clarity': 0.3,
                'fill_uniformity': 0.2,
                'artifacts': 0.1
            }
            
            quality_score = (
                weights['edge_preservation'] * edge_preservation_score +
                weights['line_clarity'] * line_clarity_score +
                weights['fill_uniformity'] * fill_uniformity_score +
                weights['artifacts'] * artifact_score
            )
            
            # Логируем детальную информацию о качестве
            logger.debug(f"Анализ качества раскраски:")
            logger.debug(f"- Сохранение контуров: {edge_preservation_score:.1f}%")
            logger.debug(f"- Четкость линий: {line_clarity_score:.1f}%")
            logger.debug(f"- Равномерность заливки: {fill_uniformity_score:.1f}%")
            logger.debug(f"- Отсутствие артефактов: {artifact_score:.1f}%")
            logger.debug(f"Итоговая оценка: {quality_score:.1f}%")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Ошибка при анализе качества: {str(e)}")
            return 0

    def generate_report(self, results, output_file):
        """
        Генерирует HTML отчет по результатам тестирования
        """
        try:
            # Создаем базовую структуру отчета
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Отчет по тестированию раскрасок</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .success { color: green; }
                    .error { color: red; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .quality-good { background-color: #dff0d8; }
                    .quality-medium { background-color: #fcf8e3; }
                    .quality-poor { background-color: #f2dede; }
                    .metric-bar {
                        width: 100px;
                        height: 10px;
                        background-color: #f0f0f0;
                        display: inline-block;
                        margin-right: 10px;
                    }
                    .metric-fill {
                        height: 100%;
                        background-color: #4CAF50;
                    }
                    .metric-value {
                        display: inline-block;
                        width: 50px;
                    }
                </style>
            </head>
            <body>
                <h1>Отчет по тестированию раскрасок</h1>
                <p>Дата: {}</p>
                <h2>Общая статистика</h2>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Добавляем общую статистику
            total_tests = len(results)
            successful_tests = len([r for r in results if r["success"]])
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            html_content += f"""
                <p>Всего тестов: {total_tests}</p>
                <p>Успешных тестов: {successful_tests}</p>
                <p>Процент успеха: {success_rate:.1f}%</p>
                <h2>Детальные результаты</h2>
            """
            
            # Добавляем таблицу с результатами
            html_content += """
                <table>
                    <tr>
                        <th>Входной файл</th>
                        <th>Выходной файл</th>
                        <th>Ожидаемый тип</th>
                        <th>Статус</th>
                        <th>Общая оценка</th>
                    </tr>
            """
            
            # Добавляем результаты по каждому тесту
            for result in results:
                status_class = "success" if result["success"] else "error"
                quality_class = ""
                if result["success"]:
                    quality_score = result["quality_score"]
                    if quality_score >= 80:
                        quality_class = "quality-good"
                    elif quality_score >= 60:
                        quality_class = "quality-medium"
                    else:
                        quality_class = "quality-poor"
                
                html_content += f"""
                    <tr class="{quality_class}">
                        <td>{result["input_file"]}</td>
                        <td>{result["output_file"]}</td>
                        <td>{result["expected_type"]}</td>
                        <td class="{status_class}">{"Успешно" if result["success"] else "Ошибка"}</td>
                        <td>{f"{result['quality_score']:.1f}%" if result["success"] else result["error"]}</td>
                    </tr>
                """
                
                # Если тест успешный, добавляем детальные метрики
                if result["success"] and "quality_metrics" in result:
                    html_content += f"""
                        <tr>
                            <td colspan="5">
                                <div style="padding: 10px;">
                                    <h4>Детальные метрики качества:</h4>
                                    <table style="width: 50%;">
                    """
                    
                    # Добавляем каждую метрику с визуальным представлением
                    metrics_names = {
                        'edge_preservation': 'Сохранение контуров',
                        'line_clarity': 'Четкость линий',
                        'fill_uniformity': 'Равномерность заливки',
                        'artifacts': 'Отсутствие артефактов'
                    }
                    
                    for metric, value in result["quality_metrics"].items():
                        bar_color = "#4CAF50" if value >= 80 else "#FFA500" if value >= 60 else "#FF0000"
                        html_content += f"""
                            <tr>
                                <td style="width: 200px;">{metrics_names[metric]}:</td>
                                <td>
                                    <div class="metric-bar">
                                        <div class="metric-fill" style="width: {value}%; background-color: {bar_color};"></div>
                                    </div>
                                    <span class="metric-value">{value:.1f}%</span>
                                </td>
                            </tr>
                        """
                    
                    html_content += """
                                    </table>
                                </div>
                            </td>
                        </tr>
                    """
            
            html_content += """
                </table>
            </body>
            </html>
            """
            
            # Сохраняем отчет
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Отчет сохранен в файл: {output_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {str(e)}")

def main():
    """
    Основная функция для запуска тестирования
    """
    try:
        print(f"\nЗапуск автоматического тестирования раскрасок")
        print(f"Лог файл: {log_file}\n")
        
        tester = AutoTester()
        results = tester.run_tests()
        
        # Выводим краткую статистику
        success_count = len([r for r in results if r["success"]])
        total_count = len(results)
        
        print("\nРезультаты тестирования:")
        print(f"Всего обработано: {total_count}")
        print(f"Успешно: {success_count}")
        if total_count > 0:
            print(f"Процент успеха: {(success_count/total_count*100):.1f}%")
        print(f"\nПодробный отчет сохранен в: test_report.html")
        print(f"Полный лог доступен в: {log_file}")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        print(f"\nПроизошла критическая ошибка. Проверьте лог файл: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 