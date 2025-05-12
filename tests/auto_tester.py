import os
import sys
import cv2
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from app.processors.image_processor import ImageType, process_image_locally, classify_image
from app.processors.ai_processor import process_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/auto_tester.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def display_images(original_path, result_path=None, title="Test Image"):
    """Display original image and its processing result side by side"""
    plt.figure(figsize=(15, 5))
    
    # Read and display original image
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(original)
    plt.title(f"Original: {os.path.basename(original_path)}")
    plt.axis('off')
    
    # If result exists, display it
    if result_path and os.path.exists(result_path):
        result = cv2.imread(result_path)
        plt.subplot(122)
        plt.imshow(result, cmap='gray')
        plt.title(f"Result: {os.path.basename(result_path)}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def check_test_data():
    """Check if all required test data is present"""
    required_files = [
        'tests/data/test_samples/cartoon1.jpg',
        'tests/data/test_samples/photo1.jpg',
        'tests/data/test_samples/sketch1.jpg',
        'tests/data/test_samples/logo1.png',
        'tests/data/test_images/jessica.jpeg',
        'tests/data/test_images/spider-man.jpeg'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing test files:")
        for file_path in missing_files:
            logger.error(f"- {file_path}")
        logger.info("Please run tests/generate_test_data.py to generate test images")
        return False
    
    return True

def test_image_classification():
    """Test image type classification accuracy"""
    test_cases = {
        'tests/data/test_samples/cartoon1.jpg': ImageType.CARTOON,
        'tests/data/test_samples/photo1.jpg': ImageType.PHOTO,
        'tests/data/test_samples/sketch1.jpg': ImageType.SKETCH,
        'tests/data/test_samples/logo1.png': ImageType.LOGO
    }
    
    success = 0
    total = len(test_cases)
    
    for image_path, expected_type in test_cases.items():
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                continue
            
            # Display test image
            display_images(image_path, title=f"Test Classification: {expected_type.value}")
            
            detected_type = classify_image(image)
            if detected_type == expected_type:
                success += 1
                logger.info(f"✓ Correctly classified {image_path} as {detected_type.value}")
            else:
                logger.error(f"✗ Misclassified {image_path} as {detected_type.value}, expected {expected_type.value}")
            
        except Exception as e:
            logger.error(f"Error testing {image_path}: {str(e)}")
    
    accuracy = (success / total) * 100
    logger.info(f"Classification accuracy: {accuracy:.1f}%")
    return accuracy

def test_image_processing():
    """Test image processing quality"""
    test_images = [
        'tests/data/test_images/jessica.jpeg',
        'tests/data/test_images/spider-man.jpeg'
    ]
    
    results = {}
    for image_path in test_images:
        try:
            # Process image
            output_dir = 'tests/data/test_results'
            os.makedirs(output_dir, exist_ok=True)
            
            result_filename = process_image_locally(image_path, output_dir)
            if not result_filename:
                logger.error(f"Failed to process {image_path}")
                continue
            
            result_path = os.path.join(output_dir, result_filename)
            
            # Display original and result
            display_images(image_path, result_path, title=f"Processing Result: {os.path.basename(image_path)}")
            
            # Evaluate result
            original = cv2.imread(image_path)
            result = cv2.imread(result_path)
            
            if original is None or result is None:
                logger.error(f"Could not read images for {image_path}")
                continue
            
            # Calculate metrics
            edge_score = evaluate_edge_preservation(original, result)
            clarity_score = evaluate_line_clarity(result)
            artifact_score = evaluate_artifacts(result)
            
            # Calculate overall quality score
            quality_score = (edge_score + clarity_score + (100 - artifact_score)) / 3
            results[image_path] = quality_score
            
            logger.info(f"Results for {os.path.basename(image_path)}:")
            logger.info(f"Edge preservation: {edge_score:.1f}%")
            logger.info(f"Line clarity: {clarity_score:.1f}%")
            logger.info(f"Artifact score: {artifact_score:.1f}%")
            logger.info(f"Overall quality: {quality_score:.1f}%")
            
        except Exception as e:
            logger.error(f"Error testing {image_path}: {str(e)}")
    
    return results

def evaluate_edge_preservation(original, result):
    """Evaluate how well edges are preserved"""
    try:
        # Convert to grayscale
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        original_edges = cv2.Canny(original_gray, 100, 200)
        result_edges = cv2.Canny(result_gray, 100, 200)
        
        # Calculate similarity
        intersection = np.logical_and(original_edges, result_edges)
        union = np.logical_or(original_edges, result_edges)
        
        if np.sum(union) == 0:
            return 0
            
        similarity = np.sum(intersection) / np.sum(union) * 100
        return similarity
        
    except Exception as e:
        logger.error(f"Error in edge preservation evaluation: {str(e)}")
        return 0

def evaluate_line_clarity(result):
    """Evaluate the clarity and consistency of lines"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient
        gradient = cv2.normalize(gradient, None, 0, 100, cv2.NORM_MINMAX)
        
        # Calculate average gradient magnitude along edges
        edges = cv2.Canny(gray, 100, 200)
        if np.sum(edges) == 0:
            return 0
            
        clarity = np.mean(gradient[edges > 0])
        return clarity
        
    except Exception as e:
        logger.error(f"Error in line clarity evaluation: {str(e)}")
        return 0

def evaluate_artifacts(result):
    """Evaluate the presence of unwanted artifacts"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to detect artifacts
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Calculate difference
        diff = cv2.absdiff(gray, closed)
        
        # Calculate artifact score
        artifact_ratio = np.sum(diff > 30) / diff.size
        artifact_score = artifact_ratio * 100
        
        return min(artifact_score * 5, 100)  # Scale up but cap at 100
        
    except Exception as e:
        logger.error(f"Error in artifact evaluation: {str(e)}")
        return 100

if __name__ == "__main__":
    logger.info("Starting automated tests...")
    
    # Check test data
    if not check_test_data():
        logger.error("Cannot proceed with tests - missing test data")
        sys.exit(1)
    
    # Test image classification
    logger.info("\nTesting image classification...")
    classification_accuracy = test_image_classification()
    
    # Test image processing
    logger.info("\nTesting image processing...")
    processing_results = test_image_processing()
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info(f"Classification Accuracy: {classification_accuracy:.1f}%")
    logger.info("Processing Quality Scores:")
    for image, score in processing_results.items():
        logger.info(f"{os.path.basename(image)}: {score:.1f}%")
    
    # Determine overall success
    classification_threshold = 80
    processing_threshold = 85
    
    all_processing_scores_good = all(score >= processing_threshold for score in processing_results.values())
    
    if classification_accuracy >= classification_threshold and all_processing_scores_good:
        logger.info("\n✓ All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests did not meet quality thresholds")
        sys.exit(1) 