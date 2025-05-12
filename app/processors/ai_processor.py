import os
from openai import OpenAI, OpenAIError
import base64
from PIL import Image
import io
import logging
import requests
from typing import Optional

# Configure logging
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
    """
    A class that handles image processing using OpenAI's DALL-E API.
    Converts regular images into coloring pages using AI.
    """
    
    def __init__(self):
        """Initialize the processor with OpenAI API credentials"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            if not api_key.startswith('sk-'):
                raise ValueError("Invalid OpenAI API key format")
            self.client = OpenAI(api_key=api_key)
            logger.debug("OpenAI client successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

    def validate_image(self, image_path: str) -> bool:
        """
        Validates if the image meets API requirements.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid
            
        Raises:
            ValueError: If image doesn't meet requirements
        """
        try:
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ['PNG', 'JPEG']:
                    raise ValueError(f"Unsupported image format: {img.format}")
                
                # Check file size (4MB limit)
                file_size = os.path.getsize(image_path)
                if file_size > 4 * 1024 * 1024:
                    raise ValueError("File size exceeds 4MB")
                
                # Check dimensions (2048x2048 limit)
                width, height = img.size
                if width > 2048 or height > 2048:
                    raise ValueError("Image dimensions exceed 2048x2048")
                
                return True
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            raise

    def _prepare_image(self, image_path: str) -> io.BytesIO:
        """
        Prepares image for API submission by converting to RGB and proper format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BytesIO: Prepared image buffer
        """
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            return img_buffer

    def create_coloring_page(self, image_path: str, output_path: str) -> bool:
        """
        Creates a coloring page from an image using OpenAI DALL-E.
        
        Args:
            image_path: Path to source image
            output_path: Path to save the coloring page
        
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            logger.debug(f"Starting image processing: {image_path}")
            
            # Check file existence
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"File not found: {image_path}")
            
            # Validate and prepare image
            self.validate_image(image_path)
            img_buffer = self._prepare_image(image_path)
            
            # Make DALL-E API request
            logger.debug("Sending request to DALL-E API")
            try:
                response = self.client.images.create_variation(
                    image=img_buffer,
                    n=1,
                    size="1024x1024",
                    response_format="url"
                )
                
                logger.debug("Received response from DALL-E API")
                
                # Get and download the generated image
                image_url = response.data[0].url
                logger.debug(f"Got result URL: {image_url}")
                
                response = requests.get(image_url)
                response.raise_for_status()
                
                # Save the result
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.debug(f"Result saved to: {output_path}")
                
                return True
                
            except OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing image through AI: {str(e)}", exc_info=True)
            return False

def process_image(input_path: str, output_dir: str) -> Optional[str]:
    """
    Process an image and create a coloring page.
    
    Args:
        input_path: Path to source image
        output_dir: Directory to save the result
    
    Returns:
        Optional[str]: Filename of created coloring page or None if failed
    """
    try:
        logger.debug(f"Starting image processing: {input_path}")
        
        # Create output filename
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_coloring{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        logger.debug(f"Output path: {output_path}")
        
        # Create coloring page
        processor = AIImageProcessor()
        if processor.create_coloring_page(input_path, output_path):
            logger.debug("Processing completed successfully")
            return output_filename
            
        logger.error("Failed to create coloring page")
        return None
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None 