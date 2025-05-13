import os
import shutil
import time
import logging
from threading import Lock
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
file_lock = Lock()

class FileManager:
    def __init__(self, upload_dir: str, results_dir: str, max_age_days: int = 7):
        self.upload_dir = Path(upload_dir)
        self.results_dir = Path(results_dir)
        self.max_age_days = max_age_days
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создает необходимые директории, если они не существуют"""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_old_files(self):
        """Удаляет файлы старше max_age_days"""
        current_time = datetime.now()
        
        for directory in [self.upload_dir, self.results_dir]:
            try:
                for file_path in directory.glob('*.*'):
                    if not file_path.is_file():
                        continue
                    
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if current_time - file_age > timedelta(days=self.max_age_days):
                        file_path.unlink()
                        logger.info(f"Удален старый файл: {file_path}")
            except Exception as e:
                logger.error(f"Ошибка при очистке старых файлов в {directory}: {e}")
    
    def save_file(self, file_data, filename: str, directory: str = 'upload') -> Optional[Path]:
        """Сохраняет файл атомарно"""
        target_dir = self.upload_dir if directory == 'upload' else self.results_dir
        final_path = target_dir / filename
        temp_path = target_dir / f"{filename}.tmp"
        
        try:
            with file_lock:
                # Сохраняем во временный файл
                if hasattr(file_data, 'save'):
                    file_data.save(str(temp_path))
                else:
                    with open(temp_path, 'wb') as f:
                        f.write(file_data)
                
                # Атомарно перемещаем в целевое расположение
                temp_path.rename(final_path)
                logger.debug(f"Файл успешно сохранен: {final_path}")
                return final_path
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении файла {filename}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None
    
    def get_file_path(self, filename: str, directory: str = 'results') -> Optional[Path]:
        """Получает путь к файлу, проверяя его существование"""
        target_dir = self.upload_dir if directory == 'upload' else self.results_dir
        file_path = target_dir / filename
        return file_path if file_path.exists() else None
    
    def cleanup_temp_files(self):
        """Очищает временные файлы"""
        try:
            for directory in [self.upload_dir, self.results_dir]:
                for temp_file in directory.glob('*.tmp'):
                    if temp_file.is_file():
                        temp_file.unlink()
                        logger.debug(f"Удален временный файл: {temp_file}")
        except Exception as e:
            logger.error(f"Ошибка при очистке временных файлов: {e}") 