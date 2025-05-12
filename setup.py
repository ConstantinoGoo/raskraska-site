from setuptools import setup, find_packages

setup(
    name="raskraska-site",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",  # для улучшенного анализа изображений
        "scikit-learn>=1.3.0",   # для кластеризации и машинного обучения
        "pytest>=7.4.0",         # для тестирования
        "pytest-cov>=4.1.0",     # для оценки покрытия тестами
    ],
    python_requires=">=3.8",
) 