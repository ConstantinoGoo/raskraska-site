#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['uploads', 'results', 'test_samples', 'test_results']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def setup_environment():
    """Setup virtual environment and install dependencies"""
    if not Path('venv').exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    # Determine the correct pip and python commands
    if sys.platform == 'win32':
        pip_cmd = r'venv\Scripts\pip'
        python_cmd = r'venv\Scripts\python'
    else:
        pip_cmd = 'venv/bin/pip'
        python_cmd = 'venv/bin/python'
    
    print("Installing dependencies...")
    subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'])
    print("✓ Dependencies installed")

def setup_env_file():
    """Setup .env file if it doesn't exist"""
    if not Path('.env').exists():
        if Path('.env.example').exists():
            Path('.env.example').copy('.env')
            print("✓ Created .env file from .env.example")
            print("⚠️  Don't forget to update your .env file with your actual API keys and settings!")
        else:
            print("⚠️  Warning: .env.example not found!")

def check_installation():
    """Run basic checks to verify installation"""
    try:
        import flask
        import openai
        import opencv_python
        import numpy
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"⚠️  Warning: Some packages are missing: {e}")

def main():
    """Main setup function"""
    print("Starting project setup...")
    
    # Create necessary directories
    create_directories()
    
    # Setup virtual environment and install dependencies
    setup_environment()
    
    # Setup environment file
    setup_env_file()
    
    # Check installation
    check_installation()
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Update the .env file with your API keys and settings")
    print("2. Activate the virtual environment:")
    if sys.platform == 'win32':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the application: flask run")

if __name__ == '__main__':
    main() 