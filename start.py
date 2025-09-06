#!/usr/bin/env python3
"""
Startup script for the Mental Health A2A Agent Ecosystem
"""

import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import transformers
        import torch
        import numpy
        import pandas
        import sqlalchemy
        import redis
        import cryptography
        import librosa
        import cv2
        import pypdf2
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment variables and configuration"""
    print("🔍 Checking environment configuration...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key in the .env file")
    else:
        print("✅ OpenAI API key found")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found, using default configuration")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    
    directories = [
        "logs",
        "data",
        "temp",
        "models",
        "frontend/static",
        "frontend/templates"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"   Created: {directory}")

def main():
    """Main startup function"""
    print("🚀 Starting Mental Health A2A Agent Ecosystem")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\n🌟 System ready! Starting server...")
    print("📱 Web interface: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔧 Admin interface: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
