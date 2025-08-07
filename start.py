#!/usr/bin/env python3
"""
Startup script for the LLM Document Processing System
Handles environment setup and starts the server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_env_variables():
    """Check if required environment variables are set"""
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n💡 Please create a .env file with:")
        for var in missing_vars:
            print(f"   {var}=your_{var.lower()}_here")
        return False
    
    print("✅ Environment variables configured")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_file_structure():
    """Check if all required files exist"""
    required_files = [
        "main.py",
        "general_document_processor.py",
        "config.py",
        "requirements.txt",
        "src/vector_storage.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting LLM Document Processing System...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main startup function"""
    print("🎯 LLM Document Processing System - Startup")
    print("="*50)
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Check file structure
    if not check_file_structure():
        return
    
    # Step 3: Check environment variables
    if not check_env_variables():
        print("\n💡 You can also set environment variables in your shell:")
        print("   export GEMINI_API_KEY=your_api_key_here")
        print("   export PINECONE_API_KEY=your_api_key_here")
        return
    
    # Step 4: Ask about dependency installation
    install_deps = input("\n📦 Install/update dependencies? (y/n) [y]: ").strip().lower()
    if install_deps != 'n':
        if not install_dependencies():
            return
    
    # Step 5: Start the server
    print("\n✅ All checks passed!")
    start_choice = input("🚀 Start the server now? (y/n) [y]: ").strip().lower()
    
    if start_choice != 'n':
        start_server()
    else:
        print("\n💡 To start manually, run: python main.py")
        print("🧪 To test the API, run: python test_api.py")
        print("🎯 For demo, run: python demo_problem_statement.py")

if __name__ == "__main__":
    main()
