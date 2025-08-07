#!/usr/bin/env python3
"""
Minimal test for API components
"""

def test_basic_imports():
    print("Testing basic imports...")
    
    try:
        import os
        import sys
        import json
        print("✅ Basic Python modules imported")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        print("✅ FastAPI and Pydantic imported")
    except Exception as e:
        print(f"❌ FastAPI/Pydantic import failed: {e}")
        return False
    
    try:
        from config import settings
        print("✅ Config imported")
        print(f"   Gemini API Key configured: {'Yes' if settings.gemini_api_key else 'No'}")
        print(f"   Pinecone API Key configured: {'Yes' if settings.pinecone_api_key else 'No'}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from general_document_processor import GeneralDocumentProcessor
        print("✅ GeneralDocumentProcessor imported")
    except Exception as e:
        print(f"❌ GeneralDocumentProcessor import failed: {e}")
        print(f"   Error details: {str(e)}")
        return False
    
    return True

def test_processor_initialization():
    print("\nTesting processor initialization...")
    
    try:
        from general_document_processor import GeneralDocumentProcessor
        processor = GeneralDocumentProcessor()
        print("✅ GeneralDocumentProcessor initialized")
        return True
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        print(f"   Error details: {str(e)}")
        return False

def test_fastapi_app():
    print("\nTesting FastAPI app...")
    
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        print(f"   Error details: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 API Component Testing")
    print("=" * 40)
    
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Check dependencies.")
        exit(1)
    
    if not test_processor_initialization():
        print("\n❌ Processor initialization failed. Check API keys and dependencies.")
        exit(1)
    
    if not test_fastapi_app():
        print("\n❌ FastAPI app failed. Check main.py imports.")
        exit(1)
    
    print("\n🎉 All tests passed! API should work.")
    print("\n💡 To start the server:")
    print("   python main.py")
