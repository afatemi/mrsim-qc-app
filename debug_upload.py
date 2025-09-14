#!/usr/bin/env python3
"""
Simple debug script to test file upload functionality
"""
import requests
import os
from pathlib import Path

def test_upload():
    """Test the upload endpoint with a simple file"""
    
    # Check if Flask app is running
    try:
        response = requests.get('http://localhost:5001', timeout=5)
        print(f"✅ Flask app is running (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ Flask app is not running: {e}")
        return
    
    # Create a test file
    test_file_path = Path("test_file.txt")
    test_file_path.write_text("This is a test file for upload debugging")
    
    try:
        # Test upload
        with open(test_file_path, 'rb') as f:
            files = {'files': ('test_file.txt', f, 'text/plain')}
            response = requests.post('http://localhost:5001/upload', files=files, timeout=10)
        
        print(f"📤 Upload response status: {response.status_code}")
        print(f"📤 Upload response URL: {response.url}")
        print(f"📤 Upload response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Upload successful!")
        elif response.status_code == 302:
            print("✅ Upload successful (redirected)!")
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()

if __name__ == "__main__":
    test_upload()
