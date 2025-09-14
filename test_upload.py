"""
Test script to demonstrate MRsim QC web application functionality.
"""

import requests
import os
from pathlib import Path

def test_mrsim_qc():
    """Test the MRsim QC web application."""
    
    base_url = "http://localhost:5001"
    
    # Test 1: Check if application is running
    print("🚀 Testing MRsim QC Web Application")
    print("=" * 50)
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Application is running successfully!")
            print(f"   URL: {base_url}")
        else:
            print(f"❌ Application returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error connecting to application: {e}")
        return
    
    # Test 2: Check if we can access the upload page
    print("\n📋 Testing Upload Page")
    print("-" * 30)
    
    if "MRsim QC" in response.text and "Upload DICOM Files" in response.text:
        print("✅ Upload page is accessible")
        print("✅ HTML template is working correctly")
    else:
        print("❌ Upload page content not found")
    
    # Test 3: Create a demo session (since we're using demo data)
    print("\n🧪 Testing Demo Analysis")
    print("-" * 30)
    
    # For demo purposes, we can test with any files since the app uses demo data
    demo_files = {"files": ("demo.dcm", b"demo_dicom_content", "application/octet-stream")}
    
    try:
        upload_response = requests.post(f"{base_url}/upload", files=demo_files)
        
        if upload_response.status_code == 302:  # Redirect to results
            print("✅ Upload processing successful")
            
            # Extract session ID from redirect
            redirect_url = upload_response.headers.get('Location')
            if redirect_url and '/results/' in redirect_url:
                session_id = redirect_url.split('/results/')[-1]
                print(f"✅ Session created: {session_id}")
                
                # Test results page
                results_url = f"{base_url}/results/{session_id}"
                results_response = requests.get(results_url)
                
                if results_response.status_code == 200:
                    print("✅ Results page accessible")
                    
                    # Test API endpoint
                    api_url = f"{base_url}/api/stats/{session_id}"
                    api_response = requests.get(api_url)
                    
                    if api_response.status_code == 200:
                        print("✅ API endpoint working")
                        stats = api_response.json()
                        print(f"   Max distortion: {stats.get('distortion_max', 'N/A'):.2f} mm")
                        print(f"   Mean distortion: {stats.get('distortion_mean', 'N/A'):.2f} mm")
                    else:
                        print("❌ API endpoint failed")
                        
                    print(f"\n🔗 Access your results at: {results_url}")
                else:
                    print("❌ Results page not accessible")
            else:
                print("❌ No session ID found in redirect")
        else:
            print(f"❌ Upload failed with status: {upload_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing upload: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 MRsim QC Testing Complete!")
    print("=" * 50)
    print(f"🌐 Web Application URL: {base_url}")
    print("📝 Features Available:")
    print("   • Upload DICOM files (magnitude + phase difference)")
    print("   • Automatic B0 field map estimation")
    print("   • Geometrical distortion analysis")
    print("   • Comprehensive visualizations")
    print("   • Download results (ZIP, field maps, distortion maps)")
    print("   • Clinical assessment and recommendations")
    
    print("\n📋 To use with real data:")
    print("   1. Upload magnitude images at different TEs")
    print("   2. Upload phase difference images")
    print("   3. Get comprehensive distortion analysis")
    print("   4. Download results for further processing")

if __name__ == "__main__":
    test_mrsim_qc()
