"""
Quick diagnostic script to test if the AI components are working
Run this to check what's wrong: python test_ai.py
"""

import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 50)
print("CropHealth AI - Diagnostic Test")
print("=" * 50)

# Check 1: Environment variables
print("\n1. Checking environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("   ✅ python-dotenv is installed")
except ImportError:
    print("   ⚠️  python-dotenv not installed (optional)")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    print(f"   ✅ GEMINI_API_KEY found (length: {len(GEMINI_API_KEY)})")
else:
    print("   ❌ GEMINI_API_KEY not found!")
    print("   → Create a .env file with: GEMINI_API_KEY=your_key_here")

# Check 2: Required packages
print("\n2. Checking Python packages...")
packages = {
    'flask': 'Flask',
    'requests': 'requests',
    'PIL': 'Pillow',
    'numpy': 'numpy',
    'tensorflow': 'TensorFlow',
    'dotenv': 'python-dotenv'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f"   ✅ {name} installed")
    except ImportError:
        print(f"   ❌ {name} NOT installed")
        print(f"      → Run: pip install {name.lower()}")

# Check 3: Model file
print("\n3. Checking model file...")
if os.path.exists('plant_disease_model.h5'):
    size_mb = os.path.getsize('plant_disease_model.h5') / (1024 * 1024)
    print(f"   ✅ plant_disease_model.h5 found ({size_mb:.1f} MB)")
else:
    print("   ❌ plant_disease_model.h5 NOT found!")
    print("   → You need to train the model or download it")

# Check 4: Test Gemini API (if key exists)
if GEMINI_API_KEY:
    print("\n4. Testing Gemini API connection...")
    try:
        import requests
        import json
        
        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [{"text": "Say 'Hello' if you can read this."}]
            }]
        }
        
        response = requests.post(GEMINI_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=10)
        
        if response.status_code == 200:
            result = response.json()['candidates'][0]['content']['parts'][0]['text']
            print(f"   ✅ Gemini API working! Response: {result[:50]}...")
        else:
            print(f"   ❌ Gemini API error: {response.status_code}")
            print(f"      Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Gemini API test failed: {e}")
else:
    print("\n4. Skipping Gemini API test (no API key)")

print("\n" + "=" * 50)
print("Diagnostic complete!")
print("=" * 50)

