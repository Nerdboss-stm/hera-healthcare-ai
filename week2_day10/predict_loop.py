import requests
import time

for i in range(50):
    response = requests.get("http://localhost:5000/predict")
    if response.ok:
        print(f"✅ Patient {i+1}:", response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
    time.sleep(2)
