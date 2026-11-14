import urllib.request
import urllib.parse
import json

url = "http://127.0.0.1:8000/predict-temporal"
data = urllib.parse.urlencode({
    "sample_path_a": "d:/projects/Deforestation Detection from Satellite Images/data/Forest-Change/extracted/Forest-Change-dataset/images/test/A/test_000003.png",
    "sample_path_b": "d:/projects/Deforestation Detection from Satellite Images/data/Forest-Change/extracted/Forest-Change-dataset/images/test/B/test_000003.png",
    "threshold": "0.5"
}).encode('utf-8')

req = urllib.request.Request(url, data=data)
with urllib.request.urlopen(req) as response:
    res = json.loads(response.read().decode('utf-8'))
    print(f"Percentage: {res.get('deforestation_percentage')}")
    print(f"Changed pixels: {res.get('changed_pixels')}")
