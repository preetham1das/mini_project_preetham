import urllib.request
import urllib.parse
import json

url = "http://127.0.0.1:8000/predict-temporal"

for t in [0.3, 0.5, 0.7, 0.9]:
    data = urllib.parse.urlencode({
        "sample_path_a": "d:/projects/Deforestation Detection from Satellite Images/data/Forest-Change/extracted/Forest-Change-dataset/images/test/A/test_000001.png",
        "sample_path_b": "d:/projects/Deforestation Detection from Satellite Images/data/Forest-Change/extracted/Forest-Change-dataset/images/test/B/test_000001.png",
        "threshold": str(t)
    }).encode('utf-8')

    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print(f"Threshold {t}: {res.get('deforestation_percentage')}% ({res.get('changed_pixels')} pixels)")
