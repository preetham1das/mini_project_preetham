import requests

url = "http://127.0.0.1:8000/predict-temporal"
files = {
    "before_image": ("t1.png", open("data/Forest-Change/extracted/Forest-Change-dataset/images/test/A/test_000001.png", "rb"), "image/png"),
    "after_image": ("t2.png", open("data/Forest-Change/extracted/Forest-Change-dataset/images/test/B/test_000001.png", "rb"), "image/png")
}
data = {"threshold": 0.5}
res = requests.post(url, files=files, data=data)
try:
    print(res.json())
except:
    print(res.text)
