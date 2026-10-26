from app import app
client = app.test_client()
resp = client.get('/')
print('Status:', resp.status_code)
