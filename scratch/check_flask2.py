import sys, os
# Add project root (parent directory) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from app import app
client = app.test_client()
resp = client.get('/')
print('Status:', resp.status_code)
