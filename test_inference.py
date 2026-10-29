"""Quick inference test for the live Flask API."""
import urllib.request
import json

boundary = b'----WebKitFormBoundary7MA4YWxkTrZu0gW'

with open('sample_images/sample_apple_scab.jpg', 'rb') as f:
    img_data = f.read()

body = (
    b'--' + boundary + b'\r\n'
    b'Content-Disposition: form-data; name="image"; filename="sample_apple_scab.jpg"\r\n'
    b'Content-Type: image/jpeg\r\n\r\n' + img_data + b'\r\n'
    b'--' + boundary + b'--\r\n'
)

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary.decode()}
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    print('Success:', data['success'])
    print('Predicted class:', data['predicted_class'])
    print('Display name:', data['display_name'])
    print('Crop:', data['crop'])
    print('Disease:', data['disease'])
    print('Is healthy:', data['is_healthy'])
    print('Confidence: {:.2f}%'.format(data['confidence_percentage']))
    print('Top 3 predictions:')
    for p in data['top_predictions']:
        print(f"  #{p['rank']}: {p['display_name']} ({p['confidence_percent']}%)")
    print('\nINFERENCE TEST PASSED')
except Exception as e:
    print('ERROR:', e)
