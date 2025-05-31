from flask import Flask, render_template
import requests
import os
import cv2

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

files = {
    'age_net.caffemodel': '1YAF-VCMHxwmotgjAfhQKpeTcaDOf2PfA',
    'gender_net.caffemodel': '1cEtMSvleOjQkOMVsAa0vGGQW1e0QQbL7',
    'deploy_age.prototxt': 'YOUR_AGE_PROTO_ID',
    'deploy_gender.prototxt': 'YOUR_GENDER_PROTO_ID'
}

for filename, file_id in files.items():
    if not os.path.exists(filename):
        download_file_from_google_drive(file_id, filename)

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
