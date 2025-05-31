from flask import Flask, render_template
import requests
import os

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if not os.path.exists('age_net.caffemodel'):
    download_file_from_google_drive('1YAF-VCMHxwmotgjAfhQKpeTcaDOf2PfA', 'age_net.caffemodel')

if not os.path.exists('gender_net.caffemodel'):
    download_file_from_google_drive('1cEtMSvleOjQkOMVsAa0vGGQW1e0QQbL7', 'gender_net.caffemodel')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)