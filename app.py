from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
import os
import hashlib
import cloudinary
import cloudinary.uploader
import cloudinary.api
from datetime import datetime
import requests

app = Flask(__name__)

# Cloudinary 설정
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

# 클래스 라벨
class_names = ['Pimples', 'blackhead', 'conglobata', 'crystanlline', 'cystic',
               'folliculitis', 'keloid', 'milium', 'papular', 'purulent']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Import your image first.'}), 400

        file = request.files['file']
        files = {'file': (file.filename, file.read(), file.mimetype)}

        # Spaces API
        response = requests.post("https://whii-spaces-predict-api.hf.space/predict", files=files)

        if response.status_code != 200:
            return jsonify({'error': 'Model server error'}), 500

        result = response.json()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/consent', methods=['POST'])
def consent():
    try:
        if 'file' not in request.files:
            return jsonify({'success': True})

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        max_prob = float(request.form.get('max_prob', 0))  # 클라이언트에서 전달받은 값
        now = datetime.utcnow()
        deadline = datetime(2025, 7, 8)

        # 조건 충족 시만 업로드 시도
        if max_prob >= 0.9 and now < deadline:
            usage = cloudinary.api.usage()
            storage_used = usage['storage']['usage']
            MAX_STORAGE = 15 * 1024 * 1024 * 1024  # 15GB

            print(f"[Cloudinary] 사용량: {storage_used} / {MAX_STORAGE} bytes")

            if storage_used < MAX_STORAGE:
                file.seek(0)
                img_hash = hashlib.sha256(file.read()).hexdigest()
                file.seek(0)

                width, height = image.size
                buffer = io.BytesIO()

                if width > 1024 or height > 1024:
                    resized = image.resize((1024, 1024))
                    resized.save(buffer, format='JPEG', quality=85)
                else:
                    image.save(buffer, format='JPEG', quality=85)

                buffer.seek(0)

                if buffer.getbuffer().nbytes <= 5 * 1024 * 1024:
                    cloudinary.uploader.upload(
                        buffer,
                        public_id=img_hash,
                        overwrite=False,
                        unique_filename=False
                    )

        return jsonify({'success': True})

    except Exception as e:
        print(f"[Consent Error] {e}")
        return jsonify({'success': True})  # 클라이언트에는 항상 성공 메시지

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
