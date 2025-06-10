from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms
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

# ONNX 모델 다운로드 및 로딩
try:
    model_path = hf_hub_download(
        repo_id="whii/Swin-Transformer-Pretrained_multilabel-acne",
        filename="model.onnx",
        token=os.environ["HF_TOKEN"],
        local_files_only=True
    )
except EntryNotFoundError:
    print("Model not cached. Downloading...")
    model_path = hf_hub_download(
        repo_id="whii/Swin-Transformer-Pretrained_multilabel-acne",
        filename="model.onnx",
        token=os.environ["HF_TOKEN"],
        local_files_only=False
    )

# ONNX 세션 생성
ort_session = ort.InferenceSession(model_path)

# 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

        if max_prob >= 0.9 and now < deadline:
            usage = cloudinary.api.usage()
            remaining_storage = usage['storage']['limit'] - usage['storage']['usage']

            if remaining_storage >= 10 * 1024 * 1024:
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
        return jsonify({'success': True})  # 에러가 나도 사용자에겐 무조건 성공 처리

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
