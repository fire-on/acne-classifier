from flask import Flask, request, jsonify, render_template
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import onnxruntime as ort
import io
import os
import hashlib
import cloudinary
import cloudinary.uploader
import cloudinary.api
from datetime import datetime

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
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = preprocess(image).unsqueeze(0)  # shape: (1, 3, 224, 224)

        # ONNX는 numpy array 입력 필요
        input_tensor = tensor.numpy()
        input_name = ort_session.get_inputs()[0].name

        # 추론 실행
        outputs = ort_session.run(None, {input_name: input_tensor})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Import your image first.'}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.sigmoid(outputs[0])

        class_probs = {class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))}

        return jsonify({'probs': class_probs})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

        now = datetime.utcnow()
        deadline = datetime(2025, 7, 8)

        if max_prob >= 0.9 and now < deadline:
            usage = cloudinary.api.usage()
            remaining_storage = usage['storage']['limit'] - usage['storage']['usage']

            if remaining_storage >= 10 * 1024 * 1024:
                file.seek(0)
                img_hash = hashlib.sha256(file.read()).hexdigest()
                file.seek(0)

                resized = image.resize((1024, 1024))
                buffer = io.BytesIO()
                resized.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)

                if buffer.getbuffer().nbytes <= 5 * 1024 * 1024:
                    cloudinary.uploader.upload(
                        buffer,
                        public_id=img_hash,
                        overwrite=False,
                        unique_filename=False
                    )

        return jsonify({'probs': class_probs})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/consent', methods=['POST'])
def consent():
    try:
        if 'file' not in request.files:
            # 여기서도 에러 리턴은 해주지만, 클라이언트는 무조건 감사 메시지 노출
            return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        now = datetime.utcnow()
        deadline = datetime(2025, 7, 8)

        if now < deadline:
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
        # 항상 성공 리턴 (결과는 클라이언트가 알 필요 없음)
        return jsonify({'success': True})

    except Exception as e:
        # 내부 에러 로깅만, 클라이언트에는 무조건 성공 리턴
        print(f"[consent error] {e}")
        return jsonify({'success': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
