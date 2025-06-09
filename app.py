from flask import Flask, request, jsonify, render_template
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
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
class_names = ['Pimples', 'blackhead', 'conglobata', 'crystanlline', 'cystic', 'folliculitis', 'keloid', 'milium', 'papular', 'purulent']

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SwinClassifier, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

try:
    model_path = hf_hub_download(
        repo_id="whii/Swin-Transformer-Pretrained_multilabel-acne",
        filename="99best_acne_swin.pth",
        token=os.environ["HF_TOKEN"],
        local_files_only=True  # True면 캐시 없으면 실패함
    )
except EntryNotFoundError:
    print("Model not cached yet. Trying to download...")
    model_path = hf_hub_download(
        repo_id="whii/Swin-Transformer-Pretrained_multilabel-acne",
        filename="99best_acne_swin.pth",
        token=os.environ["HF_TOKEN"],
        local_files_only=False  # 새로 다운로드 허용
    )

# 모델 로드
model = SwinClassifier(num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 전처리 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.sigmoid(outputs[0])

        class_probs = {class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))}

        # 조건 확인 후 Cloudinary 저장 시도
        max_prob = max(probs).item()
        now = datetime.utcnow()
        deadline = datetime(2025, 7, 8)

        if max_prob >= 0.9 and now < deadline:
            usage = cloudinary.api.usage()
            remaining_storage = usage['storage']['limit'] - usage['storage']['usage']

            if remaining_storage >= 10 * 1024 * 1024:  # 10MB 이상 여유 있을 때
                # 해시 생성 (중복 업로드 방지)
                file.seek(0)
                img_hash = hashlib.sha256(file.read()).hexdigest()
                file.seek(0)

                # 이미지 리사이즈 및 저장 준비
                resized = image.resize((1024, 1024))
                buffer = io.BytesIO()
                resized.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)

                if buffer.getbuffer().nbytes <= 5 * 1024 * 1024:  # 5MB 이하일 경우만 업로드
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
