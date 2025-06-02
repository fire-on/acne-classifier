from flask import Flask, request, jsonify, render_template
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import io
import os
app = Flask(__name__)

# Set class_names
class_names = ['Pimples', 'blackhead', 'conglobata', 'crystanlline', 'cystic', 'folliculitis', 'keloid', 'milium', 'papular', 'purulent']

class SwinClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SwinClassifier, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)

# cuda check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = hf_hub_download(
    repo_id="your-username/your-model-repo",
    filename="best_Acne_swin_model.pth",
    token=os.environ["HF_TOKEN"])

# Load the Model
model = SwinClassifier(num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
        return jsonify({'probs': class_probs})
    except Exception as e:
        print("Error:", str(e))  # 터미널 출력
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
