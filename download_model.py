from huggingface_hub import hf_hub_download
import os

model_path = hf_hub_download(
    repo_id="whii/Swin-Transformer-Pretrained_multilabel-acne",
    filename="99best_acne_swin.pth",
    token=os.environ["HF_TOKEN"]
)
