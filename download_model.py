from huggingface_hub import hf_hub_download
import os

model_path = hf_hub_download(
    repo_id="your-username/your-model-repo",
    filename="model.pth",
    token=os.environ["HF_TOKEN"]
)
