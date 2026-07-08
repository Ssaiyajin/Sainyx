import os
# ── Load model + vocab together ───────────────────
device = 'cpu'

model_path = 'sainyx_v2_full.pt'

if not os.path.exists(model_path):
    print("Downloading model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id='ssaiyajin/sainyx-model',
            filename='sainyx_v2_full.pt',
            repo_type='model'
        )
        print(f"✅ Model downloaded to: {model_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)