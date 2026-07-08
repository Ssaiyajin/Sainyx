import os
import torch

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
            repo_type='model',
            token=os.environ.get('HF_TOKEN')
        )
        print(f"✅ Model downloaded to: {model_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)
print("✅ Checkpoint loaded")

chars = checkpoint['chars']
stoi  = checkpoint['stoi']
itos  = {int(k) if isinstance(k, str) else k: v for k, v in checkpoint['itos'].items()}

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

state_dict = checkpoint['model_state_dict']
vocab_size  = state_dict['token_embedding.weight'].shape[0]
print(f"Vocab size: {vocab_size}")

model = Sainyx(vocab_size=vocab_size).to(device)
print("✅ Model created")
model.load_state_dict(state_dict)
print("✅ Weights loaded")
model.eval()
print("🔥 Sainyx ready!")
checkpoint = torch.load(model_path, map_location=device)