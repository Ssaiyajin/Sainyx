import os

# ── Debug — find model ────────────────────────────
def find_model():
    search_paths = [
        'sainyx_v2_full.pt',
        '/app/sainyx_v2_full.pt',
        '/home/user/app/sainyx_v2_full.pt',
    ]
    # also search current directory
    print(f"Current dir: {os.getcwd()}")
    print(f"Files in current dir: {os.listdir('.')}")
    print(f"Files in /app: {os.listdir('/app')}")
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Found model at: {path}")
            return path
    
    print("❌ Model not found in any path!")
    return None

model_path = find_model()
checkpoint = torch.load(model_path, map_location=device)