import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict

# ── Make project root importable (chat.py now lives in generation/text/) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.gpt import Sainyx, BLOCK_SIZE

device = 'cpu'

# ── Load model + vocab together ───────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sainyx_v2_full.pt')
checkpoint = torch.load(MODEL_PATH, map_location=device)

chars = checkpoint['chars']
stoi  = checkpoint['stoi']
itos  = {int(k) if isinstance(k, str) else k: v for k, v in checkpoint['itos'].items()}

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

state_dict = checkpoint['model_state_dict']
vocab_size  = state_dict['token_embedding.weight'].shape[0]
print(f"Model vocab size: {vocab_size}")

model = Sainyx(vocab_size=vocab_size).to(device)
model.load_state_dict(state_dict)
model.eval()

print("🔥 Sainyx is online — type anything!")
print("   Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        print("Sainyx: later. ⚡")
        break
    if not user_input:
        continue

    context = torch.tensor(encode(user_input), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=200)

    response = decode(output[0].tolist())
    response = response[len(user_input):]
    print(f"Sainyx: {response}\n")