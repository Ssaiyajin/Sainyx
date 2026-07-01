import torch
import torch.nn.functional as F
from collections import OrderedDict
from model.gpt import Sainyx, BLOCK_SIZE

# ── Load vocab from dataset ───────────────────────
with open('data/sainyx_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Load state dict first ─────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('sainyx_v2.pt', map_location=device)

# strip 'module.' prefix if saved with DataParallel
if list(state_dict.keys())[0].startswith('module.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    state_dict = new_state_dict

# ── Get vocab size FROM checkpoint ────────────────
vocab_size = state_dict['token_embedding.weight'].shape[0]
print(f"Model vocab size: {vocab_size}")

# ── Create model with correct vocab size ──────────
model = Sainyx(vocab_size=vocab_size).to(device)
model.load_state_dict(state_dict)
model.eval()

print("🔥 Sainyx is online — type anything!")
print("   Type 'quit' to exit\n")

# ── Chat loop ─────────────────────────────────────
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