import torch
import torch.nn.functional as F
from model.gpt import Sainyx, BLOCK_SIZE

# ── Load vocab from dataset ───────────────────────
with open('data/sainyx_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Load model ────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Sainyx(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load('sainyx_v1.pt', map_location=device))
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

    # encode input
    context = torch.tensor(encode(user_input), dtype=torch.long).unsqueeze(0).to(device)
    
    # generate
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=200)
    
    response = decode(output[0].tolist())
    # show only what was generated after input
    response = response[len(user_input):]
    
    print(f"Sainyx: {response}\n")