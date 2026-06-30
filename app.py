from flask import Flask, render_template, request, jsonify
import torch
from model.gpt import Sainyx, BLOCK_SIZE

app = Flask(__name__)

# ── Load vocab ─────────────────────────────────────
with open('data/sainyx_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Load model ─────────────────────────────────────
device = 'cpu'
model = Sainyx(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load('sainyx_v1.pt', map_location=device))
model.eval()

# ── Routes ──────────────────────────────────────────
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    
    if not user_input:
        return jsonify({'response': '...'})
    
    context = torch.tensor(encode(user_input), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=200)
    
    response = decode(output[0].tolist())
    response = response[len(user_input):]
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)