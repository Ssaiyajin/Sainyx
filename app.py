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

# ── Load model + vocab together ───────────────────
device = 'cpu'
checkpoint = torch.load('sainyx_v2_full.pt', map_location=device)

chars = checkpoint['chars']
stoi  = checkpoint['stoi']
itos  = {int(k) if isinstance(k, str) else k: v for k, v in checkpoint['itos'].items()}

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

state_dict = checkpoint['model_state_dict']
vocab_size  = state_dict['token_embedding.weight'].shape[0]
model = Sainyx(vocab_size=vocab_size).to(device)
model.load_state_dict(state_dict)
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
    
    # wrap in Q&A format so model knows to answer
    prompt = f"Question: {user_input}\nAnswer:"
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=200)
    
    response = decode(output[0].tolist())
    # strip the prompt, return only the answer
    response = response[len(prompt):]
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)