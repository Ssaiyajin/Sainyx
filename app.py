
import torch
import os
from model.gpt import Sainyx, BLOCK_SIZE
from flask import Flask, render_template, request, jsonify
from data_analysis.analyzer import analyze_csv, generate_charts, summarize

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
        output = model.generate(context, max_new_tokens=80)
    
    response = decode(output[0].tolist())
    # strip the prompt, return only the answer
    response = response[len(prompt):]
    
    return jsonify({'response': response})

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files supported'})
    
    # save temporarily
    os.makedirs('uploads', exist_ok=True)
    filepath = f"uploads/{file.filename}"
    file.save(filepath)
    
    # analyze
    df, report = analyze_csv(filepath)
    charts = generate_charts(df)
    summary = summarize(report)
    
    # cleanup
    os.remove(filepath)
    
    return jsonify({
        'summary': summary,
        'report': report,
        'charts': charts
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)