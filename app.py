import os
import io
import torch
from model.gpt import Sainyx, BLOCK_SIZE
from flask import Flask, render_template, request, jsonify, send_file
from data_analysis.analyzer import analyze_csv, generate_charts, summarize
from data_analysis.pdf_export import generate_pdf

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

# ── Flask app ──────────────────────────────────────
app = Flask(__name__)

# ── Routes ────────────────────────────────────────
@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({'response': '...'})
    prompt = f"Question: {user_input}\nAnswer:"
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=80)
    response = decode(output[0].tolist())
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
    os.makedirs('uploads', exist_ok=True)
    filepath = f"uploads/{file.filename}"
    file.save(filepath)
    df, report = analyze_csv(filepath)
    charts = generate_charts(df)
    summary = summarize(report)
    os.remove(filepath)
    return jsonify({
        'summary': summary,
        'report': report,
        'charts': [{'title': t, 'data': d} for t, d in charts]
    })

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    data = request.json
    pdf_bytes = generate_pdf(
        data['report'],
        data['summary'],
        data['charts']
    )
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='sainyx_report.pdf'
    )

app.run(host='0.0.0.0', port=7860, debug=False)