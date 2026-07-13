import os
import io
import pandas as pd
import base64
import requests as req
import torch
from model.gpt import Sainyx, BLOCK_SIZE
from flask import Flask, render_template, request, jsonify, send_file
from data_analysis.analyzer import analyze_csv, generate_charts, summarize
from data_analysis.pdf_export import generate_pdf
from data_analysis.scientist import train_model

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

from flask import Response, stream_with_context

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({'response': '...'})

    prompt = f"Question: {user_input}\nAnswer:"
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    def generate_stream():
        with torch.no_grad():
            idx = context.clone()
            for _ in range(80):
                idx_cond = idx[:, -32:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
                char = itos.get(next_token.item(), '?')
                yield f"data: {char}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate_stream()),
        mimetype='text/event-stream'
    )

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

@app.route('/scientist', methods=['POST'])
def scientist():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    target = request.form.get('target', '')
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files supported'})
    
    if not target:
        return jsonify({'error': 'No target column selected'})
    
    os.makedirs('uploads', exist_ok=True)
    filepath = f"uploads/{file.filename}"
    file.save(filepath)
    
    df = pd.read_csv(filepath)
    os.remove(filepath)
    
    if target not in df.columns:
        return jsonify({'error': f'Column {target} not found'})
    
    result = train_model(df, target)
    return jsonify(result)

@app.route('/scientist-columns', methods=['POST'])
def scientist_columns():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})
    file = request.files['file']
    df = pd.read_csv(file)
    return jsonify({'columns': list(df.columns)})

@app.route('/scientist-page')
def scientist_page():
    return render_template('scientist.html')

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

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '').strip()

    if not prompt:
        return jsonify({'error': 'No prompt provided'})

    # enhance for better quality
    enhanced = f"{prompt}, digital art, high quality, detailed, 4k, artstation"

    try:
        response = req.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
            headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"},
            json={"inputs": enhanced},
            timeout=120
        )

        if response.status_code == 200:
            img_b64 = base64.b64encode(response.content).decode('utf-8')
            return jsonify({'image': img_b64, 'prompt': enhanced})
        elif response.status_code == 503:
            return jsonify({'error': 'Model is loading, please try again in 20 seconds'})
        else:
            return jsonify({'error': f'Generation failed ({response.status_code})'})

    except Exception as e:
        return jsonify({'error': str(e)})


        
app.run(host='0.0.0.0', port=7860, debug=False)