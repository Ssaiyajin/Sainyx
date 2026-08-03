import os
import io
import base64
import urllib.parse

import pandas as pd
import torch
from generation.video.diffusion import NoiseScheduler
from torchvision.utils import save_image

from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from huggingface_hub import InferenceClient, hf_hub_download

from model.gpt import Sainyx, BLOCK_SIZE
from generation.data_analysis.analyzer import analyze_csv, generate_charts, summarize
from generation.data_analysis.pdf_export import generate_pdf
from generation.data_analysis.scientist import train_model
from generation.image.generate import load_model as load_diffusion_model, generate_images
import config

import requests as req


# ── Load text model + vocab together ───────────────
device = config.DEVICE

model_path = config.TEXT_MODEL_LOCAL_PATH

if not os.path.exists(model_path):
    print("Downloading model from HuggingFace...")
    try:
        model_path = hf_hub_download(
            repo_id=config.TEXT_MODEL_REPO_ID,
            filename=config.TEXT_MODEL_FILENAME,
            repo_type='model',
            token=config.HF_TOKEN
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


# ── Load diffusion model ────────────────────────────
diffusion_model_path = config.IMAGE_MODEL_LOCAL_PATH

if not os.path.exists(diffusion_model_path):
    print("Downloading diffusion model from HuggingFace...")
    try:
        diffusion_model_path = hf_hub_download(
            repo_id=config.IMAGE_MODEL_REPO_ID,
            filename=config.IMAGE_MODEL_FILENAME,
            repo_type='model',
            token=config.HF_TOKEN
        )
        print(f"✅ Diffusion model downloaded to: {diffusion_model_path}")
    except Exception as e:
        print(f"❌ Diffusion model download failed: {e}")
        diffusion_model_path = None

diffusion_model = None
diffusion_image_size = config.IMAGE_SIZE_DEFAULT
diffusion_timesteps = config.IMAGE_TIMESTEPS_DEFAULT

if diffusion_model_path:
    diffusion_model, diffusion_image_size, diffusion_timesteps = load_diffusion_model(
        diffusion_model_path, device=device
    )


# ── Image generation client (Pollinations fallback) ─
image_client = InferenceClient(token=config.HF_TOKEN)

# ── Load video model (Tier 1, unconditional, may not exist yet) ─────
from core.models.factory import get_video_model

video_result = get_video_model()
video_model = video_result["model"] if video_result else None
video_image_size = video_result["image_size"] if video_result else config.VIDEO_IMAGE_SIZE_DEFAULT
video_timesteps = video_result["timesteps"] if video_result else config.VIDEO_TIMESTEPS_DEFAULT


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

    use_own_model = data.get('use_own_model', False)

    if use_own_model and diffusion_model is not None:
        try:
            samples = generate_images(
                diffusion_model, image_size=diffusion_image_size,
                timesteps=diffusion_timesteps, num_images=1, device=device
            )
            buffer = io.BytesIO()
            save_image(samples, buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return jsonify({'image': img_b64, 'prompt': prompt, 'source': 'sainyx-diffusion'})
        except Exception as e:
            return jsonify({'error': f'Sainyx model generation failed: {e}'})

    # Fallback: Pollinations
    enhanced = f"{prompt}, digital art, high quality, detailed, 4k, artstation"
    try:
        encoded_prompt = urllib.parse.quote(enhanced)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        response = req.get(url, timeout=60)
        if response.status_code == 200:
            img_b64 = base64.b64encode(response.content).decode('utf-8')
            return jsonify({'image': img_b64, 'prompt': enhanced, 'source': 'pollinations'})
        else:
            return jsonify({'error': f'Generation failed ({response.status_code})'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/generate-video', methods=['POST'])
def generate_video():
    if video_model is None:
        return jsonify({'error': 'Video model not available yet — no checkpoint pushed from Kaggle.'})

    try:
        scheduler = NoiseScheduler(timesteps=video_timesteps, device=device)
        samples = scheduler.sample(
            video_model, image_size=video_image_size,
            batch_size=1, channels=3, device=device
        )
        samples = (samples.clamp(-1, 1) + 1) / 2  # denormalize

        buffer = io.BytesIO()
        save_image(samples, buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({'image': img_b64, 'source': 'sainyx-video-tier1'})
    except Exception as e:
        return jsonify({'error': f'Video generation failed: {e}'})


app.run(host='0.0.0.0', port=7860, debug=False)