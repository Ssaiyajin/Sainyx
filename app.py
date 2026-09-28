"""Flask web app for chat, media generation, analysis, and the versioned API.

Model instances are loaded through ``ModelFactory`` so browser routes and API
routes share the same cached checkpoints instead of allocating duplicates.
"""

import os
import io
import base64

import pandas as pd
import torch
from generation.video.diffusion import NoiseScheduler
from torchvision.utils import save_image

from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context

from generation.data_analysis.analyzer import analyze_csv, generate_charts, summarize
from generation.data_analysis.pdf_export import generate_pdf
from generation.data_analysis.scientist import train_model
from generation.image.generate import generate_images
from generation.audio.voice import generate_voice_audio
from api.api import api
from core.models.factory import get_image_model, get_text_model, get_video_model

import config

COMING_SOON_FEATURES = {
    'voice_generation': {
        'label': 'Voice generation',
        'status': 'planned',
        'summary': 'A voice synthesis layer is being prepared for a future release once a trained model is available.',
        'eta': 'Coming soon'
    },
    'style_image_generation': {
        'label': 'Styled image generation',
        'status': 'planned',
        'summary': 'Prompt presets for anime, game assets, and concept art are being structured for later rollout.',
        'eta': 'Coming soon'
    },
    'api_layer': {
        'label': 'API layer',
        'status': 'in_progress',
        'summary': 'A more structured API experience is being organized for future public access.',
        'eta': 'Planned'
    }
}

# Load each checkpoint through the shared cache used by API routes too.
device = config.DEVICE
model, text_vocab = get_text_model()
encode = text_vocab['encode']
itos = text_vocab['itos']

image_result = get_image_model()
diffusion_model = image_result['model'] if image_result else None
diffusion_image_size = image_result['image_size'] if image_result else config.IMAGE_SIZE_DEFAULT
diffusion_timesteps = image_result['timesteps'] if image_result else config.IMAGE_TIMESTEPS_DEFAULT

video_result = get_video_model()
video_model = video_result["model"] if video_result else None
video_image_size = video_result["image_size"] if video_result else config.VIDEO_IMAGE_SIZE_DEFAULT
video_timesteps = video_result["timesteps"] if video_result else config.VIDEO_TIMESTEPS_DEFAULT


# ── Flask app ──────────────────────────────────────
app = Flask(__name__)
app.register_blueprint(api)


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

    if diffusion_model is None:
        return jsonify({'error': 'Sainyx image model is not available. Add the image checkpoint and restart the Space.'}), 503

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
        return jsonify({'error': f'Sainyx model generation failed: {e}'}), 500

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


@app.route('/generate-voice', methods=['POST'])
def generate_voice():
    data = request.get_json(silent=True) or {}
    text = data.get('text', data.get('prompt', ''))
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Text is required'}), 400

    voice = data.get('voice', 'neutral')
    if not isinstance(voice, str):
        return jsonify({'error': 'Voice must be a string'}), 400

    audio = generate_voice_audio(text.strip(), voice=voice.strip() or 'neutral')
    return jsonify({
        'status': 'ready',
        'text': text.strip(),
        'voice': voice.strip() or 'neutral',
        'audio_base64': base64.b64encode(audio).decode('ascii'),
        'mime_type': 'audio/wav',
        'source': 'sainyx-synthetic-voice'
    })


@app.route('/generate-image-styled', methods=['POST'])
def generate_image_styled():
    feature = COMING_SOON_FEATURES['style_image_generation']
    return jsonify({
        'status': 'planned',
        'feature': 'style_image_generation',
        'label': feature['label'],
        'message': feature['summary'],
        'eta': feature['eta']
    })


@app.route('/feature-status', methods=['GET'])
def feature_status():
    return jsonify({
        'status': 'planned',
        'message': 'These capabilities are being prepared and will become available as the product roadmap expands.',
        'features': COMING_SOON_FEATURES
    })


app.run(host='0.0.0.0', port=7860, debug=False)