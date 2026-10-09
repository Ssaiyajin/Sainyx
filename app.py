"""Flask web app for chat, media generation, analysis, and the versioned API.

Model instances are loaded through ``ModelFactory`` so browser routes and API
routes share the same cached checkpoints instead of allocating duplicates.
"""

import os
import io
import time
import base64

import pandas as pd
import torch
from generation.video.render import generate_clip_gif
from torchvision.utils import save_image

from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context

from generation.data_analysis.analyzer import analyze_csv, generate_charts, summarize
from generation.data_analysis.pdf_export import generate_pdf
from generation.data_analysis.scientist import train_model
from generation.image.generate import generate_images
from generation.audio.voice import generate_voice_audio
from api.api import api
from generation.text.retrieval import get_default_store, smalltalk
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

# Text chat answers come from stored text first (see generation/text/retrieval.py).
qa_store = get_default_store()
TEXT_FALLBACK = os.environ.get('SAINYX_TEXT_FALLBACK', 'decline').lower()
TEXT_DECLINE_MESSAGE = (
    "I don't have a reliable answer for that. I can answer questions about Dragon Ball, anime and the games "
    "I was trained on, so try asking about a character, series or game by name."
)


# ── Flask app ──────────────────────────────────────
app = Flask(__name__)
app.register_blueprint(api)


# ── Routes ────────────────────────────────────────
@app.route('/')
def home():
    return render_template('chat.html')


def _sse(text):
    return f"data: {text.replace(chr(10), ' ')}\n\n"


def _stream_text(text, chunk=4, delay=0.012):
    """Send a finished answer a few characters at a time so the UI keeps its typing effect."""
    for i in range(0, len(text), chunk):
        yield _sse(text[i:i + chunk])
        time.sleep(delay)
    yield "data: [DONE]\n\n"


def _model_answer(user_input):
    """Last-resort answer from the 57M model: full context window, low temperature, stops at the answer's end."""
    prompt = f"Question: {user_input}\nAnswer:"
    idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    text = ""
    with torch.no_grad():
        for _ in range(260):
            logits, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / 0.2
            top = torch.topk(logits, 10)[0]
            logits[logits < top[:, [-1]]] = -float('inf')
            next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            text += itos.get(next_token.item(), '?')
            if '\n\n' in text or 'Question:' in text:
                break
    return text.split('\n\n')[0].split('Question:')[0].strip()


@app.route('/chat', methods=['POST'])
def chat():
    user_input = (request.get_json(silent=True) or {}).get('message', '').strip()
    if not user_input:
        return jsonify({'response': '...'})

    # 1. Greetings, thanks and "what can you do" get a fixed reply.
    reply = smalltalk(user_input, qa_store)
    pair = None if reply else qa_store.compare(user_input)
    hit = None if (reply or pair) else qa_store.answer(user_input)

    # 2. Answer from stored Wikipedia / hand-written text when we can match the question.
    if reply:
        pass
    elif pair:
        reply = f"{pair[0].matched}: {pair[0].text} {pair[1].matched}: {pair[1].text}"
    elif hit is not None:
        reply = hit.text
        if hit.corrected:
            reply = f"Showing results for {hit.matched}. {reply}"
    # 3. Otherwise either say so (default) or let the small model try (SAINYX_TEXT_FALLBACK=model).
    elif TEXT_FALLBACK == 'model':
        reply = _model_answer(user_input) or TEXT_DECLINE_MESSAGE
    else:
        near = qa_store.suggest(user_input)
        reply = TEXT_DECLINE_MESSAGE
        if near:
            reply += " Did you mean " + (", ".join(near[:-1]) + " or " + near[-1] if len(near) > 1 else near[0]) + "?"

    return Response(
        stream_with_context(_stream_text(reply)),
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
    if video_result is None:
        return jsonify({'error': 'Video model not available yet — no checkpoint pushed from Kaggle.'})

    try:
        gif_bytes = generate_clip_gif(video_result, device)
        gif_b64 = base64.b64encode(gif_bytes).decode('utf-8')
        return jsonify({'gif': gif_b64, 'frames': video_result['clip_len'], 'source': 'sainyx-video'})
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