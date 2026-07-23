import os
import io
import base64
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify, send_file
from torchvision.utils import save_image

# Import refactored components
from config import FLASK_CONFIG, DEVICE, INFERENCE_CONFIG, HF_INFERENCE_CONFIG
from core import ModelFactory, get_vocab
from data_analysis.analyzer import analyze_csv, generate_charts, summarize
from data_analysis.pdf_export import generate_pdf
from data_analysis.scientist import train_model


# ── Initialize Flask App ───────────────────────────
app = Flask(__name__)
app.config.update(FLASK_CONFIG)

# ── Load Models on Startup ──────────────────────────
print("Loading models...")
try:
    text_model, text_vocab = ModelFactory.load_text_model()
    print("✅ Text model loaded")
except Exception as e:
    print(f"❌ Text model loading failed: {e}")
    text_model = None
    text_vocab = None

try:
    diffusion_result = ModelFactory.load_diffusion_model()
    diffusion_model = diffusion_result['model'] if diffusion_result else None
    print("✅ Diffusion model loaded")
except Exception as e:
    print(f"⚠️  Diffusion model loading failed: {e}")
    diffusion_model = None

# Fallback to HuggingFace inference
try:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(token=os.environ.get('HF_TOKEN'))
    print("✅ HuggingFace inference client ready")
except Exception as e:
    print(f"⚠️  HuggingFace client failed: {e}")
    hf_client = None


# ── Inference Functions ────────────────────────────

def generate_text(prompt: str, max_tokens: int = 200) -> str:
    """Generate text using Sainyx model"""
    if not text_model or not text_vocab:
        return "Text model not available"
    
    try:
        encode = text_vocab['encode']
        decode = text_vocab['decode']
        
        context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = text_model.generate(context, max_new_tokens=max_tokens)
        
        response = decode(output[0].tolist())
        return response[len(prompt):]  # Remove prompt from output
    except Exception as e:
        print(f"Text generation error: {e}")
        return f"Error: {str(e)}"


def generate_images(prompt: str, num_images: int = 1):
    """Generate images using diffusion model or HuggingFace"""
    if diffusion_model:
        try:
            # Use local diffusion model
            from generation.image.generate import generate_images as generate_local
            return generate_local(prompt, num_images)
        except Exception as e:
            print(f"Local image generation failed: {e}")
    
    # Fallback to HuggingFace
    if hf_client:
        try:
            images = []
            for _ in range(num_images):
                image = hf_client.text_to_image(prompt)
                images.append(image)
            return images
        except Exception as e:
            print(f"HuggingFace image generation failed: {e}")
    
    return None


# ── Web Routes ─────────────────────────────────────

@app.route('/')
def index():
    """Homepage"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    data = request.json
    user_input = data.get('message', '')
    
    response = generate_text(user_input, max_tokens=INFERENCE_CONFIG['max_tokens'])
    
    return jsonify({
        'response': response,
        'model': 'Sainyx v2'
    })


@app.route('/api/generate/images', methods=['POST'])
def generate_image_endpoint():
    """Image generation endpoint"""
    data = request.json
    prompt = data.get('prompt', '')
    num_images = data.get('num_images', 1)
    
    images = generate_images(prompt, num_images)
    
    if images:
        # Convert to base64 for response
        image_b64_list = []
        for img in images:
            buffer = io.BytesIO()
            if isinstance(img, torch.Tensor):
                save_image(img, buffer, format='PNG')
            else:
                img.save(buffer, format='PNG')
            buffer.seek(0)
            image_b64_list.append(base64.b64encode(buffer.getvalue()).decode())
        
        return jsonify({
            'images': image_b64_list,
            'success': True
        })
    else:
        return jsonify({
            'error': 'Image generation failed',
            'success': False
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Data analysis endpoint"""
    file = request.files.get('file')
    
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Analyze
        analysis = analyze_csv(df)
        charts = generate_charts(df)
        summary = summarize(df)
        
        return jsonify({
            'analysis': analysis,
            'charts': charts,
            'summary': summary,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """PDF export endpoint"""
    data = request.json
    analysis_data = data.get('analysis', {})
    
    try:
        pdf_buffer = generate_pdf(analysis_data)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='sainyx_analysis.pdf'
        )
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """Train model endpoint"""
    file = request.files.get('file')
    
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    try:
        df = pd.read_csv(file)
        model_info = train_model(df)
        
        return jsonify({
            'model': model_info,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'models_loaded': {
            'text': text_model is not None,
            'diffusion': diffusion_model is not None,
        }
    })


# ── Error Handlers ─────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


# ── Main ───────────────────────────────────────────

if __name__ == '__main__':
    print(f"Starting Sainyx on {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print(f"Device: {DEVICE}")
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )
