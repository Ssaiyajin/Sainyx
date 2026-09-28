---
title: Sainyx
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Sainyx 🔥

An open-source AI platform built from scratch by ssaiyajin.
Language • Data Analysis • Image • Video — all in one, all free.

## Live Demo
👉 [Try Sainyx on Hugging Face](https://huggingface.co/spaces/ssaiyajin/Sainyx)

## Vision
Sainyx is not just a chatbot. It's a full AI platform:
- 🧠 LLM built from scratch (PyTorch)
- 📊 Data science gateway (Pandas, Matplotlib, PDF reports)
- 🎨 Image generation (Stable Diffusion) — coming soon
- 🎬 Video generation (ModelScope) — coming soon
- 🌐 REST API for developers

## Current Status
- ✅ GPT transformer built from scratch (5M parameters)
- ✅ Trained on custom anime/gaming/DBZ dataset (1.5M characters)
- ✅ Jarvis HUD + MUI aura animation UI
- ✅ Data analysis module with 5 chart types
- ✅ PDF report export
- ✅ Deployed on Hugging Face
- 🔲 Image generation module
- 🔲 Video generation module
- ✅ REST API (`/api/v1/status` and `/api/v1/generate`)

## REST API

The versioned API supports text, image, video-frame, and voice-demo generation. Check live capabilities without authentication:

```bash
curl http://localhost:7860/api/v1/status
```

API clients need only the base URL and an API key from the Sainyx owner; they do not need Hugging Face access. The owner configures `SAINYX_API_KEY` in the Space settings and shares that key with authorized clients. The OpenAPI 3.1 spec is served at `/api/v1/openapi.yaml`; the [API quickstart](docs/API.md) has curl and Python examples.

```bash
curl -X POST http://localhost:7860/api/v1/generate \
	-H "X-API-Key: $SAINYX_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'
```

Generation supports deterministic seeds where applicable, JSON/base64 or downloadable binary media, and queued jobs with status/result endpoints. Requests are capped at 64 KiB and 60 per minute per client IP. Image generation is currently unconditional (the prompt is metadata only); video returns a single PNG frame; voice currently returns a WAV tone demo, not spoken TTS.

GitHub Actions runs a real text-generation API smoke test on pull requests targeting `main` or `Dev`, and blocks the Hugging Face deployment workflows unless it passes. After production deployment to Hugging Face, the `main` workflow also waits for the public API to restart and sends an authenticated sample request to verify the live endpoint. The Actions log includes the sample prompt and generated reply. Configure `HF_TOKEN` and `SAINYX_API_KEY` as GitHub Actions secrets; the model repository must be public or `HF_TOKEN` must have access. Set the same `SAINYX_API_KEY` in the Hugging Face Space secrets.

## Stack
All free. No paid APIs. Ever.

| Module | Tech |
|---|---|
| LLM | PyTorch |
| Data Analysis | Pandas, Matplotlib |
| PDF Export | ReportLab |
| UI | Flask + HTML |
| Deploy | Hugging Face Spaces |
| Training | Kaggle Free GPU |

## Run Locally
```bash
git clone https://github.com/Ssaiyajin/Sainyx.git
cd Sainyx
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Author
Built by ssaiyajin — learning in public, scaling in silence. ⚡