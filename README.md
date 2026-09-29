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

**Sainyx V2.0**

An open-source AI platform in development, built with Flask by ssaiyajin.
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

## Stack

| Area | Technologies |
|---|---|
| Web application and API | Python, Flask |
| Model runtime | PyTorch, torchvision |
| Data analysis and modeling | Pandas, NumPy, scikit-learn, Matplotlib |
| PDF reports | ReportLab |
| Browser interface | HTML, CSS, JavaScript, Jinja templates |
| Model checkpoints | Hugging Face Hub |
| Deployment | Docker, Hugging Face Spaces |

## Run Locally

Use Python 3.11 or another version supported by PyTorch. Install PyTorch separately from `requirements.txt` so you can choose the build for your operating system and hardware.

```powershell
git clone https://github.com/Ssaiyajin/Sainyx.git
cd Sainyx
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python app.py
```

The app listens on `http://localhost:7860`. Its text checkpoint is required; image and video features depend on their respective checkpoints. Checkpoints are loaded from Hugging Face when the app starts. Set `HF_TOKEN` if access to a model repository requires authentication.

## REST API

Check the public API status and available model capabilities:

```bash
curl http://localhost:7860/api/v1/status
```

The Space owner must set `SAINYX_API_KEY` in the server environment. Clients send that key in the `X-API-Key` header; status and OpenAPI routes are public. For examples, request formats, asynchronous jobs, and limits, see the [API guide](docs/API.md). The live OpenAPI document is served at `/api/v1/openapi.yaml`.

Example text-generation request:

```bash
curl -X POST http://localhost:7860/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'
```

Generation requests are limited to 64 KiB and 60 per minute per client IP. The asynchronous job queue and rate limits are held in process memory; use a single application worker unless shared storage and rate limiting are configured externally.

## Deployment

The repository includes a Dockerfile configured for Hugging Face Spaces on port `7860`. The deployment workflows run API smoke checks; configure the required `HF_TOKEN` and `SAINYX_API_KEY` GitHub Actions secrets, and set `SAINYX_API_KEY` in the Space secrets as well. See [docs/API.md](docs/API.md) for API key setup and [api/openapi.yaml](api/openapi.yaml) for the API contract.

## Author
Built by **Nihar Sawant** (known online as **ssaiyajin**). — learning in public, scaling in silence. ⚡