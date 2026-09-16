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

Check available generators without authentication:

```bash
curl http://localhost:7860/api/v1/status
```

Set `SAINYX_API_KEY` on the server, then send the key with generation requests:

```bash
curl -X POST http://localhost:7860/api/v1/generate \
	-H "X-API-Key: $SAINYX_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"type":"text","prompt":"Who is Goku?","max_tokens":80}'
```

Supported generation types are `text`, `image`, `video`, and `voice`. Video availability depends on the trained checkpoint; voice uses Sainyx's lightweight WAV generator.

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