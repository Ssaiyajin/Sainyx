FROM python:3.11-slim

WORKDIR /app

# espeak-ng is needed by Kokoro's phonemizer, libsndfile1 by soundfile
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# install torch/torchvision as CPU-only wheels first - HF Spaces free tier has no GPU,
# and the default PyPI build pulls large unused CUDA binaries that slow down builds
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install dependencies first (cached layer - only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code after (changes here don't invalidate pip cache)
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]