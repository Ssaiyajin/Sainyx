FROM python:3.11-slim

WORKDIR /app

# install dependencies first (cached layer - only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code after (changes here don't invalidate pip cache)
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]