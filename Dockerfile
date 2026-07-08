FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project code
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]