FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose port
EXPOSE 7860

# run app
CMD ["python", "app.py"]