FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --no-cache --default-timeout=300 --retries=10 -r requirements.txt

COPY docker_video.py .
COPY yolo11n.pt .
COPY data/video/surveillance.mp4 data/video/surveillance.mp4

CMD ["python", "docker_video.py"]
