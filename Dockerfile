FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Force a clean installation of the entire PyTorch trio to avoid conflicts
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip uninstall -y torch torchaudio torchvision && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
