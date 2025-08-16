FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Force a clean installation of libraries to avoid conflicts
# First, copy the requirements file
COPY requirements.txt .

# Then, run a multi-step pip command to ensure a clean state
RUN pip install --upgrade pip && \
    pip uninstall -y torch torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy the handler file
COPY handler.py .

# Command to run the handler
CMD ["python", "-u", "handler.py"]
