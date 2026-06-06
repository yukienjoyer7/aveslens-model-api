FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    onnxruntime==1.20.1 \
    python-multipart \
    Pillow \
    pydantic \
    huggingface_hub \
    numpy

COPY app/ app/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
