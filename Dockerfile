# Lightweight serving image: ONNX Runtime + FastAPI + Streamlit, no PyTorch.
# Models and the replay session are mounted from outputs/ at run time.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY src ./src
COPY app ./app

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
