FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
# The registry is mounted at runtime so the image is model-agnostic:
#   docker run -p 8000:8000 -v $(pwd)/model_registry:/app/model_registry credit-pd
EXPOSE 8000
CMD ["uvicorn", "credit_risk.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
