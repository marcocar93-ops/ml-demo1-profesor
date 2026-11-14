FROM python:3.10-slim
WORKDIR /app
COPY requirements_mlops.txt .
RUN pip install --no-cache-dir -r requirements_mlops.txt
COPY app/ ./app/
COPY models/rf_final_b2_pipeline.pkl ./models/
COPY models/random_forest.pkl ./models/
COPY models/label_encoder.pkl ./models/
COPY models/rf_final_b2_pipeline.onnx ./models/
COPY config.json ./config.json
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
