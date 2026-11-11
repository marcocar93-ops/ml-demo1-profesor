FROM python:3.10-slim
WORKDIR /app
COPY requirements_mlops.txt .
RUN pip install --no-cache-dir -r requirements_mlops.txt
COPY app/ ./app/
COPY models/ ./models/
COPY config.json ./config.json
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
