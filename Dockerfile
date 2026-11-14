# API image:  docker build -t aqi-api . && docker run -p 8000:8000 aqi-api
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
COPY airq ./airq
COPY api ./api
COPY models ./models
COPY data/processed ./data/processed
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
