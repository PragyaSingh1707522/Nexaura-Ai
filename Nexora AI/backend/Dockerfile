FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; else pip install --no-cache-dir -r requirements.txt; fi
COPY app ./app
COPY models ./models
RUN mkdir -p /app/data /app/reports
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
