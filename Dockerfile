# Crop Recommendation web app (Streamlit).
#   docker build -t crop-recommender .
#   docker run --rm -p 8501:8501 crop-recommender     -> http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# catdoc reads legacy .doc reports (see src/report_parser.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends catdoc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY data/original_dataset.csv data/data_dictionary.csv data/
COPY .streamlit/ .streamlit/

# Build the model inside the image from the raw data, so the image never depends on local artifacts.
RUN python -m src.train --skip-cost

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
