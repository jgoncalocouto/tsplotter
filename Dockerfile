# Minimal Dockerfile for the Time-Series Plot Helper (Streamlit + Plotly)
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy your app code (ensure your main file is named main_tsplotter.py)
COPY . /app

EXPOSE 8501
ENV PORT=8501

# Start Streamlit
CMD ["bash", "-lc", "streamlit run main.py --server.port ${PORT} --server.address 0.0.0.0"]
