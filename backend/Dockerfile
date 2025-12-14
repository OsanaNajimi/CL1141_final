FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for numpy/pandas compilation sometimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (we will create this next)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Railway sets PORT env var)
ENV PORT=5000
EXPOSE $PORT

# Run the application
CMD ["python", "generate_names.py"]
