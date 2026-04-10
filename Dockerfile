FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# PySpark needs a Java runtime inside the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy the package metadata first to keep rebuilds faster when source changes.
COPY setup.py ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Default command: run the first pipeline step so the image is easy to test.
CMD ["python", "src/bet/pipelines/create_bronze_dataset.py"]
