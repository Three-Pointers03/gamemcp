FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional: for pillow, etc.)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose default port; DigitalOcean will set PORT env
EXPOSE 8087

# Runtime env vars expected:
# - AUTH_TOKEN
# - MY_NUMBER
# - PORT (provided by platform)

CMD ["python", "mcp-bearer-token/social_gaming_mcp.py"]


