FROM python:3.12-slim
# biotite 0.38.0 publishes no cp312 wheel on any manylinux tag, so pip builds it
# from source. build-essential (gcc + g++ + make + libc6-dev) supplies the
# toolchain that build needs. Every other pin in this set installs from a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Always keep basic python packaging tools fresh to avoid CVEs even in old python versions
RUN pip install --upgrade pip wheel setuptools
# Install dependencies to leverage Docker cache
COPY requirements-antifold.txt .
RUN pip install --no-cache-dir -r requirements-antifold.txt
# Copy package source
COPY . /app
# Set Python path
ENV PYTHONPATH=/app
# Default command (will be overridden by entrypoint)
CMD ["bash"]
