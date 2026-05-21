# Stage 1: Base builder stage
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

# Install system dependencies needed for compiling certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime container
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS runner

WORKDIR /app

# Copy python virtual environment/installed packages from builder
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Copy modular repository code
COPY src/ ./src
COPY scripts/ ./scripts
COPY notebooks/ ./notebooks
COPY tests/ ./tests
COPY README.md .

# Create directory structure for inputs and models
RUN mkdir -p data/raw data/processed models/checkpoints models/lora_adapter

# Expose Jupyter port for interactive runs
EXPOSE 8888

# Default entrypoint runs the test suite to ensure setup is functional
CMD ["pytest"]
