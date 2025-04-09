FROM python:3.10-slim

# Install system dependencies needed for dlib + face-recognition
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    curl \
    && apt-get clean

# Install Python packaging tools
RUN pip install --upgrade pip wheel setuptools

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --prefer-binary -r requirements.txt

# Copy app code
COPY . .

# Expose port and run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
