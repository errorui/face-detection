FROM python:3.10-slim

# Install build tools and required libs
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && apt-get clean

# Upgrade CMake to a compatible version
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh -o cmake.sh && \
    mkdir /opt/cmake && \
    sh cmake.sh --skip-license --prefix=/opt/cmake && \
    ln -sf /opt/cmake/bin/* /usr/local/bin/

# Set environment so CMake picks up the new version
ENV PATH="/opt/cmake/bin:${PATH}"

# Install Python dependencies
WORKDIR /app
COPY . /app

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install requirements (dlib will now build)
RUN pip install -r requirements.txt
