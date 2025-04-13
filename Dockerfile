FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies for dlib (used by face_recognition)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*  # Clean up apt cache to reduce image size

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

