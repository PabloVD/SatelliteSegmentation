# Use an official Python image as base
FROM python:3.13-slim

# Set environment variables to avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and GDAL
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    ffmpeg libsm6 libxext6

# Install the project into `/app`
WORKDIR /app

COPY . /app

# Install project
RUN pip install .

# Install GDAL python dependency, which depends on the numpy version
RUN pip install --no-cache-dir --force-reinstall 'GDAL[numpy]==3.6.2'
