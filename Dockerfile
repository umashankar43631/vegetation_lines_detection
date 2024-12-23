# Specify which base layers (default dependencies) to use
# You may find more base layers at https://hub.docker.com/
FROM python:3.10.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app/src/
#
# Creates directory within your Docker image
RUN mkdir -p /app/src/
#
# Copies file from your Local system TO path in Docker image
COPY . /app/src/

# Install system dependencies, including libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#
# Installs dependencies within you Docker image
# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app/src && \
chmod -R 777 /app/src

RUN pip install -e ./segment-anything-2 --no-cache-dir --root-user-action=ignore

RUN pip install -r requirements.txt

CMD ["python", "lines_identification.py"]