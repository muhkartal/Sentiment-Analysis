# Docker Support for C++ Sentiment Analysis Engine

This document provides comprehensive instructions for using Docker with the C++ Sentiment Analysis Engine.

## Table of Contents

-  [Overview](#overview)
-  [Prerequisites](#prerequisites)
-  [Docker Image](#docker-image)
-  [Running with Docker](#running-with-docker)
-  [Docker Compose](#docker-compose)
-  [Environment Variables](#environment-variables)
-  [Custom Data](#custom-data)
-  [Building Custom Images](#building-custom-images)
-  [Troubleshooting](#troubleshooting)

## Overview

The C++ Sentiment Analysis Engine provides Docker support for easy deployment and consistent runtime environments. The Docker implementation uses multi-stage builds to create efficient images while maintaining all functionality.

## Prerequisites

-  Docker Engine 19.03+ installed
-  Docker Compose 1.27+ (optional, for docker-compose.yml usage)
-  At least 2GB of free disk space

## Docker Image

The project's Docker image is based on Debian Bullseye and includes:

-  The compiled sentiment analysis engine
-  The API example executable
-  Sample data included in the project
-  All necessary runtime dependencies

The image is built using a multi-stage approach to minimize size:

1. A build stage that compiles the code
2. A runtime stage that contains only the necessary components

## Running with Docker

### Basic Usage

```bash
# Build the Docker image
docker build -t sentiment-analysis .

# Run in interactive mode
docker run -it sentiment-analysis

# Run with your own data directory
docker run -it -v $(pwd)/your-data:/app/data sentiment-analysis

# Run API example
docker run -it sentiment-analysis /app/api_example
```

### Command-Line Options

You can pass all the normal command-line options to the container:

```bash
# Run with a specific data file
docker run -it sentiment-analysis --file /app/data/my-dataset.csv

# Run without interactive mode
docker run -it sentiment-analysis --file /app/data/sample_data.csv
```

## Docker Compose

For more complex setups, we provide a `docker-compose.yml` file:

```bash
# Start the sentiment analyzer
docker-compose up

# Run in detached mode
docker-compose up -d

# Run only the API example
docker-compose run api-example

# Stop all containers
docker-compose down
```

## Environment Variables

The Docker image supports the following environment variables:

| Variable               | Description                                     | Default     |
| ---------------------- | ----------------------------------------------- | ----------- |
| `SENTIMENT_LOG_LEVEL`  | Logging verbosity (debug, info, warning, error) | info        |
| `SENTIMENT_MODEL_PATH` | Path to load/save models                        | /app/models |
| `SENTIMENT_DATA_PATH`  | Path to data directory                          | /app/data   |

Example:

```bash
docker run -it -e SENTIMENT_LOG_LEVEL=debug sentiment-analysis
```

## Custom Data

You can mount your own data directory to the container:

```bash
# Mount a local directory
docker run -it -v $(pwd)/your-data:/app/data sentiment-analysis

# Mount a single file
docker run -it -v $(pwd)/your-dataset.csv:/app/data/custom.csv sentiment-analysis --file /app/data/custom.csv
```

## Building Custom Images

You can extend the base image for custom scenarios:

```dockerfile
FROM sentiment-analysis:latest

# Add your custom data
COPY your-data/ /app/data/

# Install additional tools
RUN apt-get update && apt-get install -y your-package

# Set custom environment variables
ENV SENTIMENT_LOG_LEVEL=debug

# Override the default command
CMD ["--file", "/app/data/your-custom-file.csv"]
```

## Troubleshooting

### Common Issues

1. **Permission errors when mounting volumes**

   If you encounter permission issues when mounting volumes, make sure the user inside the container has permission to read/write the mounted directory:

   ```bash
   # Fix permissions before mounting
   chmod -R 777 ./your-data
   ```

2. **Container exits immediately**

   The default mode requires a TTY for interactive usage. Make sure to include the `-it` flags:

   ```bash
   docker run -it sentiment-analysis
   ```

3. **Out of memory during build**

   If Docker runs out of memory during the build process, increase the available memory in Docker settings.

### Getting Help

If you encounter problems not covered here, please:

1. Check the logs with `docker logs <container-id>`
2. Run the container in debug mode: `docker run -it -e SENTIMENT_LOG_LEVEL=debug sentiment-analysis`
3. Submit an issue on GitHub with the complete logs and environment details
