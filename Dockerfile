FROM debian:bullseye-slim AS builder

# Set up build environment
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /app

# Copy source code
COPY . .

# Build the project
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc) && \
    cmake --install . --prefix=/app/install

# Create a smaller runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy built executables and libraries from builder stage
COPY --from=builder /app/install/bin/sentiment_analyzer /app/
COPY --from=builder /app/install/bin/api_example /app/
COPY --from=builder /app/install/lib/libsentiment.a /app/lib/
COPY --from=builder /app/data /app/data

# Set environment variables
ENV LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

# Create volume for user data
VOLUME /app/data

# Expose port if we add API server capabilities in the future
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["/app/sentiment_analyzer"]
CMD ["--interactive"]
