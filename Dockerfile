# Stage 1: Build the C++ binary
FROM gcc:bookworm AS builder
WORKDIR /build

# Copy source code and header dependencies
COPY main.cpp httplib.h ./

# Compile with C++17 optimizations and pthread support
RUN g++ -std=c++17 -Wall -Wextra -pedantic -O2 main.cpp -o vectordb -pthread

# Stage 2: Minimal runtime container
FROM debian:bookworm-slim
WORKDIR /app

# Install standard runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled executable and frontend
COPY --from=builder /build/vectordb /app/vectordb
COPY index.html /app/index.html

# Create directories for data and documents
RUN mkdir -p /app/documents /data

# Default environment configuration
ENV VECTORDB_HOST=0.0.0.0
ENV PORT=8080
ENV VECTORDB_DATA_PATH=/data/vectors.jsonl
ENV VECTORDB_WATCH_DIR=/data/documents

EXPOSE 8080

CMD ["./vectordb"]
