# Build stage
FROM debian:bookworm AS builder

RUN apt-get update && \
    apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY main.cpp .
COPY httplib.h .

RUN g++ -std=c++17 -Wall -Wextra -O2 main.cpp -o vectordb -pthread


# Runtime stage
FROM debian:bookworm

RUN apt-get update && \
    apt-get install -y libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/vectordb .
COPY index.html .

RUN mkdir -p /app/documents /app/data

EXPOSE 8080

CMD ["./vectordb"]