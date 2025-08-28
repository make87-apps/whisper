# ---------- Build ----------
FROM rust:1.86-slim-bookworm AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl pkg-config build-essential g++ cmake clang libclang-dev git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# (optional) silence rustfmt warning
# RUN rustup component add rustfmt

COPY . .
RUN cargo build --release

# fetch Whisper weights during build (bake into image)
RUN mkdir -p /models && \
    curl -L -o /models/ggml-tiny.en.bin \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin?download=true"

# ---------- Runtime ----------
FROM debian:bookworm-slim
WORKDIR /app
# runtime needs only C++ stdlib (add libgomp1 if you enable OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/target/release/app /usr/local/bin/app
COPY --from=build /models /models

ENV WHISPER_MODEL=/models/ggml-tiny.en.bin
CMD ["/usr/local/bin/app"]
