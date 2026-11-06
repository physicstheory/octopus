# Octopus on a free Hugging Face Space (or any Docker host).
# The Space needs two secrets: OCTOPUS_PASSWORD (the invite code people use to create accounts)
# and HF_TOKEN (a Hugging Face "Write" token, used to save your data and for 3D and image GPU time).
# Build pulls the latest Octopus from GitHub; use "Factory rebuild" on the Space to update.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# Ollama runs the chat models.
COPY --from=ollama/ollama:latest /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama/ollama:latest /usr/lib/ollama /usr/lib/ollama

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    OLLAMA_MODELS=/home/user/ollama OCTOPUS_HOME=/home/user/workspace \
    OCTOPUS_HOST=0.0.0.0 OCTOPUS_PORT=7860 OLLAMA_KEEP_ALIVE=30m

ARG OCTOPUS_REPO=https://github.com/physicstheory/octopus
ARG OCTOPUS_REF=main
RUN git clone --depth 1 -b $OCTOPUS_REF $OCTOPUS_REPO /home/user/app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r requirements.txt "huggingface_hub>=0.34"

# Build the everyday chat models into the image, so the Space answers straight away after a restart.
ARG CHAT_MODELS="qwen2.5-coder:1.5b qwen3:4b-instruct-2507-q4_K_M"
RUN ollama serve >/tmp/ollama.log 2>&1 & \
    for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done; \
    for m in $CHAT_MODELS; do ollama pull $m || exit 1; done

EXPOSE 7860
CMD ["bash", "cloud/start.sh"]
