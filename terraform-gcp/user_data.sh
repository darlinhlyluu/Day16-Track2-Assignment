#!/bin/bash
set -euo pipefail

ENABLE_GPU="${enable_gpu}"

# Base packages (CPU or GPU)
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ca-certificates curl gnupg python3 python3-pip python3-venv

if [[ "$${ENABLE_GPU}" != "true" ]]; then
  echo "CPU mode: skipping NVIDIA toolkit + vLLM container."
  exit 0
fi

# Install Docker
apt-get install -y --no-install-recommends docker.io
systemctl enable docker
systemctl start docker

# Install NVIDIA container toolkit (drivers are provided by the Deep Learning image)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/"$distribution"/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Run vLLM with the Gemma model
docker rm -f vllm >/dev/null 2>&1 || true
docker run -d \
  --name vllm \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN="${hf_token}" \
  vllm/vllm-openai:latest \
  --model "${model_id}" \
  --dtype half \
  --max-model-len 4096
