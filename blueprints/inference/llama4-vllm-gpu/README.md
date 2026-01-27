# Llama 4 vLLM Inference Blueprint

This blueprint deploys Llama 4 models using vLLM inference engine on Amazon EKS with EKS Auto Mode for automatic GPU node provisioning.

## Features

- **Direct vLLM Deployment**: No Ray dependency, simple Kubernetes Deployment
- **EKS Auto Mode**: Automatic GPU node provisioning when workloads are scheduled
- **OpenAI-Compatible API**: `/v1/chat/completions` and `/v1/models` endpoints
- **Multiple Model Variants**: Support for 8B (single GPU) and 70B (multi-GPU) models
- **Open WebUI Integration**: ChatGPT-style web interface

## Files

| File | Description |
|------|-------------|
| `llama4-vllm-deployment.yml` | Llama 4 8B model deployment (single GPU) |
| `llama4-vllm-deployment-70b.yml` | Llama 4 70B model deployment (8 GPUs) |
| `open-webui.yaml` | Open WebUI deployment for chat interface |

## Prerequisites

- EKS Cluster >= 1.30 with Auto Mode enabled
- HuggingFace account with access to Llama 4 models
- `kubectl` and `envsubst` installed

## Quick Start

1. Export your HuggingFace token:
```bash
export HUGGING_FACE_HUB_TOKEN=$(echo -n "your-token" | base64)
```

2. Deploy the model:
```bash
envsubst < llama4-vllm-deployment.yml | kubectl apply -f -
```

3. Test the deployment:
```bash
kubectl port-forward -n llama4-vllm svc/llama4-vllm-svc 8000:8000
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-4-8B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Documentation

For detailed deployment instructions, see the [Llama 4 vLLM Blueprint Guide](../../../website/docs/blueprints/inference/GPUs/llama4-vllm.md).

## GPU Memory Requirements

| Model | Parameters | BF16 Memory | Recommended GPU | tensor_parallel_size |
|-------|------------|-------------|-----------------|---------------------|
| Llama 4 8B | 8B | ~16 GiB | A10G (24GB), L4 (24GB) | 1 |
| Llama 4 70B | 70B | ~140 GiB | 8x A10G or 2x A100 (80GB) | 8 or 2 |
