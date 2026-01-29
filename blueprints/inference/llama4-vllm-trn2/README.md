# Llama 4 with vLLM on Amazon EKS using Trainium2

This blueprint deploys Llama 4 models (Scout and Maverick) using vLLM with NxD Inference on AWS Trainium2 instances.

## Overview

Llama 4 models use a Mixture of Experts (MoE) architecture that requires significant compute resources. This blueprint leverages AWS Trainium2 (trn2) instances with the Neuron SDK for cost-effective inference.

## Model Support

| Model | Parameters | Experts | Instance Required | tensor_parallel_size |
|-------|------------|---------|-------------------|---------------------|
| Llama 4 Scout | 17B active / ~109B total | 16 | trn2.48xlarge | 64 |
| Llama 4 Maverick | 17B active / ~400B total | 128 | trn2.48xlarge | 64 |

## Prerequisites

1. EKS cluster with Trainium2 node support
2. Neuron device plugin installed
3. Hugging Face account with Llama 4 model access
4. Neuron SDK 2.21+ with vLLM-Neuron plugin

## Deployment

### Step 1: Export Hugging Face Token

```bash
export HUGGING_FACE_HUB_TOKEN=$(echo -n "your-hf-token" | base64)
```

### Step 2: Deploy Llama 4 Scout

```bash
envsubst < llama4-vllm-trn2-deployment.yaml | kubectl apply -f -
```

### Step 3: Deploy Open WebUI (Optional)

```bash
kubectl apply -f open-webui.yaml
kubectl -n open-webui port-forward svc/open-webui 8080:80
```

### Step 4: Test with curl

```bash
kubectl -n llama4-vllm port-forward svc/llama4-vllm-trn2-svc 8000:8000

# Text completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Multimodal (image + text)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://httpbin.org/image/png"}},
        {"type": "text", "text": "Describe this image"}
      ]
    }]
  }'
```

## Deploying Maverick Model

For the larger Maverick model with 128 experts:

```bash
envsubst < llama4-vllm-trn2-maverick.yaml | kubectl apply -f -
```

## Key Configuration

### Neuron Configuration

The deployment includes optimized Neuron configuration for Llama 4:

- `tensor_parallel_size=64`: Distributes model across all 64 Neuron cores
- `context_encoding_buckets`: Optimized bucket sizes for variable input lengths
- `async_mode=true`: Enables asynchronous execution for better throughput
- `cp_degree=16`: Context parallelism for efficient attention computation

### Resource Requirements

| Resource | Scout | Maverick |
|----------|-------|----------|
| Neuron Devices | 32 | 32 |
| CPU | 128-192 | 128-192 |
| Memory | 512-768 Gi | 512-768 Gi |
| Model Storage | 1 Ti | 2 Ti |

## Model Compilation (Tracing)

Before deployment, models must be compiled (traced) for Neuron. This is typically done once and the artifacts are stored:

```python
# Example tracing script (run on trn2 instance)
import os
os.environ['NEURON_COMPILED_ARTIFACTS'] = "/models/traced_models/Llama-4-Scout-17B-16E-Instruct"

# Tracing happens automatically on first run with vLLM-Neuron
```

## Cleanup

```bash
kubectl delete -f open-webui.yaml
kubectl delete -f llama4-vllm-trn2-deployment.yaml
# Or for Maverick:
kubectl delete -f llama4-vllm-trn2-maverick.yaml
```

## References

- [NxD Inference Llama 4 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/llama4-tutorial.html)
- [vLLM-Neuron Plugin](https://github.com/aws-neuron/vllm-neuron)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
