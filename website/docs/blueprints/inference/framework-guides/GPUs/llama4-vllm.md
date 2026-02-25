---
title: Llama 4 with vLLM on EKS
sidebar_position: 10
description: Deploy Llama 4 Scout and Maverick models using vLLM on Amazon EKS with GPU acceleration.
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::danger

Use of Llama 4 models is governed by the [Meta Llama License](https://www.llama.com/llama4/license/).
Please visit [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and accept the license before requesting access.

:::

# Llama 4 Inference with vLLM on Amazon EKS

In this guide, we'll explore deploying [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) models using [vLLM](https://github.com/vllm-project/vllm) inference engine on [Amazon EKS](https://aws.amazon.com/eks/).

Llama 4 introduces a **Mixture of Experts (MoE)** architecture, where only a subset of parameters are active per token, enabling efficient inference for its large total parameter count. Two model variants are available:

- **Llama 4 Scout** (17B active / 109B total, 16 experts) - Mid-size multimodal model
- **Llama 4 Maverick** (17B active / 400B total, 128 experts) - Large multimodal model

Both models support multimodal inputs (text and images) and provide OpenAI-compatible API endpoints via vLLM.

## Understanding GPU Memory Requirements

Deploying MoE models requires loading all expert weights into GPU memory, even though only a subset is active per token. This means total parameter count determines memory requirements, not active parameters.

| Model | Total Params | Experts | BF16 Memory | FP8 Memory | Recommended GPU Instance |
|-------|-------------|---------|-------------|------------|--------------------------|
| Scout 17B-16E | ~109B | 16 | ~220 GiB | ~110 GiB | p4d.24xlarge (8x A100 40GB = 320 GiB) |
| Maverick 17B-128E | ~400B | 128 | ~800 GiB | ~400 GiB | p5.48xlarge (8x H100 80GB = 640 GiB, **FP8 required**) |

:::warning

**Maverick on GPU requires FP8 quantization.** The BF16 model weights (~800 GiB) exceed the p5.48xlarge total GPU memory (640 GiB). Use the FP8-quantized variant (`meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`) which fits within 640 GiB.

For running Maverick without quantization, consider [Trainium2 deployment](/docs/blueprints/inference/framework-guides/Neuron/llama4-trn2) which provides 1.5 TiB HBM memory.

:::

<CollapsibleContent header={<h2><span>Deploying the Inference-Ready EKS Cluster</span></h2>}>

This guide assumes you have an existing EKS cluster with GPU support. We recommend using the [Inference-Ready EKS Cluster](/docs/infra/inference/inference-ready-cluster) which comes pre-configured with EKS Auto Mode and GPU NodePool.

### Prerequisites

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Helm 3.0+](https://helm.sh/docs/intro/install/)

### Deploy the Cluster

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/inference-ready-cluster
```

Ensure `enable_eks_auto_mode = true` in `terraform/blueprint.tfvars`, then run:

```bash
./install.sh
```

### Configure kubectl

```bash
aws eks --region <REGION> update-kubeconfig --name inference-cluster
```

### Verify EKS Auto Mode Resources

```bash
# Verify NodePools (gpu, neuron, general-purpose, system)
kubectl get nodepools

# Verify nodes (GPU nodes are provisioned on-demand when workloads are scheduled)
kubectl get nodes
```

</CollapsibleContent>

## Deploy Llama 4 Scout (17B-16E)

### Step 1: Create Hugging Face Token Secret

Create a [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens) and store it as a Kubernetes secret:

```bash
kubectl create secret generic hf-token --from-literal=token=<your-huggingface-token>
```

### Step 2: Deploy with Helm

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

helm install llama4-scout ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-scout-17b-lws-vllm.yaml
```

:::info

The Scout model uses **LeaderWorkerSet (LWS)** for multi-node tensor parallelism across 8 GPUs. Model download and initialization may take several minutes on first deployment.

:::

### Step 3: Verify Deployment

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/component=llama-4-scout

# Watch logs for model loading progress
kubectl logs -l app.kubernetes.io/component=llama-4-scout -f
```

Wait until you see the vLLM server ready message in the logs:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Deploy Llama 4 Maverick (17B-128E) on GPU

Maverick requires FP8 quantization on GPU due to its large model size (~800 GiB BF16).

```bash
helm install llama4-maverick ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-maverick-17b-lws-vllm.yaml
```

:::warning

Maverick deployment requires **p5.48xlarge** instances (8x H100 80GB). Ensure your AWS account has sufficient [service quota](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) for P5 instances in your region.

:::

## Test the Model

### Port Forward

```bash
kubectl port-forward svc/llama-4-scout 8000:8000
```

### Chat Completion Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {"role": "user", "content": "What are the key differences between Mixture of Experts and dense transformer models?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

### Multimodal Request (Text + Image)

Llama 4 supports vision inputs. You can send image URLs alongside text:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image?"},
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

## Deploy Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-style interface for interacting with the model.

The inference-ready cluster includes Open WebUI. To access it:

```bash
kubectl port-forward svc/open-webui 8080:80 -n open-webui
```

Open [http://localhost:8080](http://localhost:8080) in your browser and register a new account. The model will appear in the model selector.

## Monitoring

### Check Inference Logs

```bash
# View vLLM logs for throughput and latency metrics
kubectl logs -l app.kubernetes.io/component=llama-4-scout --tail=100

# Watch for token generation throughput
kubectl logs -l app.kubernetes.io/component=llama-4-scout -f | grep "tokens/s"
```

### GPU Utilization

If the observability stack is enabled on your cluster, access Grafana for GPU metrics:

```bash
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
```

## Cleanup

Remove the model deployment:

```bash
# Remove Scout
helm uninstall llama4-scout

# Remove Maverick (if deployed)
helm uninstall llama4-maverick
```

To destroy the entire cluster infrastructure:

```bash
cd ai-on-eks/infra/solutions/inference-ready-cluster
./cleanup.sh
```
