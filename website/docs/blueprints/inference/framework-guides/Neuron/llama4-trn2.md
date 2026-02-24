---
title: Llama 4 with vLLM on Trainium2
sidebar_position: 7
description: Deploy Llama 4 models using vLLM with NxD Inference on AWS Trainium2 instances.
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::danger

Use of Llama 4 models is governed by the [Meta Llama License](https://www.llama.com/llama4/license/).
Please visit [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and accept the license before requesting access.

:::

# Llama 4 Inference with vLLM on AWS Trainium2

This guide demonstrates deploying [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) models using [vLLM](https://github.com/vllm-project/vllm) with [NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html) on AWS Trainium2 instances.

## Why Trainium2 for Llama 4?

AWS Trainium2 (`trn2.48xlarge`) provides **1.5 TiB of HBM memory** across 64 Neuron cores, making it an excellent choice for large MoE models like Llama 4:

| Advantage | Detail |
|-----------|--------|
| **No quantization needed** | 1.5 TiB HBM supports both Scout (~220 GiB) and Maverick (~800 GiB) in native BF16 |
| **Cost efficient** | Purpose-built ML silicon at competitive price-performance |
| **EKS Auto Mode support** | Trn family instances are natively supported - no separate NodePool required |
| **SOCI parallel pull** | Automatically enabled for Trn instances in EKS Auto Mode for faster image pulls |

### Memory Comparison: GPU vs Trainium2

| Model | BF16 Memory | GPU (FP8 required?) | Trainium2 (BF16 direct) |
|-------|-------------|---------------------|-------------------------|
| Scout 17B-16E | ~220 GiB | p4d.24xlarge (320 GiB) - No | trn2.48xlarge (1.5 TiB) - No |
| Maverick 17B-128E | ~800 GiB | p5.48xlarge (640 GiB) - **Yes, FP8 required** | trn2.48xlarge (1.5 TiB) - No |

:::info

Trainium2 can run the full Maverick model in BF16 precision without any quantization, while GPU deployment requires FP8 to fit within memory constraints. This means higher model quality on Trainium2 for Maverick.

:::

## Model Compilation (Required)

Deploying on Trainium2 requires a one-time model compilation (tracing) step using NxD Inference. This converts the model into a Neuron-optimized format.

### Compile the Model

Use an EC2 Trainium2 instance or a compilation job to trace the model:

```bash
# Example: Compile Llama 4 Scout for 64 Neuron cores
python3 -m nxd_inference.trace \
  --model-id meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 64 \
  --output-dir ./compiled-llama4-scout
```

### Upload to S3

Store the compiled artifacts in S3 for use during deployment:

```bash
aws s3 cp --recursive ./compiled-llama4-scout s3://<your-bucket>/llama4-scout-compiled/
```

:::warning

Model compilation can take significant time depending on model size and instance type. Compile once and reuse the artifacts across deployments via S3.

:::

<CollapsibleContent header={<h2><span>Deploying the Inference-Ready EKS Cluster</span></h2>}>

This guide assumes you have an existing EKS cluster with Trainium support. We recommend using the [Inference-Ready EKS Cluster](/docs/infra/inference/inference-ready-cluster) which comes pre-configured with all necessary components.

### Prerequisites

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Helm 3.0+](https://helm.sh/docs/intro/install/)

### Deploy the Cluster

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/inference-ready-cluster
```

Update the region in `terraform/blueprint.tfvars`, then run:

```bash
./install.sh
```

### Configure kubectl

```bash
aws eks --region <REGION> update-kubeconfig --name inference-cluster
```

### Verify Resources

```bash
kubectl get nodes
```

</CollapsibleContent>

## Deploy Llama 4 Scout on Trainium2

### Step 1: Create Hugging Face Token Secret

```bash
kubectl create secret generic hf-token --from-literal=token=<your-huggingface-token>
```

### Step 2: Deploy with Helm

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

helm install llama4-scout-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-scout-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://<your-bucket>/llama4-scout-compiled/"
```

### Step 3: Verify Deployment

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/component=llama-4-scout-neuron

# Watch logs for model loading progress
kubectl logs -l app.kubernetes.io/component=llama-4-scout-neuron -f
```

:::info

Neuron model loading includes loading pre-compiled artifacts from S3. This may take several minutes on first deployment, especially for large models. SOCI parallel pull in EKS Auto Mode helps accelerate container image pulls.

:::

## Deploy Llama 4 Maverick on Trainium2

Maverick runs in native BF16 on Trainium2 without quantization:

```bash
helm install llama4-maverick-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-maverick-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://<your-bucket>/llama4-maverick-compiled/"
```

:::warning

Ensure your AWS account has sufficient [service quota](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) for `trn2.48xlarge` instances in your region.

:::

## Test the Model

### Port Forward

```bash
kubectl port-forward svc/llama-4-scout-neuron 8000:8000
```

### Chat Completion Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {"role": "user", "content": "Explain the benefits of Mixture of Experts architecture in large language models."}
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

Llama 4 on Trainium2 supports multimodal inference. Send image URLs alongside text:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe what you see in this image."},
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
# View vLLM Neuron logs
kubectl logs -l app.kubernetes.io/component=llama-4-scout-neuron --tail=100

# Monitor token generation throughput
kubectl logs -l app.kubernetes.io/component=llama-4-scout-neuron -f | grep "tokens/s"
```

### Observability Dashboard

If the observability stack is enabled on your cluster, access Grafana:

```bash
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
```

## Cleanup

Remove the model deployment:

```bash
# Remove Scout
helm uninstall llama4-scout-neuron

# Remove Maverick (if deployed)
helm uninstall llama4-maverick-neuron
```

To destroy the entire cluster infrastructure:

```bash
cd ai-on-eks/infra/solutions/inference-ready-cluster
./cleanup.sh
```
