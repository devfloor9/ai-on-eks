---
title: Llama 4 with vLLM on Trainium
sidebar_position: 7
description: Deploy Llama 4 models using vLLM on AWS Trainium instances with EKS and Karpenter.
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::danger

Use of Llama 4 models is governed by the [Meta Llama License](https://www.llama.com/llama4/license/).
Please visit [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) and accept the license before requesting access.

:::

# Llama 4 Inference with vLLM on AWS Trainium

This guide covers deploying [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) models using [vLLM](https://github.com/vllm-project/vllm) with [optimum-neuron](https://huggingface.co/docs/optimum-neuron/index) on AWS Trainium instances.

:::warning[Model Compilation Required]

Llama 4 inference on Neuron is supported via **optimum-neuron >= 0.4.0** with the `Llama4NeuronModelForCausalLM` class. However, the first deployment requires **Neuron model compilation**, which happens automatically when `vllm serve` runs but can take **30-60+ minutes**. Pre-compiled artifacts may not yet be available in the [optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache) for all configurations.

The `optimum-cli export neuron` command does **not** support Llama 4. Use `vllm serve` directly, which invokes the inference-path compilation internally.

:::

## Why Trainium for Llama 4?

AWS Trainium provides large HBM memory capacity, making it an excellent choice for large MoE models like Llama 4:

| Instance | Chips | NeuronCores | HBM Memory | Karpenter | EKS Auto Mode |
|----------|-------|-------------|------------|-----------|---------------|
| trn1.32xlarge | 16 Trainium v1 | 32 | 512 GiB | Supported | Supported |
| trn2.48xlarge | 16 Trainium v2 | 64 | 1.5 TiB | Supported | Not yet supported |

| Advantage | Detail |
|-----------|--------|
| **No quantization needed** | Both trn1 (512 GiB) and trn2 (1.5 TiB) support Scout (~220 GiB) in native BF16 |
| **Karpenter auto-provisioning** | Neuron NodePool provisions Trainium nodes on-demand when workloads are scheduled |
| **trn2 for Maverick** | trn2.48xlarge (1.5 TiB) supports Maverick (~800 GiB) in BF16 without quantization |

### Memory Comparison: GPU vs Trainium

| Model | BF16 Memory | GPU (FP8 required?) | trn1.32xlarge (512 GiB) | trn2.48xlarge (1.5 TiB) |
|-------|-------------|---------------------|-------------------------|-------------------------|
| Scout 17B-16E | ~220 GiB | p4d.24xlarge (320 GiB) - No | Fits in BF16 | Fits in BF16 |
| Maverick 17B-128E | ~800 GiB | p5.48xlarge (640 GiB) - **Yes, FP8** | Does not fit | Fits in BF16 |

:::info

For Maverick, only `trn2.48xlarge` has sufficient memory (1.5 TiB) for BF16. GPU deployment requires FP8 quantization, and `trn1.32xlarge` (512 GiB) is insufficient.

:::

:::warning

Trainium instance availability varies by region:

```bash
aws ec2 describe-instance-type-offerings --region <REGION> \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=trn*" \
  --query 'InstanceTypeOfferings[].{Type:InstanceType,Zone:Location}' --output table
```

- **trn2.48xlarge**: Limited availability (`us-east-2`). **Not supported by EKS Auto Mode** — use Karpenter with the inference-ready cluster.
- **trn1.32xlarge**: Available in more regions (`us-west-2`, `us-east-1`, `us-east-2`, etc.).

:::

## Model Compilation

The AWS Neuron DLC uses **optimum-neuron** to run vLLM on Trainium. Models must be pre-compiled for Neuron before serving. The DLC checks the [optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache) on Hugging Face for pre-compiled model artifacts matching your configuration (model, batch size, sequence length, tensor parallelism, dtype).

:::info

The `optimum-cli export neuron` command does **not** support `llama4` as a model type. However, `vllm serve` uses a separate inference code path (`optimum.neuron.models.inference.llama4`) that includes full MoE support via `Llama4NeuronModelForCausalLM`. Compilation is triggered automatically on first serve.

:::

## Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Neuron SDK | 2.26.1 | Required |
| optimum-neuron | >= 0.4.0 | Llama 4 inference support added in v0.4.0 |
| vLLM | 0.11.0 | With optimum-neuron Neuron platform plugin |
| neuronx-distributed | 0.15 | MoE module used by Llama 4 inference |
| DLC Image | `763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-vllm-inference-neuronx:0.11.0-optimum0.4.5-neuronx-py310-sdk2.26.1-ubuntu22.04` | Latest available |

<CollapsibleContent header={<h2><span>Deploying the Inference-Ready EKS Cluster</span></h2>}>

This guide assumes you have an existing EKS cluster with Trainium support. We recommend using the [Inference-Ready EKS Cluster](/docs/infra/inference/inference-ready-cluster) which uses Karpenter for node provisioning and includes pre-configured Neuron NodePools.

### Prerequisites

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Helm 3.0+](https://helm.sh/docs/intro/install/)

### Deploy the Cluster

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/inference-ready-cluster
```

The default `terraform/blueprint.tfvars` uses Karpenter (not EKS Auto Mode). The cluster creates Karpenter NodePools including `trn1-neuron` for Trainium workloads.

To add trn2 support, add `trn2-neuron` to the additional EC2NodeClass names:

```hcl
region                              = "us-east-2"  # trn2 is available in us-east-2
karpenter_additional_ec2nodeclassnames = ["trn2-neuron"]
```

:::note

`us-east-2` has only 3 availability zones. Set `availability_zones_count = 3` in your tfvars.

:::

Then deploy:

```bash
./install.sh
```

### Configure kubectl

```bash
aws eks --region <REGION> update-kubeconfig --name inference-cluster
```

### Verify Karpenter Resources

```bash
# Verify NodePools
kubectl get nodepools

# Verify EC2NodeClasses
kubectl get ec2nodeclasses
```

Expected output:

```
NAME            NODECLASS       NODES   READY   AGE
trn1-neuron     trn1-neuron     0       True    3m
trn2-neuron     trn2-neuron     0       True    3m
g5-nvidia       g5-nvidia       0       True    3m
...
```

The `trn1-neuron` and `trn2-neuron` NodePools include the `aws.amazon.com/neuron` taint. Trainium nodes are provisioned automatically when a workload with the matching toleration is scheduled.

### Install Neuron Device Plugin

The Neuron device plugin is **required** for Trainium workloads. Install it using the AWS Neuron Helm chart:

```bash
# Create required namespace
kubectl create namespace neuron-healthcheck-system

# Install Neuron Helm chart (includes device plugin and node-problem-detector)
helm install neuron-helm-chart \
  oci://public.ecr.aws/neuron/neuron-helm-chart \
  --namespace kube-system \
  --version 1.3.0
```

Verify the installation:

```bash
# Check Neuron device plugin DaemonSet (0 desired is expected until Neuron nodes are provisioned)
kubectl get daemonset neuron-device-plugin -n kube-system
```

:::note

The Neuron device plugin DaemonSet will show `0/0` desired/ready pods until a Trainium node is provisioned. This is expected behavior - the DaemonSet pods will automatically start when a Neuron node joins the cluster.

:::

### Neuron Resource Names

When a Trainium node is provisioned, the device plugin exposes the following extended resources:

| Resource | Description | trn1.32xlarge | trn2.48xlarge |
|----------|-------------|---------------|---------------|
| `aws.amazon.com/neuron` | Neuron devices (chips) | 16 | 16 |
| `aws.amazon.com/neuroncore` | NeuronCores (2 per v1 chip, 4 per v2 chip) | 32 | 64 |

Use `aws.amazon.com/neuron` in pod resource requests to allocate Neuron devices.

</CollapsibleContent>

## Deploy Llama 4 Scout on Trainium

### Step 1: Create Hugging Face Token Secret

```bash
kubectl create secret generic hf-token --from-literal=token=<your-huggingface-token>
```

### Step 2: Deploy with Helm

For **trn1.32xlarge** (Scout):

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

helm install llama4-scout-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-scout-17b-vllm-neuron.yaml
```

:::info

Key deployment parameters:
- **tensor_parallel_size: 16** (one per Trainium chip, not per NeuronCore)
- **Docker image**: AWS Neuron DLC from private ECR (`763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-vllm-inference-neuronx`)
- **Neuron device requests**: `aws.amazon.com/neuron: 16` for all 16 chips
- **CPU memory**: `384Gi` minimum (weight sharding requires loading the full model into CPU memory)
- **Instance type**: `trn1.32xlarge` for Scout, `trn2.48xlarge` for Maverick
- **Environment variable**: `VLLM_NEURON_FRAMEWORK=optimum` is required for on-the-fly Neuron compilation

:::

### Step 3: Monitor Deployment

After deploying, Karpenter will automatically provision a Trainium node:

```bash
# Watch node provisioning
kubectl get nodeclaims -w

# Check pod status
kubectl get pods -w
```

During deployment, the pod will go through these stages:
1. **Pending** - waiting for Trainium node provisioning (~5 minutes)
2. **ContainerCreating** - pulling the Neuron DLC image (~2.9 GiB)
3. **Running** - Neuron model compilation (30-60+ minutes on first run)
4. **Ready** - vLLM server is serving requests

:::warning[CPU Memory Requirements]

The pod requires **at least 384 GiB of CPU memory** for model weight sharding across 16 Neuron devices. With insufficient memory (e.g., 64 GiB), the pod will be OOMKilled during weight loading. The trn2.48xlarge instance provides ~2 TiB of system memory, so this is well within capacity.

:::

:::warning

The first deployment takes significantly longer due to Neuron model compilation. Subsequent deployments with the same configuration will use cached artifacts. Monitor the compilation progress in the logs:

```bash
kubectl logs -f -l app.kubernetes.io/instance=llama4-scout-neuron
```

:::

**Tested deployment timeline on trn2.48xlarge (Scout):**

| Phase | Duration | Description |
|-------|----------|-------------|
| Node provisioning | ~5 min | Karpenter provisions trn2.48xlarge |
| Image pull | ~30 sec | DLC image (~2.9 GiB, cached after first pull) |
| HLO generation | ~60 sec | Generates HLOs for context_encoding and token_generation |
| Neuron compilation | ~200 sec | neuronx-cc compiles HLOs to NEFFs (target=trn2) |
| Model build | ~650 sec | Weight layout transformation |
| Weight loading | ~5 min | Download, shard, and load weights to 16 Neuron devices |
| **Total (first deploy)** | **~20 min** | Subsequent deploys reuse cached compilation artifacts |

Once complete, the vLLM server will start:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Deploy Llama 4 Maverick on Trainium2

Maverick requires `trn2.48xlarge` (1.5 TiB HBM) and runs in native BF16 without quantization. Ensure your cluster has the `trn2-neuron` Karpenter NodePool configured (see cluster setup above).

```bash
helm install llama4-maverick-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-maverick-17b-vllm-neuron.yaml
```

:::warning

- `trn2.48xlarge` is available in limited regions (`us-east-2`). Verify availability before deploying.
- Ensure your AWS account has sufficient [service quota](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) for Trainium instances (Maverick requires 192 vCPUs).

:::

## Test the Model

### Port Forward

```bash
kubectl port-forward svc/llama4-scout-neuron 8000:8000
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

Llama 4 supports multimodal inference. Send image URLs alongside text:

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

```bash
helm repo add open-webui https://helm.openwebui.com/
helm repo update

helm install open-webui open-webui/open-webui \
  --namespace open-webui --create-namespace \
  --set ollama.enabled=false \
  --set env.OPENAI_API_BASE_URL=http://llama4-scout-neuron.default.svc.cluster.local:8000/v1 \
  --set env.OPENAI_API_KEY=dummy
```

Access the UI:

```bash
kubectl port-forward svc/open-webui 8080:80 -n open-webui
```

Open [http://localhost:8080](http://localhost:8080) in your browser and register a new account. The model will appear in the model selector.

## Monitoring

### Check Inference Logs

```bash
# View vLLM Neuron logs
kubectl logs -l app.kubernetes.io/instance=llama4-scout-neuron --tail=100

# Monitor token generation throughput
kubectl logs -l app.kubernetes.io/instance=llama4-scout-neuron -f | grep "tokens/s"
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
