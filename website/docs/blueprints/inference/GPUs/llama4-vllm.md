---
title: Llama 4 with vLLM on EKS
sidebar_position: 6
---
import CollapsibleContent from '../../../../src/components/CollapsibleContent';

# Llama 4 with vLLM on Amazon EKS

In this guide, we'll explore deploying [Llama 4](https://huggingface.co/meta-llama) models using [vLLM](https://github.com/vllm-project/vllm) inference engine on [Amazon EKS](https://aws.amazon.com/eks/) with EKS Auto Mode for automatic GPU node provisioning.

:::info
This blueprint uses **EKS Auto Mode** for automatic GPU node provisioning. When you deploy a GPU workload, EKS automatically provisions the appropriate GPU nodes without requiring Karpenter or manual node group configuration.
:::

## Understanding the GPU Memory Requirements

Deploying Llama 4 models requires careful memory planning. Llama 4 uses a Mixture of Experts (MoE) architecture where all expert weights must be loaded into GPU memory, even though only a subset of experts are activated per token.

### Model Memory Requirements

| Model | Active Params | Total Params | BF16 Memory | FP8 Memory | Min GPU Config | tensor_parallel_size |
|-------|---------------|--------------|-------------|------------|----------------|---------------------|
| Llama 4 Scout (17B-16E) | 17B | ~109B | ~220 GiB | ~110 GiB | 8x A100 (40GB) | 8 |
| Llama 4 Maverick (17B-128E) | 17B | ~400B | ~800 GiB | ~400 GiB | 8x A100 (80GB) | 8 |

:::warning
Llama 4 Maverick in BF16 requires ~800GB GPU memory, which exceeds the capacity of any single-node GPU instance (max 640GB on p4de/p5). This blueprint uses the **FP8 quantized version** (`RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8`) which reduces memory requirements to ~400GB.
:::

### EC2 Instance Selection Guide

:::danger
Llama 4 Scout requires **at least 220GB of GPU memory**. Common GPU instances like g5.48xlarge (8x A10G = 192GB) will fail with CUDA Out of Memory errors.
:::

| Instance Type | GPU | GPU Memory | Total VRAM | Scout (220GB) | Maverick FP8 (400GB) | Maverick BF16 (800GB) | Cost/hr |
|--------------|-----|------------|------------|---------------|----------------------|-----------------------|---------|
| g5.48xlarge | 8x A10G | 24GB each | 192GB | ❌ Insufficient | ❌ Insufficient | ❌ Insufficient | ~$16 |
| p4d.24xlarge | 8x A100 | 40GB each | 320GB | ✅ Supported | ❌ Insufficient | ❌ Insufficient | ~$32 |
| p4de.24xlarge | 8x A100 | 80GB each | 640GB | ✅ Recommended | ✅ Supported | ❌ Insufficient | ~$40 |
| p5.48xlarge | 8x H100 | 80GB each | 640GB | ✅ Recommended | ✅ Supported | ❌ Insufficient | ~$98 |
| p5e.48xlarge | 8x H200 | 141GB each | 1,128GB | ✅ Recommended | ✅ Supported | ✅ Supported | ~$120+ |

:::info
p5e.48xlarge with 8x H200 GPUs (1.1TB total VRAM) is the only single-node instance capable of running Maverick in BF16 without quantization. However, availability is limited to specific regions.
:::

### Why MoE Models Need More Memory

Unlike dense models where memory ≈ 2 × parameters (for BF16), MoE models load **all expert weights** into memory:

```
Scout Memory = Base Model + (16 experts × expert_size) ≈ 220GB
Maverick Memory = Base Model + (128 experts × expert_size) ≈ 800GB
```

Even though only 1-2 experts are activated per token during inference, all experts must reside in GPU memory for fast routing.

### Memory Optimization Options

If you need to run on smaller GPUs, consider these alternatives:

1. **AWQ Quantization** (4-bit): Reduces memory by ~4x, but may have compatibility issues with vLLM for MoE models
2. **Smaller Models**: Use Llama 3.1 8B/70B which have more predictable memory requirements
3. **Pipeline Parallelism**: Split model across multiple nodes (requires LeaderWorkerSet)

Using vLLM with `gpu-memory-utilization=0.9`, we optimize memory usage while preventing out-of-memory (OOM) crashes.


<CollapsibleContent header={<h2><span>Prerequisites and EKS Cluster Setup</span></h2>}>

### Prerequisites

Before deploying Llama 4, ensure you have the following tools installed:

:::info
To simplify the demo process, we assume the use of an IAM role with administrative privileges. For production deployments, create an IAM role with only the necessary permissions using tools like [IAM Access Analyzer](https://aws.amazon.com/iam/access-analyzer/).
:::

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [eksctl](https://eksctl.io/installation/) (optional, for cluster creation)
4. [envsubst](https://pypi.org/project/envsubst/)

### EKS Cluster Requirements

This blueprint requires an EKS cluster with the following configuration:

- **EKS Version**: >= 1.30 (required for EKS Auto Mode support)
- **EKS Auto Mode**: Enabled (for automatic GPU node provisioning)
- **NVIDIA Device Plugin**: Automatically managed by EKS Auto Mode

### Option A: Create a New EKS Auto Mode Cluster

If you don't have an existing EKS cluster, you can quickly create one using `eksctl`:

```bash
eksctl create cluster \
  --name llama4-cluster \
  --region $AWS_REGION \
  --enabme-auto-mode
```

:::info
Cluster creation takes approximately 10-15 minutes. The `--enable-auto-mode` flag enables EKS Auto Mode, which automatically manages node provisioning.
:::

After the cluster is created, proceed to **Step 3** below to create a GPU NodePool.

### Option B: Use an Existing EKS Cluster

If you already have an EKS cluster, verify that EKS Auto Mode is enabled:

**Step 1:** Check EKS cluster version

```bash
kubectl version
```

```text
Client Version: v1.30.0
Kustomize Version: v5.x.x
Server Version: v1.30.0-eks-xxxxx
```

**Step 2:** Verify EKS Auto Mode is enabled

```bash
aws eks describe-cluster --name <cluster-name> --query 'cluster.computeConfig.enabled'
```

```text
true
```

### Step 3: Configure GPU NodePool

EKS Auto Mode requires a GPU NodePool to provision GPU instances. Check if a GPU NodePool exists:

```bash
kubectl get nodepools
```

```text
NAME              AGE
general-purpose   10m
gpu               5m
system            10m
```

If you don't see a `gpu` NodePool, create one:

```bash
cat <<EOF | kubectl apply -f -
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 30s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    spec:
      expireAfter: 336h
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: default
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: eks.amazonaws.com/instance-category
          operator: In
          values: ["g", "p"]
        - key: eks.amazonaws.com/instance-generation
          operator: Gte
          values: ["4"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule
EOF
```

:::info
The default `general-purpose` NodePool only supports CPU instance categories (c, m, r). GPU instances (g, p categories) require a dedicated GPU NodePool. The `instance-generation: Gte 4` ensures p4d/p4de/p5 instances are available for Llama 4 models.
:::

**Step 4:** Verify NVIDIA Device Plugin (Auto Mode manages this automatically)

```bash
kubectl get daemonset -n kube-system | grep nvidia
```

If EKS Auto Mode is enabled, the NVIDIA device plugin is automatically deployed when GPU workloads are scheduled.

</CollapsibleContent>

## Deploying Llama 4 Scout with vLLM

With the EKS cluster ready, we can now deploy Llama 4 Scout using vLLM.

:::caution
The use of [Llama 4](https://huggingface.co/meta-llama) models requires access through a Hugging Face account. Make sure you have accepted the model license on HuggingFace.
:::

**Step 1:** Export the Hugging Face Hub Token

Create a Hugging Face account and generate an access token:
1. Navigate to [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with read permissions
3. Export the token as a base64-encoded environment variable:

```bash
export HUGGING_FACE_HUB_TOKEN=$(echo -n "Your-Hugging-Face-Hub-Token-Value" | base64)
```

**Step 2:** Clone the repository

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks
```

**Step 3:** Deploy the vLLM service

```bash
cd blueprints/inference/llama4-vllm-gpu/
envsubst < llama4-vllm-deployment.yml | kubectl apply -f -
```

**Output:**

```text
namespace/llama4-vllm created
secret/hf-token created
deployment.apps/llama4-vllm created
service/llama4-vllm-svc created
```

**Step 4:** Monitor the deployment

```bash
kubectl get pods -n llama4-vllm -w
```

:::info
The first deployment may take 10-15 minutes as EKS Auto Mode provisions a GPU node and the model weights are downloaded from HuggingFace.
:::

```text
NAME                           READY   STATUS    RESTARTS   AGE
llama4-vllm-xxxxxxxxx-xxxxx    1/1     Running   0          10m
```

**Step 5:** Verify the service

```bash
kubectl get svc -n llama4-vllm
```

```text
NAME              TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
llama4-vllm-svc   ClusterIP   172.20.xxx.xx   <none>        8000/TCP   10m
```


## Deploy Open WebUI and Chat with Llama 4

Now, let's deploy Open WebUI, which provides a ChatGPT-style chat interface to interact with the Llama 4 model.

**Step 1:** Deploy Open WebUI

```bash
kubectl apply -f open-webui.yaml
```

**Output:**

```text
namespace/open-webui created
deployment.apps/open-webui created
service/open-webui created
```

**Step 2:** Verify the deployment

```bash
kubectl get pods -n open-webui
```

```text
NAME                          READY   STATUS    RESTARTS   AGE
open-webui-xxxxxxxxx-xxxxx    1/1     Running   0          2m
```

**Step 3:** Access the Open WebUI

```bash
kubectl -n open-webui port-forward svc/open-webui 8080:80
```

Open your browser and navigate to [http://localhost:8080](http://localhost:8080)

**Step 4:** Register and start chatting

1. Sign up with your name, email, and password
2. Click "New Chat"
3. Select the Llama 4 Scout model from the dropdown
4. Start chatting!


## Testing with curl (Optional)

You can also test the Llama 4 model directly using curl commands.

**Step 1:** Port-forward the vLLM service

```bash
kubectl -n llama4-vllm port-forward svc/llama4-vllm-svc 8000:8000
```

**Step 2:** Test the /v1/models endpoint

```bash
curl http://localhost:8000/v1/models
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

**Step 3:** Test chat completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{"role": "user", "content": "Explain what Amazon EKS is in 2 sentences."}],
    "max_tokens": 100,
    "stream": false
  }'
```

**Response:**

```json
{
  "id": "chatcmpl-xxxxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Amazon Elastic Kubernetes Service (EKS) is a managed container orchestration service that makes it easy to run Kubernetes on AWS without needing to install and operate your own Kubernetes control plane. It automatically manages the availability and scalability of the Kubernetes control plane nodes, handles upgrades, and integrates with other AWS services for security, networking, and monitoring."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 75,
    "total_tokens": 90
  }
}
```

**Step 4:** Test streaming response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```


## Monitoring and Observability

### Check vLLM Pod Logs

Monitor the vLLM server logs to check model loading status and inference metrics:

```bash
kubectl logs -n llama4-vllm -l app=llama4-vllm -f
```

**Key metrics to watch in logs:**

- **Token throughput**: `Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s`
- **GPU KV Cache utilization**: `GPU KV cache usage: X%`
- **Request processing**: `Received request` and `Finished request` entries

### Check GPU Utilization

If you have NVIDIA DCGM Exporter or similar monitoring tools deployed:

```bash
# Check GPU memory usage on the node
kubectl exec -n llama4-vllm -it $(kubectl get pods -n llama4-vllm -l app=llama4-vllm -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

**Expected output:**

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1E.0 Off |                    0 |
|  0%   45C    P0    70W / 300W |  18000MiB / 24576MiB |     25%      Default |
+-------------------------------+----------------------+----------------------+
```

### Performance Metrics

vLLM provides built-in metrics at the `/metrics` endpoint. If you haven't already, set up port-forwarding:

```bash
kubectl -n llama4-vllm port-forward svc/llama4-vllm-svc 8000:8000
```

Then query the metrics:

```bash
curl http://localhost:8000/metrics
```

Key metrics include:
- `vllm:num_requests_running` - Number of requests currently being processed
- `vllm:num_requests_waiting` - Number of requests waiting in queue
- `vllm:gpu_cache_usage_perc` - GPU KV cache utilization percentage
- `vllm:avg_generation_throughput_toks_per_s` - Average token generation throughput


## Deploying Llama 4 Maverick (Multi-GPU)

For the larger Maverick model with 128 experts, you need multiple GPUs with tensor parallelism.

:::warning
Llama 4 Maverick in BF16 requires ~800GB GPU memory, which exceeds the capacity of any single-node instance. This blueprint uses the **FP8 quantized version** from RedHatAI, which reduces memory to ~400GB and fits on p4de.24xlarge or p5.48xlarge instances.
:::

:::danger
Important: Deploying Llama 4 Maverick requires 8x A100 (80GB) or 8x H100 (80GB) GPUs. This can be very expensive. Ensure you monitor your usage carefully.
:::

**Step 1:** Deploy the 70B model

```bash
cd ai-on-eks/blueprints/inference/llama4-vllm-gpu/
envsubst < llama4-vllm-deployment-70b.yml | kubectl apply -f -
```

**Step 2:** Monitor the deployment

```bash
kubectl get pods -n llama4-vllm -l model=llama4-maverick -w
```

:::info
The Maverick model deployment may take 30-45 minutes due to larger model weights download and multi-GPU initialization.
:::

**Step 3:** Test the Maverick model

You can test using curl:

```bash
kubectl -n llama4-vllm port-forward svc/llama4-vllm-70b-svc 8001:8000
```

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Alternatively, if Open WebUI is already deployed, the Maverick model will automatically appear in the model dropdown. Simply select `RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8` from the list to start chatting.

## Cleanup

To remove all deployed resources:

**Step 1:** Delete Open WebUI

```bash
kubectl delete -f open-webui.yaml
```

**Step 2:** Delete Llama 4 vLLM deployment

```bash
kubectl delete -f llama4-vllm-deployment.yml
# Or for 70B model:
kubectl delete -f llama4-vllm-deployment-70b.yml
```

**Step 3:** Delete namespaces (optional)

```bash
kubectl delete namespace llama4-vllm
kubectl delete namespace open-webui
```

:::info
After deleting the GPU workloads, EKS Auto Mode will automatically terminate idle GPU nodes to reduce costs.
:::

## Key Takeaways

1. **EKS Auto Mode Simplifies GPU Provisioning**: No need to configure Karpenter or manage node groups manually.

2. **vLLM Provides High Performance**: Optimized memory management with PagedAttention enables efficient inference.

3. **OpenAI-Compatible API**: Easy integration with existing tools and applications.

4. **Scalable Architecture**: Support for both single-GPU (8B) and multi-GPU (70B) deployments.

5. **Cost Optimization**: EKS Auto Mode automatically terminates idle GPU nodes.
