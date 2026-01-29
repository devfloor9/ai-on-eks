---
title: Llama 4 with vLLM on Trainium2
sidebar_position: 7
---
import CollapsibleContent from '../../../../src/components/CollapsibleContent';

# Llama 4 with vLLM on Amazon EKS using Trainium2

This guide demonstrates deploying [Llama 4](https://huggingface.co/meta-llama) models using [vLLM](https://github.com/vllm-project/vllm) with [NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/) on [AWS Trainium2](https://aws.amazon.com/machine-learning/trainium/) instances.

:::info
This blueprint uses **AWS Trainium2 (trn2)** instances with the Neuron SDK for cost-effective inference. Llama 4 models support both text and image inputs (multimodal).
:::

## Understanding Trainium2 Requirements

Llama 4 models use a Mixture of Experts (MoE) architecture that requires significant compute resources. Trainium2 provides excellent price-performance for these large models.

### Model Memory Requirements

| Model | Active Params | Total Params | Experts | BF16 Memory | Instance Required | tensor_parallel_size |
|-------|---------------|--------------|---------|-------------|-------------------|---------------------|
| Llama 4 Scout | 17B | ~109B | 16 | ~220 GiB | trn2.48xlarge | 64 |
| Llama 4 Maverick | 17B | ~400B | 128 | ~800 GiB | trn2.48xlarge | 64 |

:::info
Unlike GPU deployments, Trainium2 uses the **original BF16/FP16 models** without quantization. The Neuron SDK efficiently manages memory across all 64 Neuron cores.
:::

### Trainium2 Instance Specifications

| Instance Type | Neuron Devices | Neuron Cores | HBM Memory | Use Case |
|--------------|----------------|--------------|------------|----------|
| trn2.48xlarge | 32 | 64 | 1.5 TiB | Scout & Maverick (BF16) |

:::warning
Llama 4 models require `tensor_parallel_size=64` which means you need a full trn2.48xlarge instance with all 64 Neuron cores. The 1.5 TiB HBM memory is sufficient for both Scout and Maverick models in BF16 precision.
:::


<CollapsibleContent header={<h2><span>Prerequisites and EKS Cluster Setup</span></h2>}>

### Prerequisites

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [eksctl](https://eksctl.io/installation/)
4. [envsubst](https://pypi.org/project/envsubst/)

### EKS Cluster Requirements

- **EKS Version**: >= 1.30
- **Trainium2 Node Group**: trn2.48xlarge instances
- **Neuron Device Plugin**: Installed and configured

### Create EKS Cluster with Trainium2 Support

```bash
eksctl create cluster \
  --name llama4-trn2-cluster \
  --region us-east-1 \
  --node-type trn2.48xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 2
```

### Install Neuron Device Plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin.yml
kubectl apply -f https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-scheduler-eks.yml
```

### Verify Neuron Devices

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["aws.amazon.com/neuron"]'
```

Expected output for trn2.48xlarge:
```text
"32"
```

</CollapsibleContent>


## Deploying Llama 4 Scout with vLLM

:::caution
The use of [Llama 4](https://huggingface.co/meta-llama) models requires access through a Hugging Face account. Make sure you have accepted the model license on HuggingFace.
:::

**Step 1:** Export the Hugging Face Hub Token

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
cd blueprints/inference/llama4-vllm-trn2/
envsubst < llama4-vllm-trn2-deployment.yaml | kubectl apply -f -
```

**Output:**

```text
namespace/llama4-vllm created
secret/hf-token created
configmap/llama4-neuron-config created
deployment.apps/llama4-vllm-trn2 created
service/llama4-vllm-trn2-svc created
```

**Step 4:** Monitor the deployment

```bash
kubectl get pods -n llama4-vllm -w
```

:::info
The first deployment may take 30-60 minutes as the model needs to be compiled (traced) for Neuron. Subsequent deployments with cached artifacts will be faster.
:::

```text
NAME                                READY   STATUS    RESTARTS   AGE
llama4-vllm-trn2-xxxxxxxxx-xxxxx    1/1     Running   0          45m
```


## Deploy Open WebUI

Deploy Open WebUI for a ChatGPT-style interface:

**Step 1:** Deploy Open WebUI

```bash
kubectl apply -f open-webui.yaml
```

**Step 2:** Access the UI

```bash
kubectl -n open-webui port-forward svc/open-webui 8080:80
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

**Step 3:** Register and start chatting

1. Sign up with your name, email, and password
2. Click "New Chat"
3. Select the Llama 4 Scout model
4. Start chatting with text or upload images!


## Testing with curl

### Text Completion

```bash
kubectl -n llama4-vllm port-forward svc/llama4-vllm-trn2-svc 8000:8000
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{"role": "user", "content": "What is Amazon EKS?"}],
    "max_tokens": 100
  }'
```

### Multimodal (Image + Text)

Llama 4 supports multimodal inputs with up to 5 images per prompt:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://httpbin.org/image/png"}},
        {"type": "text", "text": "Describe this image in detail"}
      ]
    }]
  }'
```

### Multiple Images

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://httpbin.org/image/png"}},
        {"type": "image_url", "image_url": {"url": "https://httpbin.org/image/png"}},
        {"type": "text", "text": "Compare these two images"}
      ]
    }]
  }'
```


## Deploying Llama 4 Maverick

For the larger Maverick model with 128 experts:

```bash
envsubst < llama4-vllm-trn2-maverick.yaml | kubectl apply -f -
```

Monitor deployment:

```bash
kubectl get pods -n llama4-vllm -l model=llama4-maverick -w
```

:::warning
Maverick model compilation takes significantly longer (60-90 minutes) due to the larger number of experts.
:::


## Neuron Configuration

The deployment includes optimized Neuron configuration for Llama 4:

```json
{
  "text_config": {
    "batch_size": 1,
    "is_continuous_batching": true,
    "seq_len": 16384,
    "torch_dtype": "float16",
    "async_mode": true,
    "world_size": 64,
    "tp_degree": 64,
    "cp_degree": 16
  },
  "vision_config": {
    "batch_size": 1,
    "seq_len": 8192,
    "torch_dtype": "float16",
    "tp_degree": 16,
    "dp_degree": 4,
    "world_size": 64
  }
}
```

Key parameters:
- `tp_degree=64`: Tensor parallelism across all Neuron cores
- `cp_degree=16`: Context parallelism for efficient attention
- `async_mode=true`: Asynchronous execution for better throughput
- `enable_bucketing=true`: Dynamic batching for variable input lengths


## Monitoring

### Check Pod Logs

```bash
kubectl logs -n llama4-vllm -l app=llama4-vllm-trn2 -f
```

### Check Neuron Device Utilization

```bash
kubectl exec -n llama4-vllm -it $(kubectl get pods -n llama4-vllm -l app=llama4-vllm-trn2 -o jsonpath='{.items[0].metadata.name}') -- neuron-top
```


## Cleanup

```bash
kubectl delete -f open-webui.yaml
kubectl delete -f llama4-vllm-trn2-deployment.yaml
# Or for Maverick:
kubectl delete -f llama4-vllm-trn2-maverick.yaml
```


## Key Takeaways

1. **Cost-Effective Inference**: Trainium2 provides excellent price-performance for large MoE models like Llama 4.

2. **Multimodal Support**: Llama 4 on Trainium2 supports both text and image inputs.

3. **NxD Inference**: The Neuron SDK's NxD Inference library enables efficient distributed inference.

4. **OpenAI-Compatible API**: vLLM provides an OpenAI-compatible API for easy integration.

5. **Model Compilation**: First deployment requires model compilation (tracing), but subsequent deployments are faster with cached artifacts.


## References

- [NxD Inference Llama 4 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/llama4-tutorial.html)
- [vLLM-Neuron Plugin](https://github.com/aws-neuron/vllm-neuron)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Llama 4 on Hugging Face](https://huggingface.co/meta-llama)
