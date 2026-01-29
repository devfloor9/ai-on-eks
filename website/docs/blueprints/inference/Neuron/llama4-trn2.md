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

### Model Compilation (Tracing) Requirement

:::danger Important
Before deploying Llama 4 on Trainium2, the model must be **pre-compiled (traced)** for Neuron. This is a one-time process that converts the model weights to Neuron-optimized format.

The compiled artifacts must be stored in a location accessible to the deployment (e.g., S3 bucket or EFS volume) and referenced via the `NEURON_COMPILED_ARTIFACTS` environment variable.

See the [NxD Inference Llama 4 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/llama4-tutorial.html) for detailed compilation instructions.
:::


<CollapsibleContent header={<h2><span>Prerequisites and EKS Cluster Setup</span></h2>}>

### Prerequisites

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Helm](https://helm.sh/docs/intro/install/)
4. [eksctl](https://eksctl.io/installation/)

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


<CollapsibleContent header={<h2><span>Model Compilation (Required First Step)</span></h2>}>

Before deploying, you must compile the Llama 4 model for Neuron. This process creates optimized artifacts that can be reused across deployments.

### Option 1: Compile on a Trainium2 Instance

SSH into a trn2.48xlarge instance with the Neuron SDK installed:

```bash
# Activate Neuron virtual environment
source /opt/aws_neuronx_venv_pytorch/bin/activate

# Install vLLM-Neuron plugin
pip install vllm-neuron --upgrade

# Set compilation output directory
export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/llama4/traced_models/Llama-4-Scout-17B-16E-Instruct"

# Run vLLM to trigger compilation (this takes 30-60 minutes)
python3 -m vllm.entrypoints.openai.api_server \
  --model "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --tensor-parallel-size 64 \
  --port 8000
```

### Option 2: Use Pre-compiled Artifacts

If you have access to pre-compiled artifacts, upload them to S3:

```bash
aws s3 sync /home/ubuntu/llama4/traced_models/ s3://your-bucket/llama4-neuron-artifacts/
```

### Neuron Configuration for Compilation

For optimal performance, use this configuration during compilation:

```python
scout_neuron_config = {
    "text_config": {
        "batch_size": 1,
        "is_continuous_batching": True,
        "seq_len": 16384,
        "enable_bucketing": True,
        "context_encoding_buckets": [256, 512, 1024, 2048, 4096, 8192, 10240, 16384],
        "token_generation_buckets": [256, 512, 1024, 2048, 4096, 8192, 10240, 16384],
        "torch_dtype": "float16",
        "async_mode": True,
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

</CollapsibleContent>


## Deploying Llama 4 Scout with Helm

:::caution
The use of [Llama 4](https://huggingface.co/meta-llama) models requires access through a Hugging Face account. Make sure you have accepted the model license on HuggingFace.
:::

**Step 1:** Add the AI on EKS Helm repository

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts
helm repo update
```

**Step 2:** Create a Kubernetes secret for Hugging Face token

```bash
kubectl create secret generic hf-token \
  --from-literal=hf-token=$(echo -n "Your-Hugging-Face-Hub-Token-Value" | base64)
```

**Step 3:** Deploy Llama 4 Scout using Helm

```bash
helm install llama4-scout ai-on-eks/inference-charts \
  -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/main/charts/inference-charts/values-llama-4-scout-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://your-bucket/llama4-neuron-artifacts/Llama-4-Scout-17B-16E-Instruct"
```

**Step 4:** Monitor the deployment

```bash
kubectl get pods -w
```

:::info
With pre-compiled artifacts, the deployment should be ready in 10-15 minutes. Without pre-compiled artifacts, the first deployment will trigger compilation which takes 30-60 minutes.
:::

```text
NAME                                      READY   STATUS    RESTARTS   AGE
llama-4-scout-17b-vllm-nrn-xxxxx-xxxxx    1/1     Running   0          15m
```


## Deploying Llama 4 Maverick

For the larger Maverick model with 128 experts:

```bash
helm install llama4-maverick ai-on-eks/inference-charts \
  -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/main/charts/inference-charts/values-llama-4-maverick-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://your-bucket/llama4-neuron-artifacts/Llama-4-Maverick-17B-128E-Instruct"
```

:::warning
Maverick model compilation takes significantly longer (60-90 minutes) due to the larger number of experts.
:::


## Testing the Deployment

### Text Completion

```bash
kubectl port-forward svc/llama-4-scout-17b-vllm-nrn 8000:8000
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


## Helm Values Configuration

The Helm chart uses the following key configuration for Trainium2 deployments:

```yaml
# values-llama-4-scout-17b-vllm-neuron.yaml
model: meta-llama/Llama-4-Scout-17B-16E-Instruct

modelParameters:
  maxModelLen: 16384
  tensorParallelSize: 64
  maxNumSeqs: 1

inference:
  serviceName: llama-4-scout-17b-vllm-nrn
  accelerator: neuron
  framework: vllm

  modelServer:
    image:
      repository: public.ecr.aws/neuron/pytorch-inference-neuronx
      tag: 2.5.1-neuronx-py310-sdk2.21.0-ubuntu22.04
    deployment:
      resources:
        neuron:
          requests:
            aws.amazon.com/neuron: 32
            memory: 512Gi
          limits:
            aws.amazon.com/neuron: 32
            memory: 768Gi
      nodeSelector:
        node.kubernetes.io/instance-type: trn2.48xlarge
    env:
      VLLM_USE_V1: "1"
      NEURON_COMPILED_ARTIFACTS: ""  # Set via --set flag
```


## Monitoring

### Check Pod Logs

```bash
kubectl logs -l app=llama-4-scout-17b-vllm-nrn -f
```

### Check Neuron Device Utilization

```bash
kubectl exec -it $(kubectl get pods -l app=llama-4-scout-17b-vllm-nrn -o jsonpath='{.items[0].metadata.name}') -- neuron-top
```


## Cleanup

```bash
helm uninstall llama4-scout
# Or for Maverick:
helm uninstall llama4-maverick
```


## Key Takeaways

1. **Pre-compilation Required**: Unlike GPU deployments, Trainium2 requires model compilation before deployment.

2. **Cost-Effective Inference**: Trainium2 provides excellent price-performance for large MoE models like Llama 4.

3. **No Quantization Needed**: Trainium2's 1.5 TiB HBM memory supports full BF16 precision for both Scout and Maverick.

4. **Multimodal Support**: Llama 4 on Trainium2 supports both text and image inputs.

5. **Helm-based Deployment**: Use the AI on EKS inference charts for standardized, reproducible deployments.


## References

- [NxD Inference Llama 4 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/llama4-tutorial.html)
- [AI on EKS Inference Charts](https://github.com/awslabs/ai-on-eks-charts)
- [vLLM-Neuron Plugin](https://github.com/aws-neuron/vllm-neuron)
- [AWS Neuron SDK Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Llama 4 on Hugging Face](https://huggingface.co/meta-llama)
