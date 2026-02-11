---
sidebar_position: 1
---

# Inference on EKS

Deploy and run Large Language Models (LLMs) and other AI models on Amazon EKS with production-ready patterns and best practices.

## What's in This Section

This section provides practical deployment guides and Helm charts for running inference workloads on EKS. Whether you're deploying open-source LLMs, diffusion models, or custom AI models, you'll find ready-to-use configurations and step-by-step instructions.

---

## [Inference Charts](./inference-charts.md)

Helm charts for deploying popular AI models on EKS with pre-configured values for optimal performance.

**What You Get:**
- Ready-to-deploy Helm charts for vLLM, Ray-vLLM, Triton, and Diffusers
- Pre-configured values files for popular models (Llama, DeepSeek, Mistral, Stable Diffusion, and more)
- Support for both GPU (NVIDIA) and Neuron (AWS Inferentia/Trainium) deployments
- Production-ready configurations with health checks, autoscaling, and monitoring

**Use Cases:**
- Quick deployment of open-source LLMs for testing or production
- Standardized deployment patterns across your organization
- Reference implementations for custom model deployments

[Explore Inference Charts →](./inference-charts.md)

---

## [Framework-Specific Deployment Guides](./framework-guides/)

Detailed guides for deploying specific models and frameworks on EKS, organized by hardware type.

### [GPU Deployments](./framework-guides/GPUs/)

Step-by-step guides for deploying models on NVIDIA GPUs:

- **[AIBrix DeepSeek Distill](./framework-guides/GPUs/aibrix-deepseek-distill)** - Deploy DeepSeek R1 Distill Llama 8B with AIBrix optimization
- **[NVIDIA Dynamo](./framework-guides/GPUs/nvidia-dynamo)** - Deploy models with NVIDIA's Dynamo framework
- **[NVIDIA NIM Llama 3](./framework-guides/GPUs/nvidia-nim-llama3)** - Deploy Llama 3 using NVIDIA NIM
- **[NVIDIA NIM Operator](./framework-guides/GPUs/nvidia-nim-operator)** - Kubernetes operator for NVIDIA NIM deployments
- **[Ray vLLM DeepSeek](./framework-guides/GPUs/ray-vllm-deepseek)** - Deploy DeepSeek models with Ray and vLLM
- **[Stable Diffusion on GPUs](./framework-guides/GPUs/stablediffusion-gpus)** - Deploy Stable Diffusion models for image generation
- **[vLLM with NVIDIA Triton Server](./framework-guides/GPUs/vLLM-NVIDIATritonServer)** - Production inference with Triton and vLLM
- **[vLLM with Ray Serve](./framework-guides/GPUs/vLLM-rayserve)** - Scalable inference with Ray Serve and vLLM

### [Neuron Deployments](./framework-guides/Neuron/)

Step-by-step guides for deploying models on AWS Inferentia and Trainium:

- **[Mistral 7B on Inf2](./framework-guides/Neuron/Mistral-7b-inf2)** - Deploy Mistral 7B on AWS Inferentia 2
- **[Llama 2 on Inf2](./framework-guides/Neuron/llama2-inf2)** - Deploy Llama 2 13B on AWS Inferentia 2
- **[Llama 3 on Inf2](./framework-guides/Neuron/llama3-inf2)** - Deploy Llama 3 on AWS Inferentia 2
- **[Ray Serve High Availability](./framework-guides/Neuron/rayserve-ha)** - Deploy highly available Ray Serve on Neuron
- **[Stable Diffusion on Inf2](./framework-guides/Neuron/stablediffusion-inf2)** - Deploy Stable Diffusion on AWS Inferentia 2
- **[vLLM Ray on Inf2](./framework-guides/Neuron/vllm-ray-inf2)** - Deploy vLLM with Ray on AWS Inferentia 2

[Explore Framework Guides →](./framework-guides/)

---

## Getting Started

1. **Set up your infrastructure** - Start with the [Inference-Ready Cluster](../../infra/inference/inference-ready-cluster) to provision an EKS cluster optimized for AI/ML workloads

2. **Choose your deployment method**:
   - For quick deployments with popular models → Use [Inference Charts](./inference-charts.md)
   - For specific frameworks or custom configurations → Follow [Framework-Specific Guides](./framework-guides/)

3. **Optimize your deployment** - Apply best practices from the [Guidance section](../../guidance/) to improve performance and reduce costs

---

## Need Help?

- **Infrastructure Setup**: See [Inference Infrastructure](../../infra/inference/) for cluster setup and configuration
- **Optimization**: Check the [Guidance section](../../guidance/) for performance tuning and best practices
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/awslabs/ai-on-eks/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/awslabs/ai-on-eks/discussions)
