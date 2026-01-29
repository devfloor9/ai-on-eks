# Llama 4 with vLLM on Trainium2

This blueprint has been moved to the [AI on EKS Inference Charts](https://github.com/awslabs/ai-on-eks-charts) repository.

## Deployment

Deploy using Helm:

```bash
# Add the Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts
helm repo update

# Deploy Llama 4 Scout on Trainium2
helm install llama4-scout ai-on-eks/inference-charts \
  -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/main/charts/inference-charts/values-llama-4-scout-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://your-bucket/llama4-neuron-artifacts/"

# Deploy Llama 4 Maverick on Trainium2
helm install llama4-maverick ai-on-eks/inference-charts \
  -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/main/charts/inference-charts/values-llama-4-maverick-17b-vllm-neuron.yaml \
  --set inference.modelServer.env.NEURON_COMPILED_ARTIFACTS="s3://your-bucket/llama4-neuron-artifacts/"
```

## Documentation

See the full documentation at: https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/Neuron/llama4-trn2

## Important Note

Llama 4 models on Trainium2 require **pre-compiled (traced) model artifacts**. See the [NxD Inference Llama 4 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/llama4-tutorial.html) for compilation instructions.
