---
sidebar_label: Running Inference Perf on EKS
---

# Running Inference Perf on EKS

Inference Perf is a GenAI inference performance benchmarking tool designed specifically for measuring LLM inference endpoint performance on Kubernetes. It runs as a Kubernetes Job and supports multiple model servers ([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TGI](https://github.com/huggingface/text-generation-inference)) with standardized metrics.

Why use a [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)? Jobs run once and terminate when complete, making them ideal for benchmarking tasks. Results are stored locally or in cloud storage ([S3](https://aws.amazon.com/s3/)).


## Prerequisites

* Kubernetes cluster with kubectl access (version 1.21+)
* A deployed inference endpoint with OpenAI-compatible API (vLLM, SGLang, TGI, or compatible)
* Namespace for running benchmarks
* Container image: quay.io/inference-perf/inference-perf:v0.2.0
* (Optional) HuggingFace token for downloading tokenizers
* (Optional) AWS credentials for S3 storage


## Model-Specific Dependencies

⚠️ IMPORTANT: Different models require different tokenizer packages:

| Model Family | Requires sentencepiece? | Examples |
|--------------|------------------------|----------|
| Mistral (all versions) | ✅ YES | mistralai/Mistral-7B-Instruct-v0.3 |
| Llama 2 | ✅ YES | meta-llama/Llama-2-7b-hf |
| Llama 3.1 | ✅ YES | meta-llama/Meta-Llama-3.1-8B |
| SmolLM2 | ❌ NO | HuggingFaceTB/SmolLM2-135M-Instruct |
| GPT models | ❌ NO | Various GPT variants |

If using Mistral or Llama models, you must install the sentencepiece package. See "Handling Model Dependencies" section below for implementation.


## Understanding the Inference Benchmark Framework architecture

Before deploying, it's important to understand the key configuration components that define your benchmark test.

### API Configuration

Defines how the tool communicates with your inference endpoint. You'll specify whether you're using completion or chat API, and whether streaming is enabled (required for measuring TTFT and ITL metrics).

```yaml
api:
  type: completion # completion or chat
  streaming: true # Enable for TTFT/ITL metrics
```

**Determining the Correct API Type:**

The inference-charts deployment automatically configures API endpoints based on the model's capabilities. To identify which endpoints are available:

**Method 1: Check vLLM Deployment Logs (Recommended)**

Check your vLLM server logs to see which API endpoints were enabled at startup:

```bash
# View vLLM startup logs showing enabled endpoints
kubectl logs -n default -l app.kubernetes.io/name=inference-charts --tail=100 | grep -i "route\|endpoint\|application"
```

Look for output indicating enabled routes:
- `Route: /v1/completions` → Use `type: completion`
- `Route: /v1/chat/completions` → Use `type: chat`

If both routes appear, use `completion` (simpler for benchmarking).

**Method 2: Check Model Capabilities (Optional)**

For understanding the model's theoretical capabilities, review the Hugging Face model card for your model. Models with a defined chat template typically support the chat completion API, but the actual deployment configuration determines what's enabled.

**Note:** The OpenAI completion API (`v1/completions`) is deprecated by OpenAI but remains widely supported by vLLM, SGLang, and TGI. Most inference-charts deployments enable it by default without additional configuration.

### Data Generation

Controls what data is sent to your inference endpoint. You can use real datasets (ShareGPT) or synthetic data with controlled distributions. Synthetic data is useful when you need specific input/output length patterns for testing.

```yaml
data:

  type: synthetic # shareGPT, synthetic, random, shared_prefix, etc.

  input_distribution:
    mean: 512      # Average input prompt length in tokens
    std_dev: 128   # Variation in prompt length (68% within ±128 tokens of mean)
    min: 128       # Minimum input tokens (clips distribution lower bound)
    max: 2048      # Maximum input tokens (clips distribution upper bound)

  output_distribution:
    mean: 256      # Average generated response length in tokens
    std_dev: 64    # Variation in response length (68% within ±64 tokens of mean)
    min: 32        # Minimum output tokens (clips distribution lower bound)
    max: 512       # Maximum output tokens (clips distribution upper bound)
```


### Load Generation

Defines your load pattern - how many requests per second and for how long. You can use multiple stages to test different load levels, or use sweep mode for automatic saturation detection.

```yaml
load:
  type: constant              # Use 'constant' for uniform arrival (predictable load) or 'poisson' for bursty traffic (realistic production)
  stages:
    - rate: 10                # Requests per second (QPS) - increase to test higher throughput, decrease for baseline/minimal load
      duration: 300           # How long to sustain this rate in seconds - longer durations (300-600s) ensure stable measurements
  num_workers: 4              # Concurrent workers generating load - increase if inference-perf can't achieve target rate (check scheduling delay in results)
```

**Note on num_workers:** This controls the benchmark tool's internal parallelism, not concurrent users. The default value of 4 works for most scenarios. Only increase if results show high `schedule_delay` (> 10ms), indicating the tool cannot maintain the target rate.

### Server Configuration

Specifies your inference endpoint details - server type, model name, and URL.

```yaml
server:

  type: vllm # vllm, sglang, or tgi

  model_name: qwen3-8b

  base_url: http://qwen3-vllm.default:8000

  ignore_eos: true
```

### Storage Configuration

Determines where benchmark results are saved. Local storage saves to the pod filesystem (requires manual copy), while S3 storage automatically persists results to your AWS bucket.

```yaml
storage:

  local_storage: # Default: saves in pod

    path: "reports-results"

  # ⚠️ Warning: local_storage results are lost when pod terminates
  # To retrieve results, add '&& sleep infinity' to the Job args and use:
  # kubectl cp <pod-name>:/workspace/reports-* ./local-results -n benchmarking

  # OR

  simple_storage_service: # S3: automatic persistence

    bucket_name: "my-results-bucket"

    path: "inference-perf/results"
```

### Metrics Collection (Optional)

Enables advanced metrics collection from Prometheus if your inference server exposes metrics.

```yaml
metrics:

  type: prometheus

  prometheus:

    url: http://kube-prometheus-stack-prometheus.monitoring:9090 # For ai-on-eks Path A; adjust service name/namespace for custom Prometheus

    scrape_interval: 15
```

**Note:** The Prometheus URL uses Kubernetes DNS format: `http://<service-name>.<namespace>:<port>`. If your Prometheus is deployed in a different namespace (e.g., `monitoring`, `observability`), update the URL accordingly. The benchmark Job runs in the `benchmarking` namespace, so cross-namespace service access must be specified.

## Infrastructure Topology for Reproducible Results

For accurate, comparable benchmarks across multiple runs, the inference-perf Job **MUST** be placed in the same AZ as your inference deployment.

### Why This Matters:

Without proper placement, benchmark results become unreliable:

* Cross-AZ network latency adds 1-2ms per request
* Results vary unpredictably across benchmark runs
* You cannot determine if performance changes are real or due to infrastructure placement
* Optimization decisions become impossible

### Example of the Problem:

```
First benchmark run:
- Benchmark pod in us-west-2a → Inference pod in us-west-2a
- Result: TTFT = 800ms

Second benchmark run (after pod restart):
- Benchmark pod in us-west-2b → Inference pod in us-west-2a
- Result: TTFT = 850ms
```

The 50ms difference is cross-AZ latency, not actual performance change.

**Note on Cross-AZ Testing:** While same-AZ placement is recommended for baseline benchmarking and performance optimization, cross-AZ testing is valuable for validating high-availability (HA) deployments where your inference service spans multiple availability zones. If your production deployment uses multi-AZ load balancing for fault tolerance, conduct separate benchmarks with cross-AZ placement to understand the latency impact users may experience during zone-specific routing.

### Required Configuration:

All benchmark Job examples in this guide include `affinity` configuration to enforce same-AZ placement using the standard Kubernetes topology label `topology.kubernetes.io/zone`:

```yaml
spec:
  template:
    spec:
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app.kubernetes.io/component: qwen3-vllm
            topologyKey: topology.kubernetes.io/zone
```


**IMPORTANT:** The `matchLabels` must match your actual vLLM deployment labels. Check your deployment's pod labels with:

```bash
kubectl get deployment qwen3-vllm -n default -o jsonpath='{.spec.template.metadata.labels}' && echo
```


Common label patterns:

* Standard deployments: `app: <service-name>` (simple pattern)
* inference-charts deployments: `app.kubernetes.io/component: <service-name>` (used in this guide's examples)
* Other Helm charts: `app.kubernetes.io/name: <service-name>`

Update the `matchLabels` section in the examples to match your deployment's actual pod labels.

### Verification:

After deploying, confirm both pods are in the same AZ:

```bash
# Check both pods - they should show the same zone
kubectl get pods -n default -o wide -l app.kubernetes.io/component=qwen3-vllm
kubectl get pods -n benchmarking -o wide -l app=inference-perf

# Expected output - both in same zone:
# qwen3-vllm-xxx      ip-10-0-1-100.us-west-2a...
# inference-perf-yyy  ip-10-0-1-200.us-west-2a...
```


### Optional: Instance Type Consistency

**Instance Sizing for Benchmark Pods**

The benchmark pod runs on a separate CPU node from your GPU-based inference deployment. The m6i.2xlarge instance type (8 vCPU, 32 GB RAM) provides sufficient capacity for load generation without competing for GPU node resources.

**Important:** The pod affinity configuration (`topology.kubernetes.io/zone`) ensures both pods are in the same availability zone for network consistency, NOT on the same physical node. Your cluster must have capacity for both:
- GPU nodes for inference (e.g., g5.2xlarge for models like Qwen3-8B)
- CPU nodes for benchmarking (e.g., m6i.2xlarge)

If using Karpenter, it will automatically provision the appropriate node types in the same AZ.

For maximum reproducibility (baseline benchmarks, CI/CD pipelines), you can specify the instance type:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: m6i.2xlarge
      affinity:
        podAffinity:
          # ... same as above
```


**When to use instance type selectors:**

* Creating benchmark baselines for documentation
* CI/CD pipelines requiring consistent results
* Preventing Karpenter from provisioning different instance families

**When NOT needed:**

* Homogeneous CPU node pools
* Comparative testing (before/after on same infrastructure)

### Troubleshooting:

If your benchmark Job stays in `Pending`:

```bash
kubectl describe pod -n benchmarking <pod-name>
```


Common issues:

* **No capacity in target AZ**: Scale cluster or use `preferredDuringSchedulingIgnoredDuringExecution`
* **Label mismatch**: Verify deployment labels match podAffinity selector
