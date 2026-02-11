---
sidebar_label: Scenario 4 - Production Simulation
---

# SCENARIO 4: Production Simulation

## When to use this scenario:
Deploy production simulation as your final validation before launch; it replicates real-world traffic chaos with variable request sizes and Poisson (bursty) arrivals instead of uniform load. Use this after optimizing based on baseline and saturation tests to answer "will users have a good experience under realistic conditions?" Real production traffic doesn't consist of identical 512-token requests arriving like clockwork; users send varying lengths at random intervals, and this test validates your system handles that heterogeneity while maintaining acceptable percentile latencies for SLA setting.

## Deployment

### Using Helm Chart (Recommended)

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# Install production scenario
helm install production-sim ai-on-eks/benchmark-charts \
  --set benchmark.scenario=production \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# Monitor progress - expect variable latency due to bursty traffic
kubectl logs -n benchmarking -l benchmark.scenario=production -f
```

### Customizing Traffic Patterns

Adjust burst rate and variability:

```yaml
# custom-production.yaml
benchmark:
  scenario: production
  target:
    baseUrl: http://your-model.your-namespace:8000
  scenarios:
    production:
      data:
        input:
          mean: 2048          # Longer average prompts
          stdDev: 1024        # Higher variability
          min: 256
          max: 8192
      load:
        type: poisson         # Keeps bursty arrivals
        stages:
          - rate: 20          # Higher target QPS
            duration: 900     # Longer test (15 min)
```

```bash
helm install production-sim ai-on-eks/benchmark-charts -f custom-production.yaml -n benchmarking
```

## Key Configuration:

* Variable synthetic data (Gaussian distributions for input/output)
* Wide token distributions (mean 1024/512 with high variance)
* Poisson (bursty) arrivals instead of uniform load
* Streaming enabled
* 8 concurrent workers

## Understanding the results:
Focus exclusively on P99 and P95 latency; these percentiles represent the worst experience that 99% and 95% of users encounter, unlike averages that hide poor tail performance. The wide input/output distributions create natural variability, so expect higher variance than baseline tests; this is normal and reflects production reality. Poisson bursts cause temporary queue buildup even at sustainable average rates, so if P99 is significantly worse than uniform-load testing suggested, you need more headroom than expected. Set SLAs based on these realistic percentiles, not averages; if P99 TTFT is 1200ms, don't promise sub-second latency even though mean might be 400ms.

<details>
<summary><strong>Alternative: Raw Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-production
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: synthetic
      input_distribution:
        mean: 1024
        std_dev: 512
        min: 128
        max: 4096
      output_distribution:
        mean: 512
        std_dev: 256
        min: 50
        max: 2048

    load:
      type: poisson  # Realistic bursty arrivals
      stages:
        - rate: 15
          duration: 600
      num_workers: 8

    server:
      type: vllm
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

    tokenizer:
      pretrained_model_name_or_path: Qwen/Qwen3-8B

    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "production-sim/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-production
  namespace: benchmarking
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: inference-perf-sa

      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app.kubernetes.io/component: qwen3-vllm
            topologyKey: topology.kubernetes.io/zone

      containers:
      - name: inference-perf
        image: quay.io/inference-perf/inference-perf:v0.2.0
        command: ["/bin/sh", "-c"]
        args:
        - |
          inference-perf --config_file /workspace/config.yml
        volumeMounts:
        - name: config
          mountPath: /workspace/config.yml
          subPath: config.yml
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"

      volumes:
      - name: config
        configMap:
          name: inference-perf-production
```

</details>
