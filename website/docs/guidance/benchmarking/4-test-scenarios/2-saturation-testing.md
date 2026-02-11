---
sidebar_label: Scenario 2 - Saturation Testing
---

# SCENARIO 2: Saturation Testing

## When to use this scenario:
Use multi-stage saturation testing when you need to empirically determine your system's maximum sustainable throughput before performance degrades. This is critical before production launch, when planning capacity, or setting autoscaling thresholds, as it answers "what's the highest QPS we can reliably handle?" Through systematic load increases, you'll observe where latency starts climbing or errors appear, revealing your true capacity ceiling that marketing materials and theoretical calculations often overestimate.

## Deployment

### Using Helm Chart (Recommended)

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# Install saturation scenario
helm install saturation-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=saturation \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# Monitor progress through multiple stages
kubectl logs -n benchmarking -l benchmark.scenario=saturation -f
```

### Customizing Load Stages

Adjust the QPS stages to match your expected capacity:

```yaml
# custom-saturation.yaml
benchmark:
  scenario: saturation
  target:
    baseUrl: http://your-model.your-namespace:8000
  scenarios:
    saturation:
      load:
        stages:
          - rate: 10
            duration: 180
          - rate: 25
            duration: 180
          - rate: 50
            duration: 180
          - rate: 75
            duration: 180
```

```bash
helm install saturation-test ai-on-eks/benchmark-charts -f custom-saturation.yaml -n benchmarking
```

## Key Configuration:

* Variable synthetic data distributions (mean 512/256 tokens, realistic variance)
* Multi-stage constant load: 5 → 10 → 20 → 40 QPS (3 minutes each)
* Streaming enabled
* 8 concurrent workers

## Understanding the results:
Plot P50, P95, and P99 latency across all stages to visually identify the saturation point—look for the stage where percentiles diverge sharply or error rates spike. The "knee" in your latency curve (where it hockey-sticks upward) indicates your practical capacity limit, not the theoretical maximum QPS your system handled. Set production targets 20-30% below this saturation point to maintain headroom for traffic spikes; if saturation occurs at 35 QPS, target 24-28 QPS sustained load. Compare different model configurations or hardware setups using identical test stages to make objective scaling decisions backed by data rather than vendor claims.

<details>
<summary><strong>Alternative: Raw Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-saturation
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: synthetic
      input_distribution:
        mean: 512
        std_dev: 128
        min: 128
        max: 2048
      output_distribution:
        mean: 256
        std_dev: 64
        min: 32
        max: 512

    load:
      type: constant
      stages:
        - rate: 5
          duration: 180
        - rate: 10
          duration: 180
        - rate: 20
          duration: 180
        - rate: 40
          duration: 180
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
        path: "saturation-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-saturation
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
          name: inference-perf-saturation
```

</details>
