---
sidebar_label: Scenario 1 - Baseline Performance
---

# SCENARIO 1: Baseline Performance

## When to use this scenario:
Use baseline testing when establishing your system's optimal performance with zero contention, essentially taking your infrastructure's vital signs. This is your starting point before any capacity planning or optimization work, ideal when you've just deployed a new endpoint or made infrastructure changes. It answers "what's the best performance this system can deliver?" without queueing or resource competition, giving you a clean reference point for all future testing.

## Deployment

### Using Helm Chart (Recommended)

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# Install baseline scenario
helm install baseline-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# Monitor progress
kubectl logs -n benchmarking -l benchmark.scenario=baseline -f
```

### Customizing Configuration

Override specific values using `--set` or a custom values file:

```bash
# Adjust test duration or resources
helm install baseline-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://your-model.your-namespace:8000 \
  --set benchmark.scenarios.baseline.load.stages[0].duration=600 \
  --set benchmark.resources.main.requests.cpu=4 \
  --namespace benchmarking
```

Or create a custom `my-values.yaml`:

```yaml
benchmark:
  scenario: baseline
  target:
    baseUrl: http://your-model.your-namespace:8000
    modelName: your-model-name
  scenarios:
    baseline:
      load:
        stages:
          - rate: 1
            duration: 600  # Longer test
```

```bash
helm install baseline-test ai-on-eks/benchmark-charts -f my-values.yaml -n benchmarking
```

## Key Configuration:

* Fixed-length synthetic data (512 input / 128 output tokens)
* Constant load at 1 QPS for 300 seconds
* Streaming enabled
* Pod affinity ensures same-AZ placement with inference pods

## Understanding the results:
Your TTFT and ITL at 1 QPS represent the theoretical minimum latency, the absolute fastest your system can respond with zero queueing or contention. If baseline TTFT is 800ms, users will never see faster response times regardless of optimizations, like adding replicas, load balancers, or autoscaling, because these improve **throughput and concurrency**, not single-request speed. Focus on these metrics as your performance floor: schedule delay should be near zero (`<10ms`), and any deviation indicates the test runner itself needs more resources. Compare baseline numbers against your Service Level Agreement (SLA) targets; if baseline performance doesn't meet requirements, you need model/hardware optimization before worrying about scale, as adding capacity won't improve fundamental inference speed.

<details>
<summary><strong>Alternative: Raw Kubernetes YAML</strong> (for educational purposes or custom deployments)</summary>

If you prefer not to use Helm or need to customize beyond values, here's the complete Kubernetes manifest:

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-baseline
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
        std_dev: 0
        min: 512
        max: 512
      output_distribution:
        mean: 128
        std_dev: 0
        min: 128
        max: 128

    load:
      type: constant
      stages:
        - rate: 1
          duration: 300
      num_workers: 4

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
        path: "baseline-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-baseline
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
          name: inference-perf-baseline
```

Apply with: `kubectl apply -f 01-scenario-baseline.yaml`

</details>
