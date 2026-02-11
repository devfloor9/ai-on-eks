---
sidebar_label: Scenario 5 - Real Dataset Testing
---

# SCENARIO 5: Real Dataset Testing

## When to use this scenario:
Use real dataset testing to validate production-ready performance with actual user prompts and query patterns. This is essential when your model is fine-tuned for specific conversation patterns, when comparing model versions with real-world performance guarantees, or when you need to tell stakeholders "this is how it performs on actual conversations, not theoretical data." The tradeoff is less control over distributions, but you gain authenticity and the ability to discover edge cases that synthetic data misses.

## Deployment

### Using Helm Chart (Recommended)

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# ShareGPT dataset testing uses same scenarios but with real data
helm install sharegpt-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# Monitor for natural conversation complexity patterns
kubectl logs -n benchmarking -l app.kubernetes.io/component=benchmark -f
```

**Note:** Real dataset testing uses the same load patterns as scenarios 1-4, but with `data.type: shareGPT` instead of `data.type: synthetic`. You can apply real data to any scenario (baseline, saturation, sweep, or production).

### Using Custom Dataset

Provide your own conversation dataset:

```yaml
# custom-dataset.yaml
benchmark:
  scenario: saturation  # Or any scenario
  target:
    baseUrl: http://your-model.your-namespace:8000
  # Override data configuration to use custom dataset
  customData:
    enabled: true
    type: custom
    path: /path/to/your/conversations.json
    format: sharegpt  # or openai, alpaca, etc.
```

For custom datasets, you'll need to mount the data file into the benchmark pod using a ConfigMap or PersistentVolume.

## Key Configuration:

* ShareGPT real conversation dataset (varies by scenario choice)
* Any load pattern (constant/poisson, depends on selected scenario)
* Streaming enabled
* Natural conversation complexity and length variance

## Understanding the results:
Real conversations reveal natural complexity patterns and edge cases absent from synthetic data—look for latency outliers that expose problematic conversation structures or phrasings causing slow processing. Compare real data performance against synthetic tests at similar QPS; significant degradation suggests real conversations are more complex than your synthetic parameters assumed, helping calibrate future synthetic tests. TTFT variability will be higher due to natural context length variance from multi-turn dialogues, and any consistent error patterns with specific conversation types reveal production vulnerabilities worth targeted optimization. Use these results as ground truth for stakeholder commitments—base your "P99 latency will be X" promises on real data, not synthetic.

**⚠️ Critical:** Regularly update your test dataset with recent anonymized production samples to prevent drift. If your benchmark dataset is 6 months old but user behavior has shifted to longer prompts, your performance predictions will be inaccurate.

<details>
<summary><strong>Alternative: Raw Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-sharegpt
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: shareGPT  # Real conversation data

    load:
      type: constant
      stages:
        - rate: 10
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
        path: "sharegpt-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-sharegpt
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
          name: inference-perf-sharegpt
```

</details>
