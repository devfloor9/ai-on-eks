---
sidebar_label: Resources
---

# Resources

## Helm Chart Repository

The official benchmark charts are maintained in the [AI on EKS Charts repository](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/benchmark-charts). This repository contains:

- **values.yaml**: Complete configuration reference with all available options
- **templates/**: Kubernetes resource templates for jobs, configmaps, and service accounts
- **scenarios/**: Pre-configured scenario definitions (baseline, saturation, sweep, production)
- **README.md**: Detailed usage instructions and examples

### Customizing via values.yaml

Create a custom values file to override defaults:

```yaml
# custom-benchmark.yaml
benchmark:
  scenario: saturation
  target:
    baseUrl: http://your-model.your-namespace:8000
    modelName: your-model-name

  # Override scenario-specific settings
  scenarios:
    saturation:
      load:
        stages:
          - rate: 10
            duration: 300
          - rate: 50
            duration: 300

  # Resource allocation
  resources:
    requests:
      cpu: "4"
      memory: "8Gi"

  # Pod affinity customization
  affinity:
    enabled: true
    targetLabels:
      app: your-inference-service
```

Deploy with custom values:
```bash
helm install my-benchmark ai-on-eks/benchmark-charts -f custom-benchmark.yaml -n benchmarking
```

## Alternative: Custom Container with SentencePiece

For custom deployments outside the Helm chart, you can build a container image with pre-installed dependencies:

```bash
# Create a custom Dockerfile
cat > Dockerfile <<'EOF'
FROM quay.io/inference-perf/inference-perf:v0.2.0

# Install sentencepiece
RUN pip install --no-cache-dir sentencepiece protobuf

USER 1000
EOF

# Build and push to your registry
docker build -t <your-registry>/inference-perf:v0.2.0-sentencepiece .
docker push <your-registry>/inference-perf:v0.2.0-sentencepiece

# Update your Job to use the new image
kubectl patch job inference-perf-run -n benchmarking \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/image", "value":"<your-registry>/inference-perf:v0.2.0-sentencepiece"}]'
```

## Alternative: Complete Kubernetes Manifest

For manual deployments or educational purposes, here's a complete YAML manifest with runtime dependency installation:

```bash
cat > inference-perf-fixed.yaml <<'EOF'
---
apiVersion: v1
kind: Namespace
metadata:
  name: benchmarking
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-perf-sa
  namespace: benchmarking
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-config
  namespace: benchmarking
data:
  config.yml: |
    load_generator:
      concurrency: 10
      duration: 60

    model:
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

      tokenizer:
        pretrained_model_name_or_path: Qwen/Qwen3-8B

    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "inference-perf/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-run
  namespace: benchmarking
  labels:
    app: inference-perf
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: inference-perf
    spec:
      restartPolicy: Never
      serviceAccountName: inference-perf-sa

      # Same-AZ placement with inference pods for reproducible results
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
            echo "Installing dependencies..."
            pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5
            echo "Dependencies installed successfully"
            echo "Starting inference-perf..."
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
            name: inference-perf-config
EOF
```

## Execution

```bash
kubectl apply -f inference-perf-fixed.yaml
```
