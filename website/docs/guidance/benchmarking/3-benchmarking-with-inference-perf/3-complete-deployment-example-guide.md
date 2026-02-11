---
sidebar_label: Complete Deployment Example
---

# Complete Deployment Example

This example demonstrates a production-ready deployment with S3 storage, realistic load testing, and proper AWS integration. Follow these steps to deploy your benchmark.

## STEP 0: ENVIRONMENT SETUP (OPTIONAL)

Choose your deployment path:

### Path A (Recommended): Use ai-on-eks blueprint

* Deploys kube-prometheus-stack automatically
* Fixed Prometheus URL: http://kube-prometheus-stack-prometheus.monitoring:9090
* Follow: https://awslabs.github.io/ai-on-eks/docs/infra/inference-ready-cluster

### Path B:  Existing EKS Cluster

If you have an existing cluster, ensure these prerequisites:

* EKS cluster with Kubernetes 1.28+
* GPU nodes (g5.xlarge or larger) with NVIDIA drivers installed
* Karpenter (optional but recommended for autoscaling)
* Pod Identity or IRSA configured for S3 access
* kubectl configured with cluster access
* MUST have Prometheus pre-deployed
* Metrics collection is OPTIONAL for benchmarking
* You must know your Prometheus service name and namespace
* Example: http://&lt;your-prometheus-service&gt;.&lt;namespace&gt;:9090

## STEP 1: Deploy Inference Model

Before running benchmarks, you need an active LLM inference endpoint.

**Path A Users:** If you deployed using the ai-on-eks blueprint with a pre-configured inference deployment, skip to STEP 2.

**Path B Users:** Deploy vLLM with your chosen model using the inference-charts:

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# Deploy Qwen3-8B with vLLM
helm install qwen3-vllm ai-on-eks/inference-charts \
  --set model=Qwen/Qwen3-8B \
  --set inference.framework=vllm \
  --namespace default --create-namespace

# Verify deployment
kubectl get pods -n default -l app.kubernetes.io/name=inference-charts
kubectl logs -n default -l app.kubernetes.io/name=inference-charts -f
```

Wait for the model to be ready before proceeding to benchmarking. This typically takes 3-10 minutes depending on model size and download speed.

## STEP 2:  AWS Storage Setup (Using S3 - Recommendation)

**Note:** If you deployed your cluster using the ai-on-eks inference-ready-cluster blueprint, the EKS Pod Identity Agent addon is already installed. You can skip the addon installation command below and proceed directly to creating the S3 bucket and IAM role.

Set up AWS credentials so your benchmark pod can write results to S3 without hardcoded credentials.

```bash
# Create S3 bucket for benchmark results
export BUCKET_NAME="inference-perf-results-$(aws sts get-caller-identity --query Account --output text)"
aws s3 mb s3://${BUCKET_NAME} --region us-west-2

# Install EKS Pod Identity Agent (already deployed on the blueprint reference - https://awslabs.github.io/ai-on-eks/docs/infra/inference-ready-cluster)

aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name eks-pod-identity-agent \
  --addon-version v1.3.0-eksbuild.1

# Create IAM role with S3 permissions

cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "pods.eks.amazonaws.com"
    },
    "Action": [
      "sts:AssumeRole",
      "sts:TagSession"
    ]
  }]
}
EOF


aws iam create-role \
  --role-name InferencePerfRole \
  --assume-role-policy-document file://trust-policy.json



aws iam attach-role-policy \
  --role-name InferencePerfRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess



# Link the role to your Kubernetes service account

aws eks create-pod-identity-association \
  --cluster-name my-cluster \
  --namespace benchmarking \
  --service-account inference-perf-sa \
  --role-arn arn:aws:iam::ACCOUNT_ID:role/InferencePerfRole
```

## STEP 3: Deploy Benchmark Resources

### Option A: Using Helm Chart (Recommended)

The [AI on EKS Benchmark Helm Chart](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/benchmark-charts) provides a production-ready deployment with simplified configuration management.

**Install the benchmark:**

```bash
# Add the AI on EKS Helm repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# Deploy a production simulation test
helm install production-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=production \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace
```

**Customize with your own values:**

```yaml
# custom-benchmark.yaml
benchmark:
  scenario: production
  target:
    baseUrl: http://qwen3-vllm.default:8000
    modelName: qwen3-8b
    tokenizerPath: Qwen/Qwen3-8B

  # S3 storage configuration
  storage:
    s3:
      enabled: true
      bucketName: inference-perf-results
      path: "inference-perf/results"

  # Pod affinity for same-AZ placement
  affinity:
    enabled: true
    targetLabels:
      app.kubernetes.io/component: qwen3-vllm

  # Resource allocation
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
```

Deploy with custom values:
```bash
helm install production-test ai-on-eks/benchmark-charts \
  -f custom-benchmark.yaml \
  --namespace benchmarking --create-namespace
```

**Benefits of Helm approach:**
- **Simplified configuration** through values.yaml instead of verbose YAML
- **Pre-configured scenarios** (baseline, saturation, sweep, production)
- **Consistent defaults** for pod affinity, resources, and dependencies
- **Easy upgrades** and rollbacks with Helm versioning

### Option B: Manual Kubernetes YAML (Educational)

For learning purposes or highly customized deployments, you can deploy directly with Kubernetes manifests. This approach provides full transparency of all resources.

<details>
<summary><strong>Click to expand: Manual YAML deployment instructions</strong></summary>

#### Handling Model Dependencies

Some models require additional Python packages that aren't included in the base inference-perf container. For example, `sentencepiece` is needed for Mistral and Llama models. Qwen3 models use tiktoken which is already included, so no additional packages are required.

**Two approaches:**

#### Approach A: Runtime Installation (Recommended - Simple)
Install dependencies as part of the main container startup before running the benchmark:

```yaml
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

      ...

      containers:
      - name: inference-perf
        image: quay.io/inference-perf/inference-perf:v0.2.0
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "Installing dependencies..."
            pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5.29.2
            echo "Dependencies installed successfully"
            echo "Starting inference-perf..."
            inference-perf --config_file /workspace/config.yml

```

#### Approach B: Custom Container Image (Advanced)
Build a custom image with dependencies pre-installed:

```dockerfile
FROM quay.io/inference-perf/inference-perf:v0.2.0

RUN pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5.29.2
```

#### When to use each approach:

* Use **Approach A** for quick testing and flexibility
* Use **Approach B** for production repeatability and faster startup

### Create Namespace and Service Account

```bash
cat <<EOF | kubectl apply -f -
# Namespace for benchmark workloads
apiVersion: v1
kind: Namespace
metadata:
  name: benchmarking

---
# Service Account (linked to AWS IAM via Pod Identity)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-perf-sa
  namespace: benchmarking
EOF
```



### Create HuggingFace Token Secret (Optional but Recommended)

If your model requires authentication to download tokenizers from HuggingFace, create a secret using the command below. This approach is more secure than defining secrets in YAML files that might accidentally be committed to version control.

**Step 1: Obtain your HuggingFace token**

* Go to https://huggingface.co/settings/tokens
* Create a read token if you don't have one

**Step 2: Create the secret**

```bash
kubectl create secret generic hf-token \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN_HERE \
  --namespace=benchmarking
```


**Step 3: Verify the secret was created**

```bash
kubectl get secret hf-token -n benchmarking
```


**⚠️ Security Note:** Never commit secrets to Git repositories. Always use imperative commands or external secret management tools (AWS Secrets Manager, HashiCorp Vault, etc.) for production deployments.

### Create ConfigMap and Job

```bash
cat <<EOF | kubectl apply -f -
# Benchmark Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-config
  namespace: benchmarking
data:
  config.yml: |
    # API Configuration
    api:
      type: completion
      streaming: true
    # Data Generation - synthetic with realistic distributions
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
    # Load Pattern - Poisson arrivals at 10 QPS for 5 minutes
    load:
      type: poisson
      stages:
        - rate: 10
          duration: 300
      num_workers: 4
    # Model Server
    server:
      type: vllm
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

    # Tokenizer
    tokenizer:
      pretrained_model_name_or_path: Qwen/Qwen3-8B

    # Storage - Results automatically saved to S3
    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "inference-perf/results"
    # Optional: Prometheus metrics collection
    # metrics:
    #   type: prometheus
    #   prometheus:
    #     url: http://kube-prometheus-stack-prometheus.monitoring:9090
    #     scrape_interval: 15
EOF
---

cat <<EOF | kubectl apply -f -
# Benchmark Job
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
            echo "Starting inference-perf..."
            inference-perf --config_file /workspace/config.yml
        volumeMounts:
          - name: config
            mountPath: /workspace/config.yml
            subPath: config.yml
        env:
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token
                key: token
                optional: true
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

**💡 Tip:** The resource values shown are starting points. For higher concurrency levels or longer test durations, monitor pod resource usage with `kubectl top pod -n benchmarking` and adjust accordingly.

</details>

---

## STEP 4: Deploy and Monitor

### For Helm Deployments:

```bash
# Monitor job progress
kubectl get jobs -n benchmarking -w

# Follow logs to see benchmark progress
kubectl logs -n benchmarking -l app.kubernetes.io/component=benchmark -f

# Check Helm release status
helm status production-test -n benchmarking
```

### For Manual YAML Deployments:

Save the manifests from Option B above as `inference-perf-complete.yaml` and deploy:

```bash
# Deploy all resources
kubectl apply -f inference-perf-complete.yaml

# Monitor job progress
kubectl get jobs -n benchmarking -w

# Follow logs to see benchmark progress
kubectl logs -n benchmarking -l app=inference-perf -f
```

## STEP 5: Retrieve Results

### With S3 Storage (Recommended):
Results are automatically uploaded to your S3 bucket. Access them directly:

```bash
# List results in S3 (use bucket name from STEP 2)
aws s3 ls s3://${BUCKET_NAME}/inference-perf/ --recursive

# Download specific report
aws s3 cp s3://${BUCKET_NAME}/inference-perf/20251020-143000/summary_lifecycle_metrics.json ./
```

### With Local Storage (Alternative):
If using `local_storage` instead of S3, you must manually copy results before the pod terminates:

```bash
# In config.yml, use:

storage:

  local_storage:

    path: "reports-results"



# Get pod name

POD_NAME=$(kubectl get pods -n benchmarking -l app=inference-perf -o jsonpath='{.items[0].metadata.name}')



# Copy results from pod

kubectl cp benchmarking/$POD_NAME:/reports-* ./local-reports/
```

### Storage Comparison:

| Feature | Local Storage | S3 Storage |
|---|---|---|
| Setup | None required | AWS credentials needed |
| Persistence | Manual copy required | Automatic |
| Best for | Quick tests, experimentation | Production, automation |
| Results access | kubectl cp command | AWS S3 commands/console |
