# NVIDIA Enterprise RAG & AI-Q Research Assistant - Infrastructure Deployment

This guide covers the infrastructure deployment for the NVIDIA Enterprise RAG Blueprint and AI-Q Research Assistant on Amazon EKS.

## Table of Contents

- [What Are These Blueprints?](#what-are-these-blueprints)
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Infrastructure Components](#infrastructure-components)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Application Deployment](#application-deployment)
  - [Automated Deployment (Recommended)](#automated-deployment-recommended)
  - [Manual Deployment](#manual-deployment)
- [Next Steps](#next-steps)
- [Cleanup](#cleanup)
  - [Uninstall Applications Only](#uninstall-applications-only)
  - [Clean Up Infrastructure](#clean-up-infrastructure)
- [Cost Considerations](#cost-considerations)
- [Additional Resources](#additional-resources)

## What Are These Blueprints?

This infrastructure supports two powerful NVIDIA AI blueprints:

### NVIDIA AI-Q Research Assistant

[NVIDIA AI-Q Research Assistant](https://build.nvidia.com/nvidia/aiq) is an AI-powered research assistant that creates custom AI researchers capable of operating anywhere, informed by your own data sources, synthesizing hours of research in minutes. The AI-Q NVIDIA Blueprint enables developers to connect AI agents to enterprise data and use reasoning and tools to distill in-depth source materials with efficiency and precision.

**Key Capabilities:**
- **5x faster token generation** for rapid report synthesis
- **15x faster data ingestion** with better semantic accuracy
- Advanced semantic query with NVIDIA NeMo Retriever
- Fast reasoning with Llama Nemotron models
- Real-time web search powered by Tavily API
- Automated comprehensive research report generation

### NVIDIA Enterprise RAG Blueprint

The [NVIDIA Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) is a production-ready reference workflow that provides a complete foundation for building scalable, customizable pipelines for both retrieval and generation. Powered by NVIDIA NeMo Retriever models and NVIDIA Llama Nemotron models, the blueprint is optimized for high accuracy, strong reasoning, and enterprise-scale throughput.

**Key Features:**
- Multimodal PDF data extraction (text, tables, charts, infographics)
- Hybrid search with dense and sparse retrieval
- GPU-accelerated index creation and search
- Multi-turn conversations with query rewriting
- OpenAI-compatible APIs
- Built-in observability with Zipkin and Grafana

Together, these blueprints enable enterprises to build context-aware AI applications that ground decisions and generation in relevant enterprise data.

## Overview

This Terraform configuration deploys a complete EKS cluster with GPU support, optimized for running NVIDIA's Enterprise RAG Blueprint and AI-Q Research Assistant. The infrastructure includes:

- **EKS Cluster** with GPU-optimized configuration
- **Karpenter** for autoscaling GPU nodes (G5, P4, P5 instances)
- **OpenSearch Serverless** for vector search with Pod Identity authentication
- **GPU Operator** for NVIDIA driver management
- **EBS CSI Driver** for persistent storage
- **EKS Pod Identity Agent** for secure AWS IAM access

## Architecture

### AI-Q Research Assistant Architecture

The NVIDIA AI-Q Research Assistant Blueprint integrates multiple AI components to deliver an intelligent research assistant with RAG capabilities:

![AI-Q Architecture on AWS](imgs/aiq-aws.png)

### Enterprise RAG Blueprint Architecture

The RAG blueprint processes documents through multiple specialized NIM microservices with OpenSearch Serverless integration:

![RAG Pipeline with OpenSearch](imgs/rag-opensearch.png)

**Key Components:**
- **Llama-3.3-Nemotron-Super-49B-v1.5**: Advanced reasoning model for query processing and generation
- **NeMo Retriever Microservices**: Multi-modal document ingestion and processing
- **OpenSearch Serverless**: Vector database with Pod Identity authentication
- **NIM Microservices**: Optimized inference containers for embeddings, reranking, and vision models

### Network Architecture

The deployment creates a VPC with:
- Public subnets for load balancers and NAT gateways
- Private subnets for EKS nodes and pods

### Security Configuration

- **Pod Identity**: Kubernetes service accounts mapped to IAM roles for secure AWS access
- **Network Policies**: OpenSearch Serverless with configurable public/private access
- **Encryption**: AWS-managed keys for OpenSearch collections
- **IAM Roles**: Least-privilege access for EKS nodes and pods

### GPU Node Provisioning

Karpenter automatically provisions GPU nodes based on workload requirements:
- Nodes are created on-demand when pods are pending
- Automatic consolidation when nodes are underutilized
- Support for multi-GPU instances (1-8 GPUs per node)

## Prerequisites

> ⚠️ **Important - Cost Information**: This deployment uses GPU instances which can incur significant costs. Please review the [Cost Considerations](#cost-considerations) section at the end of this guide for detailed cost estimates before proceeding. **Always clean up resources when not in use.**

Before deploying the infrastructure, ensure you have:

- [Terraform](https://www.terraform.io/downloads.html) >= 1.3.2
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
- [Helm](https://helm.sh/docs/intro/install/)
- AWS account with appropriate permissions to create EKS clusters, VPCs, IAM roles, and OpenSearch collections
- Sufficient AWS service quotas for GPU instances (G5, P4, P5)

## Infrastructure Components

### EKS Cluster Configuration

The infrastructure creates an EKS cluster with:
- **Cluster Version**: Configurable (default: latest supported)
- **Region**: Configurable (default: us-west-2)
- **VPC**: Automatically provisioned with public and private subnets
- **Node Groups**: Managed by Karpenter for dynamic GPU provisioning

### Karpenter NodePools

Three pre-configured Karpenter NodePools for different GPU workloads:

1. **G5 NodePool** (g5-gpu-karpenter)
   - Instance types: g5.xlarge through g5.48xlarge
   - GPU: NVIDIA A10G (1-8 GPUs)
   - Use case: General-purpose inference and training

2. **P4 NodePool** (p4-gpu-karpenter) - Optional
   - Instance types: p4d.24xlarge, p4de.24xlarge
   - GPU: NVIDIA A100 (8 GPUs, 40GB or 80GB VRAM)
   - Use case: Large-scale training and inference

3. **P5 NodePool** (p5-gpu-karpenter) - Optional
   - Instance types: p5.48xlarge, p5e.48xlarge
   - GPU: NVIDIA H100 (8 GPUs)
   - Use case: Cutting-edge AI workloads

### OpenSearch Serverless

Automatically provisions:
- Vector search collection for RAG embeddings
- Encryption, network, and data access policies
- IAM role with EKS Pod Identity integration
- Kubernetes namespace and service account

## Configuration

### Basic Configuration

Edit `terraform/blueprint.tfvars` to customize your deployment:

```hcl
name          = "nvidia-deep-research"
region        = "us-west-2"              # Your AWS region

# Enable/disable P4 (A100) and P5 (H100) GPU instances
enable_p4_karpenter = true
enable_p5_karpenter = true

# OpenSearch Serverless Configuration
enable_opensearch_serverless    = true
opensearch_collection_name      = "osv-vector-dev"
opensearch_namespace            = "rag"
opensearch_service_account_name = "opensearch-access-sa"
```

### Advanced Configuration Options

The infrastructure inherits all configuration options from the base EKS blueprint. You can customize:

- **EKS Cluster Version**: Set `eks_cluster_version` (e.g., "1.33")
- **Node Group Configuration**: Modify Karpenter limits in `custom_karpenter.tf`
- **OpenSearch Settings**: Adjust collection type, policies, and access controls
- **EKS Addons**: Enable/disable addons by modifying the base configuration

For a complete list of available options, see [base/terraform/variables.tf](../base/terraform/variables.tf).

## Deployment

### Step 1: Deploy Infrastructure

Navigate to the infrastructure directory:

```bash
cd infra/nvidia-deep-research
```

Run the installation script:

```bash
./install.sh
```

This script will:
1. Copy the base Terraform configuration to `terraform/_LOCAL/`
2. Merge your custom configuration from `terraform/*.tf`
3. Initialize Terraform
4. Apply the Terraform configuration

⏱️ **Duration**: 15-20 minutes

> **✅ Infrastructure Ready**: Once the script completes successfully, your infrastructure is deployed and ready. Proceed to [Application Deployment](#application-deployment) below.

## Application Deployment

After infrastructure deployment is complete, deploy the NVIDIA Enterprise RAG Blueprint and the AI-Q Research Assistant (If Needed).

### Deployment Options

Choose based on your use case:

**Option 1: Enterprise RAG Blueprint Only**
- Deploy NVIDIA Enterprise RAG Blueprint with multi-modal document processing
- Includes NeMo Retriever microservices and OpenSearch integration
- Best for: Building custom RAG applications, document Q&A systems, knowledge bases

**Option 2: Full AI-Q Research Assistant**
- Includes everything from Option 1 plus AI-Q components
- Adds automated research report generation with web search capabilities
- Best for: Comprehensive research tasks, automated report generation, web-augmented research

### Prerequisites for Application Deployment

Before deploying applications, ensure you have:

**Tools Required:**
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) - Kubernetes command-line tool
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes package manager
- [Docker](https://docs.docker.com/get-docker/) - For building OpenSearch-enabled images
- **NGC API Key** - For accessing NVIDIA container registry. Sign up through:
  - **Option 1 (Quick Start):** [NVIDIA Developer Program](https://build.nvidia.com/) - Free account for POCs and development
  - **Option 2 (Production):** [NVIDIA AI Enterprise](https://aws.amazon.com/marketplace/pp/prodview-ozgjkov6vq3l6) - AWS Marketplace subscription for enterprise support
  - After signing up, generate your API key at: [NGC Personal Keys](https://org.ngc.nvidia.com/setup/personal-keys)
- [Tavily API Key](https://tavily.com/) - **Optional for AI-Q.** Enables web search capabilities. AI-Q can work in RAG-only mode without it. Not needed for Enterprise RAG only deployment

### Automated Deployment (Recommended)

Use the provided bash scripts to automate deployment:

> **💡 Tip**: For detailed manual deployment steps with full configuration control, see [Manual Deployment](#manual-deployment) below.

**1. Setup Environment**

Configures kubectl, collects API keys, verifies cluster, and patches Karpenter limits:

```bash
./deploy.sh setup
```

**2. Build OpenSearch Images**

Clones RAG source, integrates OpenSearch, builds images and pushes to Amazon Elastic Container Registry (ECR):

```bash
./deploy.sh build
```

⏱️ **Wait time**: 10-15 minutes for image builds

**3. Deploy Applications**

Use the deployment script with your target:

**1) Deploy Enterprise RAG Only**

For document Q&A without AI-Q research capabilities:

```bash
./deploy.sh rag
```

⏱️ **Wait time**: 15-25 minutes

---

**2) Deploy AI-Q Research Assistant**

AI-Q includes the Enterprise RAG Blueprint plus automated research report generation with optional web search capabilities.

**Option A: Deploy All at Once (Recommended - Faster)**

Deploy both RAG and AI-Q in parallel:

```bash
./deploy.sh all
```

⏱️ **Wait time**: 25-30 minutes

**Option B: Deploy Sequentially**

Deploy RAG first, then add AI-Q:

```bash
# Step 1: Deploy RAG
./deploy.sh rag

# Step 2: Deploy AI-Q
# AI-Q can work with or without web search (Tavily API is optional)
./deploy.sh aira
```

⏱️ **Wait time**: 15-25 minutes for RAG, then 20-30 minutes for AI-Q (35-55 minutes total)

---

### Manual Deployment

<details>
<summary><b>Click to expand manual deployment instructions</b></summary>

The following sections provide detailed manual deployment instructions for both Enterprise RAG and AI-Q components.

#### Step 1: Configure kubectl

Configure kubectl to access your EKS cluster:

```bash
# Set your cluster details
export CLUSTER_NAME="nvidia-deep-research"  # Or your cluster name
export REGION="us-west-2"                   # Or your region

# Configure kubectl
aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME

# Verify connection
kubectl get nodes
```

#### Step 2: Set Environment Variables

```bash
# Get AWS Account ID
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# OpenSearch Configuration (should match Terraform deployment)
export OPENSEARCH_SERVICE_ACCOUNT="opensearch-access-sa"
export OPENSEARCH_NAMESPACE="rag"
export COLLECTION_NAME="osv-vector-dev"

# Get OpenSearch endpoint from Terraform output
export OPENSEARCH_ENDPOINT=$(cd terraform/_LOCAL && terraform output -raw opensearch_collection_endpoint)

echo "OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"

# NGC API Key (required)
export NGC_API_KEY="<YOUR_NGC_API_KEY>"

# Tavily API Key for AI-Q (optional - enables web search)
export TAVILY_API_KEY="<YOUR_TAVILY_API_KEY>"  # Skip if deploying RAG only, or for AI-Q without web search
```

#### Step 3: Verify Cluster is Ready

Verify all components are healthy:

```bash
# Check cluster status
kubectl get nodes

# Verify Karpenter is running
kubectl get pods -n karpenter

# Check available Karpenter NodePools
kubectl get nodepools
```

> **Note**: GPU nodes will be automatically provisioned by Karpenter when you deploy GPU workloads.

#### Step 4: Configure Karpenter NodePool Limits

Increase the memory limit for the G5 GPU NodePool to accommodate multiple large models:

```bash
kubectl patch nodepool g5-gpu-karpenter --type='json' -p='[{"op": "replace", "path": "/spec/limits/memory", "value": "2000Gi"}]'
```

This increases the g5-gpu-karpenter NodePool's memory limit from 1000Gi to 2000Gi, allowing Karpenter to provision sufficient GPU nodes.

---

### Enterprise RAG Deployment

#### Step 5: Integrate OpenSearch and Build Docker Images

Clone the RAG source code and add OpenSearch implementation:

```bash
# Clone RAG source code
git clone -b v2.3.0 https://github.com/NVIDIA-AI-Blueprints/rag.git rag

# Download example OpenSearch implementation
COMMIT_HASH="47cd8b345e5049d49d8beb406372de84bd005abe"
curl -L https://github.com/NVIDIA/nim-deploy/archive/${COMMIT_HASH}.tar.gz | tar xz --strip=5 nim-deploy-${COMMIT_HASH}/cloud-service-providers/aws/blueprints/deep-research-blueprint-eks/opensearch
```

Integrate OpenSearch support into RAG source:

```bash
# Copy OpenSearch implementation into RAG source
cp -r opensearch/vdb/opensearch rag/src/nvidia_rag/utils/vdb/
cp opensearch/main.py rag/src/nvidia_rag/ingestor_server/main.py
cp opensearch/vdb/__init__.py rag/src/nvidia_rag/utils/vdb/__init__.py
cp opensearch/pyproject.toml rag/pyproject.toml
```

Build and push OpenSearch-enabled Docker images to ECR:

```bash
# Login to NGC registry
docker login nvcr.io  # username: $oauthtoken, password: NGC API Key

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build and push OpenSearch-enabled RAG images to ECR
./opensearch/build-opensearch-images.sh
```

⏱️ This will take 10-15 minutes to build and push images.

#### Step 6: Deploy Enterprise RAG Blueprint with OpenSearch

Deploy the Enterprise RAG Blueprint using the OpenSearch-enabled images:

```bash
# Set deployment variables
export ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
export IMAGE_TAG="2.3.0-opensearch"

# Deploy RAG with OpenSearch configuration
helm upgrade --install rag -n rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --create-namespace \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set serviceAccount.create=false \
  --set serviceAccount.name=$OPENSEARCH_SERVICE_ACCOUNT \
  --set image.repository="${ECR_REGISTRY}/nvidia-rag-server" \
  --set image.tag="${IMAGE_TAG}" \
  --set ingestor-server.image.repository="${ECR_REGISTRY}/nvidia-rag-ingestor" \
  --set ingestor-server.image.tag="${IMAGE_TAG}" \
  --set envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  -f helm/rag-values-os.yaml

# Patch ingestor-server deployment to use OpenSearch service account
kubectl patch deployment ingestor-server -n rag \
  -p "{\"spec\":{\"template\":{\"spec\":{\"serviceAccountName\":\"$OPENSEARCH_SERVICE_ACCOUNT\"}}}}"
```

⏱️ **Wait time**: 10-20 minutes for model downloads and GPU provisioning.

This deploys:
- **49B Nemotron Model** (8 GPUs) - Karpenter will provision g5.48xlarge
- **Embedding & Reranking Models** (1 GPU each) - Karpenter will provision g5.xlarge through g5.12xlarge
- **Data Ingestion Models** (1 GPU each) - Karpenter will provision g5.xlarge through g5.12xlarge
- **RAG Server** with OpenSearch Serverless integration
- **Frontend** for user interaction

#### Step 7: Verify RAG Deployment

Check that all RAG components are running:

```bash
# Check all pods in RAG namespace
kubectl get all -n rag

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=rag -n rag --timeout=600s

# Check specific components
kubectl get pods -n rag -o wide | grep -E "NAME|nim-llm|rag-server|ingestor|embedding|reranking"

# Verify service accounts are configured correctly
kubectl get pod -n rag -l app.kubernetes.io/component=rag-server -o jsonpath='{.items[0].spec.serviceAccountName}' | xargs -I {} echo "RAG Server service account: {}"
kubectl get pod -n rag -l app=ingestor-server -o jsonpath='{.items[0].spec.serviceAccountName}' | xargs -I {} echo "Ingestor Server service account: {}"
```

#### Step 8: Deploy DCGM ServiceMonitor for GPU Metrics

Enable Prometheus to scrape GPU metrics from DCGM Exporter:

```bash
# Deploy ServiceMonitor to connect RAG's Prometheus to infrastructure DCGM Exporter
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: rag
  labels:
    release: rag
spec:
  namespaceSelector:
    matchNames:
      - monitoring
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
EOF
```

This ServiceMonitor allows the Prometheus instance in the `rag` namespace to discover and scrape GPU metrics from the DCGM Exporter running in the `monitoring` namespace.

Deploy NVIDIA DCGM Grafana dashboard (optional but recommended):

```bash
# Download and deploy the official NVIDIA DCGM dashboard (with datasource fix)
curl -s https://grafana.com/api/dashboards/12239 | jq -r '.json' | \
    jq 'walk(if type == "object" and has("datasource") and (.datasource | type == "string") then .datasource = {"type": "prometheus", "uid": "prometheus"} else . end)' \
    > /tmp/dcgm-dashboard.json
kubectl create configmap nvidia-dcgm-exporter-dashboard \
    -n rag \
    --from-file=nvidia-dcgm-exporter.json=/tmp/dcgm-dashboard.json \
    --dry-run=client -o yaml | \
    kubectl label --local -f - grafana_dashboard=1 --dry-run=client -o yaml | \
    kubectl apply -f -
```

This dashboard will be automatically loaded by Grafana's sidecar and will display GPU utilization, temperature, memory usage, and other GPU metrics.

---

### AI-Q Components Deployment

> **📝 Deployment Choice**: Deploy these components if you need automated research report generation with web search capabilities. If your use case only requires the Enterprise RAG Blueprint for document Q&A, skip to [Next Steps](#next-steps).

#### Step 9: Deploy AIRA Components

Deploy the AI-Q Research Assistant:

```bash
# Verify TAVILY_API_KEY is set
echo "Tavily API Key: ${TAVILY_API_KEY:0:10}..."

# Deploy AIRA using NGC Helm chart
helm upgrade --install aira https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq-aira-v1.2.0.tgz \
  --username='$oauthtoken' \
  --password="${NGC_API_KEY}" \
  -n nv-aira --create-namespace \
  -f helm/aira-values.eks.yaml \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY" \
  --set tavilyApiSecret.password="$TAVILY_API_KEY"
```

⏱️ **Wait time**: 15-20 minutes for 70B model download.

This deploys:
- **AIRA Backend**: Research assistant functionality with automated report generation
- **70B Instruct Model**: For report generation (8 GPUs) - Karpenter will provision g5.48xlarge
- **Frontend**: User interface
- **Web Search**: Enabled if Tavily API key is provided; RAG-only mode if not

#### Step 10: Verify AIRA Deployment

Check that all AIRA components are running:

```bash
# Check all AIRA components
kubectl get all -n nv-aira

# Wait for all components to be ready
kubectl wait --for=condition=ready pod -l app=aira -n nv-aira --timeout=1200s

# Check pod distribution across nodes
kubectl get pods -n nv-aira -o wide
```

</details>

## Next Steps

Once application deployment is complete, proceed to the usage guide:

📖 **[Usage Guide](../../blueprints/inference/nvidia-deep-research/README.md)**

The usage guide covers:
- Accessing deployed services
- Data ingestion from S3 or UI
- Observability and monitoring

## Cleanup

### Uninstall Applications Only

To remove the RAG and AI-Q applications while keeping the infrastructure:

**Using Automation Script (Recommended):**

Navigate to the blueprints directory and run the cleanup script:

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

```bash
./app.sh cleanup
```

The cleanup script will:
- Stop all port-forwarding processes
- Uninstall AIRA and RAG Helm releases
- Remove local port-forward PID files

**Manual Application Cleanup:**

```bash
# Navigate to blueprints directory
cd ../../blueprints/inference/nvidia-deep-research

# Stop port-forwards
./app.sh port stop all

# Uninstall AIRA (if deployed)
helm uninstall aira -n nv-aira

# Uninstall RAG
helm uninstall rag -n rag
```

**(Optional) Clean up temporary files created during deployment:**

```bash
rm /tmp/.port-forward-*.pid
```

> **Note**: This only removes the applications. The EKS cluster and infrastructure will remain running. GPU nodes will be terminated by Karpenter within 5-10 minutes.

### Clean Up Infrastructure

To remove the entire EKS cluster and all infrastructure components:

```bash
# Navigate to infra directory (if not already there)
cd infra/nvidia-deep-research

# Run cleanup script
./cleanup.sh
```

> **Warning**: This will permanently delete:
> - EKS cluster and all workloads
> - OpenSearch Serverless collection and data
> - VPC and networking resources
> - All associated AWS resources
>
> Backup important data before proceeding.

### Manual Cleanup (if automated cleanup fails)

If the cleanup script encounters issues:

```bash
cd terraform/_LOCAL

# Destroy infrastructure
terraform destroy -auto-approve

# If specific resources fail, manually delete them from AWS Console:
# - OpenSearch Serverless collections
# - EKS cluster and node groups
# - VPC and networking resources
```

## Cost Considerations

<details>
<summary><h3>Estimated Costs for This Deployment</h3></summary>

> ⚠️ **Important**: GPU instances and supporting infrastructure can incur significant costs if left running. **Always clean up resources when not in use** to avoid unexpected charges.

### Estimated Monthly Costs

The following table shows approximate costs for the **default deployment** in US West 2 (Oregon) region. Actual costs will vary based on region, usage patterns, and workload duration.

| Resource | Configuration | Estimated Monthly Cost | Notes |
|----------|--------------|----------------------|-------|
| **EKS Control Plane** | 1 cluster | **~$73/month** | Fixed cost: $0.10/hour × 730 hours |
| **GPU Instances (RAG Only)** | 1x g5.48xlarge (8x A10G)<br/>2x g5.12xlarge (4x A10G each) | **~$20,171/month*** | Only when workloads are running<br/>Karpenter scales down when idle |
| **GPU Instances (RAG + AI-Q)** | Additional g5.48xlarge | **~$32,061/month*** | Additional 70B model requires 8 more GPUs |
| **OpenSearch Serverless** | 2-4 OCUs (typical) | **~$350-700/month** | $0.24/OCU-hour<br/>Scales based on data volume and queries |
| **NAT Gateway** | 2 AZs | **~$66/month** | Fixed: 2 gateways × $0.045/hour × 730 hours<br/>Plus data processing: $0.045/GB |
| **ECR Storage** | Docker images | **~$5-10/month** | 50-100GB of custom images<br/>ECR pricing: $0.10/GB/month |
| **EBS Volumes** | Node storage | **~$72/month** | 300GB gp3 per node × 3 nodes × $0.08/GB<br/>Only charged when GPU nodes running |
| **Data Transfer** | Cross-AZ, Internet | **Variable** | Depends on usage patterns<br/>Cross-AZ: $0.01/GB, Internet: $0.09/GB |

**\*GPU Instance Costs assume continuous operation. See breakdown below.**

### GPU Instance Cost Breakdown

GPU instances are the **primary cost driver**. Costs depend on instance type and how long they run:

**Default Configuration (G5 Instances - RAG Only):**

| Instance Type | GPUs | On-Demand Rate | Daily Cost (24hr) | Monthly Cost (730hr) |
|---------------|------|----------------|-------------------|---------------------|
| g5.48xlarge (×1) | 8x A10G | $16.288/hr | $390.91 | $11,890.24 |
| g5.12xlarge (×2) | 4x A10G each | $5.672/hr each | $136.13 each | $4,140.56 each |

**Total for RAG**: ~$20,171/month if running 24/7 (1× g5.48xlarge + 2× g5.12xlarge = $11,890 + $8,281)

**With AI-Q (Additional 70B Model):**
- Additional g5.48xlarge: $11,890.24/month
- **Total**: ~$32,061/month if running 24/7 (2× g5.48xlarge + 2× g5.12xlarge)

> **Note**: If using alternative instance types (G6e, P4, P5), costs will vary. Check [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) for your region and instance type.

</details>

## Additional Resources

**Infrastructure Documentation:**
- [EKS Blueprints](https://aws-ia.github.io/terraform-aws-eks-blueprints/)
- [Karpenter Documentation](https://karpenter.sh/)
- [OpenSearch Serverless Documentation](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [EKS Pod Identity Documentation](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html)

**AWS Services:**
- [Amazon EKS Documentation](https://docs.aws.amazon.com/eks/)
- [EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [AWS Service Quotas](https://console.aws.amazon.com/servicequotas/)
