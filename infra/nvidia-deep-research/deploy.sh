#!/bin/bash

#---------------------------------------------------------------
# Deploy NVIDIA Enterprise RAG and AI-Q Research Assistant
#---------------------------------------------------------------

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

show_help() {
    echo "Usage: ./deploy.sh <target>"
    echo ""
    echo "Targets:"
    echo "  setup   Setup environment and configure kubectl"
    echo "  build   Build OpenSearch-enabled Docker images"
    echo "  rag     Deploy Enterprise RAG Blueprint only"
    echo "  aira    Deploy AI-Q Research Assistant only"
    echo "  all     Deploy both RAG and AI-Q in parallel"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh setup   # Configure environment (run first)"
    echo "  ./deploy.sh build   # Build custom images (after setup)"
    echo "  ./deploy.sh rag     # Deploy RAG for document Q&A"
    echo "  ./deploy.sh all     # Deploy full stack (RAG + AI-Q) in parallel"
    echo ""
    echo "Complete workflow:"
    echo "  1. ./deploy.sh setup    # Setup environment (~2 minutes)"
    echo "  2. ./deploy.sh build    # Build images (~10-15 minutes)"
    echo "  3. ./deploy.sh all      # Deploy applications (~25-30 minutes)"
    echo ""
    echo "Wait times:"
    echo "  setup: ~2 minutes"
    echo "  build: 10-15 minutes"
    echo "  rag:   15-25 minutes"
    echo "  aira:  25-30 minutes"
    echo "  all:   25-30 minutes (parallel deployment)"
}

setup_environment() {
    # Get cluster details from Terraform outputs
    print_info "Retrieving cluster details from Terraform..."
    if [ ! -d "terraform/_LOCAL" ]; then
        print_error "Terraform directory not found. Please run ./install.sh first to deploy infrastructure."
        exit 1
    fi

    cd terraform/_LOCAL

    # Get kubectl configuration command from Terraform
    KUBECTL_CMD=$(terraform output -raw configure_kubectl 2>/dev/null)
    OPENSEARCH_ENDPOINT=$(terraform output -raw opensearch_collection_endpoint 2>/dev/null)

    cd - > /dev/null

    if [ -z "$KUBECTL_CMD" ]; then
        print_error "Failed to retrieve cluster details from Terraform. Ensure infrastructure is deployed successfully."
        exit 1
    fi

    # Extract region and cluster name from kubectl command for .env file
    REGION=$(echo "$KUBECTL_CMD" | sed -n 's/.*--region \([^ ]*\).*/\1/p')
    CLUSTER_NAME=$(echo "$KUBECTL_CMD" | sed -n 's/.*--name \([^ ]*\).*/\1/p')

    # Configure kubectl using Terraform's command
    print_info "Configuring kubectl..."
    eval "$KUBECTL_CMD"

    if kubectl get nodes &>/dev/null; then
        print_success "Connected to cluster"
    else
        print_error "Failed to connect to cluster"
        exit 1
    fi

    # Set environment variables
    print_info "Getting AWS Account ID..."
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

    OPENSEARCH_SERVICE_ACCOUNT="opensearch-access-sa"
    OPENSEARCH_NAMESPACE="rag"
    COLLECTION_NAME="osv-vector-dev"

    if [ -z "$OPENSEARCH_ENDPOINT" ]; then
        print_error "Failed to retrieve OpenSearch endpoint from Terraform. Ensure infrastructure is deployed."
        exit 1
    fi
    print_success "OpenSearch endpoint retrieved"

    # Collect API Keys
    if [ -z "$NGC_API_KEY" ]; then
        read -sp "Enter NGC API Key: " NGC_API_KEY
        echo ""
    fi

    if [ -z "$NGC_API_KEY" ]; then
        print_error "NGC API Key is required. Get your key from: https://org.ngc.nvidia.com/setup/personal-keys"
        exit 1
    fi

    read -p "Deploy AI-Q? (y/n) [n]: " DEPLOY_AIRA
    DEPLOY_AIRA=${DEPLOY_AIRA:-n}

    if [[ "$DEPLOY_AIRA" =~ ^[Yy] ]]; then
        if [ -z "$TAVILY_API_KEY" ]; then
            echo ""
            print_info "Tavily API Key (optional): Enables web search in AI-Q. Leave blank to use RAG-only mode."
            read -sp "Enter Tavily API Key (or press Enter to skip): " TAVILY_API_KEY
            echo ""
        fi

        if [ -z "$TAVILY_API_KEY" ]; then
            print_info "AI-Q will deploy without web search capabilities (RAG-only mode)"
        fi
    fi

    # Verify cluster
    print_info "Verifying cluster..."
    if ! kubectl get pods -n karpenter | grep -q "Running"; then
        print_error "Karpenter not running"
        exit 1
    fi
    print_success "Karpenter running"

    # Patch Karpenter NodePool
    if kubectl get nodepool g5-gpu-karpenter &>/dev/null; then
        print_info "Patching g5-gpu-karpenter memory limit to 2000Gi..."
        kubectl patch nodepool g5-gpu-karpenter --type='json' \
            -p='[{"op": "replace", "path": "/spec/limits/memory", "value": "2000Gi"}]'
        print_success "NodePool patched"
    fi

    # Save configuration
    cat > .env << EOF
export CLUSTER_NAME="$CLUSTER_NAME"
export REGION="$REGION"
export ACCOUNT_ID="$ACCOUNT_ID"
export OPENSEARCH_SERVICE_ACCOUNT="$OPENSEARCH_SERVICE_ACCOUNT"
export OPENSEARCH_NAMESPACE="$OPENSEARCH_NAMESPACE"
export COLLECTION_NAME="$COLLECTION_NAME"
export OPENSEARCH_ENDPOINT="$OPENSEARCH_ENDPOINT"
export ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
export IMAGE_TAG="2.3.0-opensearch"
export NGC_API_KEY="$NGC_API_KEY"
EOF

    if [ -n "$TAVILY_API_KEY" ]; then
        echo "export TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> .env
    fi

    print_success "Configuration saved to .env"
}

build_images() {
    # Verify required variables
    for var in ACCOUNT_ID REGION NGC_API_KEY; do
        if [ -z "${!var}" ]; then
            print_error "Missing $var. Run ./deploy.sh setup first"
            exit 1
        fi
    done

    # Check Docker
    if ! docker info &>/dev/null; then
        print_error "Docker not running"
        exit 1
    fi

    # Clone RAG source
    if [ ! -d "rag" ]; then
        print_info "Cloning RAG v2.3.0..."
        git clone -b v2.3.0 https://github.com/NVIDIA-AI-Blueprints/rag.git rag
        print_success "Cloned"
    fi

    # Download OpenSearch implementation
    if [ ! -d "opensearch" ]; then
        print_info "Downloading OpenSearch implementation..."
        COMMIT_HASH="47cd8b345e5049d49d8beb406372de84bd005abe"
        curl -sL "https://github.com/NVIDIA/nim-deploy/archive/${COMMIT_HASH}.tar.gz" | \
            tar xz --strip=5 "nim-deploy-${COMMIT_HASH}/cloud-service-providers/aws/blueprints/deep-research-blueprint-eks/opensearch"
        print_success "Downloaded"
    fi

    # Integrate OpenSearch
    print_info "Integrating OpenSearch into RAG source..."
    cp -r opensearch/vdb/opensearch rag/src/nvidia_rag/utils/vdb/
    cp opensearch/main.py rag/src/nvidia_rag/ingestor_server/main.py
    cp opensearch/vdb/__init__.py rag/src/nvidia_rag/utils/vdb/__init__.py
    cp opensearch/pyproject.toml rag/pyproject.toml
    print_success "Integrated"

    # Login to registries
    print_info "Logging into NGC..."
    echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin > /dev/null
    print_success "Logged into NGC"

    print_info "Logging into ECR..."
    aws ecr get-login-password --region "${REGION}" | \
        docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" > /dev/null
    print_success "Logged into ECR"

    # Build and push images
    print_info "Building images (this may take 15-30 minutes)..."
    bash opensearch/build-opensearch-images.sh
    print_success "Images built and pushed to ECR"
}

deploy_rag() {
    print_info "Deploying RAG Blueprint..."

    # Verify required variables
    for var in ACCOUNT_ID REGION NGC_API_KEY OPENSEARCH_ENDPOINT OPENSEARCH_SERVICE_ACCOUNT; do
        if [ -z "${!var}" ]; then
            print_error "Missing $var. Run ./deploy.sh setup"
            exit 1
        fi
    done

    # Deploy RAG with Helm
    helm upgrade --install rag -n "$OPENSEARCH_NAMESPACE" \
      https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
      --username '$oauthtoken' \
      --password "${NGC_API_KEY}" \
      --create-namespace \
      --set imagePullSecret.password="$NGC_API_KEY" \
      --set ngcApiSecret.password="$NGC_API_KEY" \
      --set serviceAccount.create=false \
      --set serviceAccount.name="$OPENSEARCH_SERVICE_ACCOUNT" \
      --set image.repository="${ECR_REGISTRY}/nvidia-rag-server" \
      --set image.tag="${IMAGE_TAG}" \
      --set ingestor-server.image.repository="${ECR_REGISTRY}/nvidia-rag-ingestor" \
      --set ingestor-server.image.tag="${IMAGE_TAG}" \
      --set envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
      --set envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
      --set ingestor-server.envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
      --set ingestor-server.envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
      -f helm/rag-values-os.yaml

    print_success "Deployment initiated"

    # Patch ingestor-server
    print_info "Patching ingestor-server service account..."
    kubectl patch deployment ingestor-server -n "$OPENSEARCH_NAMESPACE" \
      -p "{\"spec\":{\"template\":{\"spec\":{\"serviceAccountName\":\"$OPENSEARCH_SERVICE_ACCOUNT\"}}}}"
    print_success "Patched"

    # Wait for pods
    print_info "Waiting for RAG pods to be ready (may take 15-25 minutes for model downloads)..."
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance=rag \
        -n "$OPENSEARCH_NAMESPACE" \
        --timeout=1800s

    # Deploy DCGM ServiceMonitor for GPU metrics
    print_info "Deploying DCGM ServiceMonitor for GPU metrics..."
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
    print_success "DCGM ServiceMonitor deployed"

    # Deploy DCGM Grafana Dashboard
    print_info "Deploying NVIDIA DCGM Grafana dashboard..."
    curl -s https://grafana.com/api/dashboards/12239 | jq -r '.json' | \
        jq 'walk(if type == "object" and has("datasource") and (.datasource | type == "string") then .datasource = {"type": "prometheus", "uid": "prometheus"} else . end)' \
        > /tmp/dcgm-dashboard.json
    kubectl create configmap nvidia-dcgm-exporter-dashboard \
        -n rag \
        --from-file=nvidia-dcgm-exporter.json=/tmp/dcgm-dashboard.json \
        --dry-run=client -o yaml | \
        kubectl label --local -f - grafana_dashboard=1 --dry-run=client -o yaml | \
        kubectl apply -f - >/dev/null

    # Clean up temporary file
    if [ -f /tmp/dcgm-dashboard.json ]; then
        rm /tmp/dcgm-dashboard.json
    fi
    print_success "DCGM dashboard deployed"

    print_success "RAG deployed successfully"
}

deploy_aira() {
    print_info "Deploying AI-Q..."

    # Verify required variables
    if [ -z "$NGC_API_KEY" ]; then
        print_error "Missing NGC_API_KEY"
        exit 1
    fi

    # Tavily API key is optional - AI-Q can work in RAG-only mode without it
    if [ -z "$TAVILY_API_KEY" ]; then
        print_info "Tavily API key not provided. AI-Q will operate in RAG-only mode (no web search)."
        TAVILY_API_KEY="not-provided"
    fi

    # Deploy AIRA
    helm upgrade --install aira \
      https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq-aira-v1.2.0.tgz \
      --username='$oauthtoken' \
      --password="${NGC_API_KEY}" \
      -n nv-aira \
      --create-namespace \
      -f helm/aira-values.eks.yaml \
      --set imagePullSecret.password="$NGC_API_KEY" \
      --set ngcApiSecret.password="$NGC_API_KEY" \
      --set tavilyApiSecret.password="$TAVILY_API_KEY"

    print_success "Deployment initiated"

    # Wait for pods
    print_info "Waiting for AI-Q pods to be ready (may take 25-30 minutes for 70B model download)..."
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance=aira \
        -n nv-aira \
        --timeout=2400s

    print_success "AI-Q deployed successfully"
}

deploy_all() {
    print_info "Starting parallel deployment of RAG and AI-Q..."
    echo ""

    # Start RAG deployment in background
    print_info "Launching RAG deployment..."
    (
        deploy_rag > /tmp/deploy-rag.log 2>&1
        echo $? > /tmp/deploy-rag.exit
    ) &
    RAG_PID=$!

    # Start AI-Q deployment in background
    print_info "Launching AI-Q deployment..."
    (
        deploy_aira > /tmp/deploy-aira.log 2>&1
        echo $? > /tmp/deploy-aira.exit
    ) &
    AIRA_PID=$!

    echo ""
    print_info "Both deployments running in parallel..."
    print_info "  - RAG deployment: PID $RAG_PID (logs: /tmp/deploy-rag.log)"
    print_info "  - AI-Q deployment: PID $AIRA_PID (logs: /tmp/deploy-aira.log)"
    echo ""
    print_info "This will take approximately 25-30 minutes. You can monitor progress in separate terminals:"
    echo "  tail -f /tmp/deploy-rag.log"
    echo "  tail -f /tmp/deploy-aira.log"
    echo ""

    # Wait for both to complete
    wait $RAG_PID
    RAG_EXIT=$(cat /tmp/deploy-rag.exit 2>/dev/null || echo 1)

    wait $AIRA_PID
    AIRA_EXIT=$(cat /tmp/deploy-aira.exit 2>/dev/null || echo 1)

    echo ""
    echo "=================================================="

    # Check results
    if [ "$RAG_EXIT" -eq 0 ] && [ "$AIRA_EXIT" -eq 0 ]; then
        print_success "All deployments completed successfully!"

        # Cleanup temp files
        for file in /tmp/deploy-rag.log /tmp/deploy-rag.exit /tmp/deploy-aira.log /tmp/deploy-aira.exit; do
            if [ -f "$file" ]; then
                rm "$file"
            fi
        done
        exit 0
    elif [ "$RAG_EXIT" -eq 0 ]; then
        print_warning "RAG deployed successfully, but AI-Q failed"
        echo ""
        print_error "AI-Q deployment failed. Check logs:"
        echo "  cat /tmp/deploy-aira.log"
        echo ""
        print_info "You can retry AI-Q deployment separately:"
        echo "  ./deploy.sh aira"
        exit 1
    elif [ "$AIRA_EXIT" -eq 0 ]; then
        print_warning "AI-Q deployed successfully, but RAG failed"
        echo ""
        print_error "RAG deployment failed. Check logs:"
        echo "  cat /tmp/deploy-rag.log"
        echo ""
        print_info "You can retry RAG deployment separately:"
        echo "  ./deploy.sh rag"
        exit 1
    else
        print_error "Both deployments failed"
        echo ""
        echo "Check logs for details:"
        echo "  RAG: cat /tmp/deploy-rag.log"
        echo "  AI-Q: cat /tmp/deploy-aira.log"
        exit 1
    fi
}

# Main script logic
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [ -z "$1" ]; then
    show_help
    exit 0
fi

# Parse target
TARGET=$1

# Setup and build don't require .env to exist
case "$TARGET" in
    setup)
        setup_environment
        exit 0
        ;;
    build)
        # Build needs .env for some variables
        if [ -f .env ]; then
            source .env
        else
            print_error ".env not found. Run ./deploy.sh setup first"
            exit 1
        fi
        build_images
        exit 0
        ;;
esac

# For deployment targets, load environment
if [ -f .env ]; then
    source .env
else
    print_error ".env not found. Run ./deploy.sh setup first"
    exit 1
fi

case "$TARGET" in
    rag)
        deploy_rag
        ;;
    aira)
        deploy_aira
        ;;
    all)
        deploy_all
        ;;
    *)
        print_error "Invalid target: $TARGET"
        echo ""
        show_help
        exit 1
        ;;
esac
