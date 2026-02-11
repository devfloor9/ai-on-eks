# NVIDIA Enterprise RAG & AI-Q Research Assistant - Usage Guide

This guide covers how to use the NVIDIA Enterprise RAG Blueprint and AI-Q Research Assistant after deployment on Amazon EKS.

## What Are These Applications?

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

## Table of Contents

- [Prerequisites](#prerequisites)
- [Access Services](#access-services)
- [Data Ingestion](#data-ingestion)
- [Observability](#observability)
- [Cleanup](#cleanup)
- [Additional Resources](#additional-resources)

## Prerequisites

> ⚠️ **Important - Cost Information**: This deployment runs on GPU instances which can incur significant costs. See the [Cost Considerations section](../../../infra/nvidia-deep-research/README.md#cost-considerations) in the deployment guide for detailed cost estimates. **Always clean up resources when not in use.**

Before using these applications, ensure the infrastructure and applications are deployed:

📖 **[Infrastructure & Application Deployment Guide](../../../infra/nvidia-deep-research/README.md)**

**Required Components:**
- ✅ **EKS Cluster** - Deployed with Karpenter autoscaling
- ✅ **OpenSearch Serverless** - Collection with Pod Identity configured
- ✅ **Enterprise RAG Blueprint** - Deployed with NeMo Retriever microservices
- ✅ **AI-Q Research Assistant** - (Optional) Deployed if you need research report generation

**Tools Required:**
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) - Configured to access your EKS cluster
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) - For S3 data ingestion

## Access Services

Once deployment is complete, you can access the services locally using port-forwarding.

### Start Port Forwarding

**For RAG Services:**

```bash
./app.sh port start rag
```

This enables access to:
- **RAG Frontend**: http://localhost:3001 - Test RAG Q&A directly
- **Ingestor API**: http://localhost:8082/docs - API docs

**For AI-Q Services** (if deployed):

```bash
./app.sh port start aira
```

This enables access to:
- **AIRA Research Assistant**: http://localhost:3000 - Generate comprehensive research reports with web search

### Managing Port Forwarding

Check port forwarding status:
```bash
./app.sh port status
```

Stop port forwarding:
```bash
./app.sh port stop rag      # Stop RAG services
./app.sh port stop aira     # Stop AI-Q services
./app.sh port stop all      # Stop all services
```

### Using the Applications

**RAG Frontend (http://localhost:3001):**
- Upload documents directly through the UI
- Ask questions about your ingested documents
- Test multi-turn conversations
- View citations and sources

**AI-Q Research Assistant (http://localhost:3000):**
- Define research topics and questions
- Leverage both uploaded documents and web search
- Generate comprehensive research reports automatically
- Export reports in various formats

**Ingestor API (http://localhost:8082/docs):**
- Programmatic document ingestion
- Batch upload capabilities
- Collection management
- View OpenAPI documentation

## Data Ingestion

After deploying RAG (and optionally AI-Q), you can ingest documents into the OpenSearch vector database.

### Supported File Types

The RAG pipeline supports multi-modal document ingestion including:
- PDF documents
- Text files (.txt, .md)
- Images (.jpg, .png)
- Office documents (.docx, .pptx)
- HTML files

The NeMo Retriever microservices will automatically extract text, tables, charts, and images from these documents.

### Ingestion Methods

You have two options for ingesting documents:

#### Method 1: UI Upload (Testing/Small Datasets)

Upload individual documents directly through the frontend interfaces:

1. **RAG Frontend** (http://localhost:3001) - Ideal for testing individual documents
2. **AIRA Frontend** (http://localhost:3000) - Upload documents for research tasks

This method is perfect for:
- Testing the RAG pipeline
- Small document collections (< 100 documents)
- Quick experimentation
- Ad-hoc document uploads

#### Method 2: S3 Batch Ingestion (Production/Large Datasets)

Use the data ingestion script to batch process documents from an S3 bucket. Recommended for:
- Production deployments
- Large document collections (hundreds to thousands of documents)
- Automated ingestion workflows
- Scheduled data updates

**Steps:**

1. Ensure the RAG port-forward is running:
   ```bash
   ./app.sh port start rag
   ```

2. Run the data ingestion script (it will prompt for S3 bucket details):
   ```bash
   ./app.sh ingest
   ```

3. Or set environment variables to skip prompts:
   ```bash
   export S3_BUCKET_NAME="your-pdf-bucket-name"
   export S3_PREFIX="documents/"  # Optional folder path
   ./app.sh ingest
   ```

The script will:
- Download documents from your S3 bucket
- Process them through the NeMo Retriever pipeline
- Store embeddings in OpenSearch Serverless
- Display ingestion progress and statistics

> **Additional Resources**:
> - [RAG batch_ingestion.py documentation](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.3.0/scripts)
> - [AI-Q bulk data ingestion documentation](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant/blob/main/data/readme.md#bulk-upload-via-python)

### Verifying Ingestion

After ingestion, verify your documents are available:

1. **Via RAG Frontend**: Navigate to http://localhost:3001 and ask a question about your documents
2. **Via Ingestor API**: Check http://localhost:8082/docs for collection statistics
3. **Via OpenSearch**: Query the OpenSearch collection directly using the AWS Console

## Observability

The RAG and AI-Q deployments include built-in observability tools for monitoring performance, tracing requests, and viewing metrics.

### Access Monitoring Services

**Automated Approach (Recommended):**

Start port-forwarding for all observability services:

```bash
./app.sh port start observability
```

This automatically port-forwards:
- **Zipkin**: http://localhost:9411 - RAG distributed tracing
- **Grafana**: http://localhost:8080 - RAG metrics and dashboards
- **Phoenix**: http://localhost:6006 - AI-Q workflow tracing (if deployed)

Check status:
```bash
./app.sh port status
```

Stop observability port-forwards:
```bash
./app.sh port stop observability
```

**Manual Approach:**

<details>
<summary>Manual kubectl port-forward commands</summary>

**RAG Observability (Zipkin & Grafana):**

```bash
# Port-forward Zipkin for distributed tracing (run in a separate terminal)
kubectl port-forward -n rag svc/rag-zipkin 9411:9411

# Port-forward Grafana for metrics and dashboards (run in another separate terminal)
kubectl port-forward -n rag svc/rag-grafana 8080:80
```

**AI-Q Observability (Phoenix):**

```bash
# Port-forward Phoenix for AI-Q tracing (run in a separate terminal)
kubectl port-forward -n nv-aira svc/aira-phoenix 6006:6006
```

</details>

### Access Monitoring UIs

Once port-forwarding is active:

- **Zipkin UI** (RAG tracing): http://localhost:9411
  - View end-to-end request traces
  - Analyze latency bottlenecks
  - Debug multi-service interactions

- **Grafana UI** (RAG metrics): http://localhost:8080
  - Default credentials: admin/admin
  - Pre-built dashboards for RAG metrics
  - GPU utilization and throughput monitoring

- **Phoenix UI** (AI-Q tracing): http://localhost:6006
  - Agent workflow visualization
  - LLM call tracing
  - Research report generation analysis

> **Note**: For detailed information on using these observability tools, refer to:
> - [Viewing Traces in Zipkin](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-traces-in-zipkin)
> - [Viewing Metrics in Grafana Dashboard](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-metrics-in-grafana)

> **Alternative**: If you need to expose monitoring services publicly, you can create an Ingress resource with appropriate authentication and security controls.

## Cleanup

To uninstall the applications or clean up the entire infrastructure, refer to the [Cleanup section in the Infrastructure README](../../infra/nvidia-deep-research/README.md#cleanup).

This includes instructions for:
- Uninstalling only the RAG and AI-Q applications while keeping the infrastructure
- Complete infrastructure teardown including EKS cluster and all AWS resources

## Additional Resources

**NVIDIA Blueprints:**
- [NVIDIA Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline)
- [NVIDIA Enterprise RAG Blueprint GitHub](https://github.com/NVIDIA-AI-Blueprints/rag)
- [NVIDIA AI-Q Research Assistant](https://build.nvidia.com/nvidia/aiq)
- [NVIDIA AI-Q Research Assistant GitHub](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)

**AWS Services:**
- [OpenSearch Serverless Documentation](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html)
- [EKS Best Practices Guide](https://aws.github.io/aws-eks-best-practices/)
- [Amazon EKS Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html)

**Troubleshooting:**
- [RAG Troubleshooting Guide](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/troubleshooting.md)
- [AI-Q GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant/issues)
- [EKS Troubleshooting](https://docs.aws.amazon.com/eks/latest/userguide/troubleshooting.html)
