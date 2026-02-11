name = "nvidia-deep-research"
# region              = "us-west-2"
# eks_cluster_version = "1.33"

# -------------------------------------------------------------------------------------
# EKS Addons Configuration
#
# These are the EKS Cluster Addons managed by Terrafrom stack.
# You can enable or disable any addon by setting the value to `true` or `false`.
#
# If you need to add a new addon that isn't listed here:
# 1. Add the addon name to the `enable_cluster_addons` variable in `base/terraform/variables.tf`
# 2. Update the `locals.cluster_addons` logic in `eks.tf` to include any required configuration
#
# -------------------------------------------------------------------------------------
enable_cluster_addons = {
  amazon-cloudwatch-observability = false
}

# -------------------------------------------------------------------------------------
# Karpenter Custom NodePools Configuration
#
# Enable/disable P4 (A100) and P5 (H100) GPU instance support
# P4 families: p4d, p4de (8x A100 - 40GB or 80GB VRAM)
# P5 families: p5, p5e, p5en (1-8x H100)
# -------------------------------------------------------------------------------------
enable_p4_karpenter = true # p4d/p4de.24xlarge (8x A100)
enable_p5_karpenter = true # p5*.{4,48}xlarge (1-8x H100)

# -------------------------------------------------------------------------------------
# OpenSearch Serverless Configuration
#
# Set enable_opensearch_serverless = true to create everything:
# 1. OpenSearch Serverless collection (vector search)
# 2. Encryption, network, and data access policies
# 3. IAM role and policy (EKS Pod Identity)
# 4. Kubernetes namespace and service account
#
# Based on the manual setup guide in the README
# -------------------------------------------------------------------------------------
enable_opensearch_serverless    = true
opensearch_collection_name      = "osv-vector-dev"
opensearch_collection_type      = "VECTORSEARCH"
opensearch_policy_name          = "osv-vector-dev-policy"
opensearch_allow_public_access  = true
opensearch_namespace            = "rag" # Namespace where your pods will run
opensearch_service_account_name = "opensearch-access-sa"
