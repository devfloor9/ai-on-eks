name                             = "envoy-gateway-cluster"
enable_ai_ml_observability_stack = true
availability_zones_count         = 4
enable_soci_snapshotter          = true
enable_redis                     = true
enable_envoy_gateway             = true
enable_envoy_ai_gateway_crds     = true
enable_envoy_ai_gateway          = true
# region                           = "us-west-2"
# eks_cluster_version              = "1.33"

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
  metrics-server                  = false
  eks-node-monitoring-agent       = false
  amazon-cloudwatch-observability = false
}
