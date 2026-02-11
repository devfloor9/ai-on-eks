name                             = "slurm-on-eks"
enable_aws_fsx_csi_driver        = true
deploy_fsx_volume                = true
fsx_pvc_namespace                = "slurm"
enable_aws_efa_k8s_device_plugin = true
enable_cert_manager              = true
enable_mariadb_operator          = true
enable_kube_prometheus_stack     = true
enable_slurm_operator            = true
enable_service_mutator_webhook   = true
enable_nvidia_gpu_operator       = true
# region                         = "us-west-2"
# eks_cluster_version            = "1.34"

# -------------------------------------------------------------------------------------
# EKS Addons Configuration
#
# These are the EKS Cluster Addons managed by Terraform stack.
# You can enable or disable any addon by setting the value to `true` or `false`.
#
# If you need to add a new addon that isn't listed here:
# 1. Add the addon name to the `enable_cluster_addons` variable in `base/terraform/variables.tf`
# 2. Update the `locals.cluster_addons` logic in `eks.tf` to include any required configuration
#
# -------------------------------------------------------------------------------------

# enable_cluster_addons = {
#   coredns                         = true
#   kube-proxy                      = true
#   vpc-cni                         = true
#   eks-pod-identity-agent          = true
#   aws-ebs-csi-driver              = true
#   metrics-server                  = true
#   eks-node-monitoring-agent       = true
#   amazon-cloudwatch-observability = true
# }
