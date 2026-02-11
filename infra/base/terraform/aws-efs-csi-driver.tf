resource "aws_eks_addon" "aws_efs_csi_driver" {
  count                       = var.enable_aws_efs_csi_driver ? 1 : 0
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-efs-csi-driver"
  resolve_conflicts_on_update = "PRESERVE"
  depends_on                  = [module.aws_efs_csi_pod_identity]
  tags                        = local.tags
}

module "aws_efs_csi_pod_identity" {
  count   = var.enable_aws_efs_csi_driver ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"
  name    = "aws-efs-csi"

  attach_aws_efs_csi_policy = true

  associations = {
    efs_csi = {
      cluster_name    = module.eks.cluster_name
      namespace       = "kube-system"
      service_account = "efs-csi-controller-sa"
    }
  }
  tags = local.tags
}
