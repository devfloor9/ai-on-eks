resource "aws_eks_addon" "aws_fsx_csi_driver" {
  count                       = var.enable_aws_fsx_csi_driver ? 1 : 0
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-fsx-csi-driver"
  resolve_conflicts_on_update = "PRESERVE"
  depends_on                  = [module.aws_fsx_csi_pod_identity]
  tags                        = local.tags
}

module "aws_fsx_csi_pod_identity" {
  count   = var.enable_aws_fsx_csi_driver ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"
  name    = "aws-fsx-csi"

  attach_aws_fsx_lustre_csi_policy     = true
  aws_fsx_lustre_csi_service_role_arns = ["arn:aws:iam::*:role/aws-service-role/s3.data-source.lustre.fsx.amazonaws.com/*"]

  associations = {
    fsx_csi = {
      cluster_name    = module.eks.cluster_name
      namespace       = "kube-system"
      service_account = "fsx-csi-controller-sa"
    }
  }
  tags = local.tags
}
