locals {
  aws_load_balancer_controller_service_account = "aws-load-balancer-controller-sa"
  aws_load_balancer_controller_namespace       = "kube-system"

  aws_load_balancer_controller_values = templatefile("${path.module}/helm-values/aws-load-balancer-controller.yaml", {
    cluster_name                   = module.eks.cluster_name
    service_account                = local.aws_load_balancer_controller_service_account
    region                         = local.region
    vpc_id                         = module.vpc.vpc_id
    enable_service_mutator_webhook = var.enable_service_mutator_webhook
  })
}

#---------------------------------------------------------------
# Pod Identity for AWS Load Balancer Controller
#---------------------------------------------------------------
module "aws_load_balancer_controller_pod_identity" {
  count   = var.enable_aws_load_balancer_controller && !var.enable_eks_auto_mode ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"

  name                            = "aws-load-balancer-controller"
  attach_aws_lb_controller_policy = true

  associations = {
    aws_load_balancer_controller = {
      cluster_name    = module.eks.cluster_name
      namespace       = local.aws_load_balancer_controller_namespace
      service_account = local.aws_load_balancer_controller_service_account
    }
  }
  tags = local.tags
}

#---------------------------------------------------------------
# AWS Load Balancer Controller Application
#---------------------------------------------------------------
resource "kubectl_manifest" "aws_load_balancer_controller" {
  count = var.enable_aws_load_balancer_controller && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aws-load-balancer-controller.yaml", {
    user_values_yaml = indent(8, local.aws_load_balancer_controller_values)
  })

  depends_on = [
    helm_release.argocd,
    module.aws_load_balancer_controller_pod_identity
  ]
}
