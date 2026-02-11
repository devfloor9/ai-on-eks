locals {
  argo_events_namespace       = "argo-events"
  argo_events_service_account = "argo-events"
  argo_events_values = yamldecode(templatefile("${path.module}/helm-values/argo-events.yaml", {
  }))
}

#---------------------------------------------------------------
# Argo Events Namespace and Service Account
#---------------------------------------------------------------
resource "kubectl_manifest" "argo_events_manifests" {
  for_each = var.enable_argo_events ? fileset("${path.module}/helm-values/argo-events", "*.yaml") : []

  yaml_body = file("${path.module}/helm-values/argo-events/${each.value}")

  depends_on = [
    kubectl_manifest.argo_events
  ]
}

#---------------------------------------------------------------
# Pod Identity for Argo Events SQS Access
#---------------------------------------------------------------
module "argo_events_pod_identity" {
  count   = var.enable_argo_events ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"

  name = "ai-on-eks-argo-events"

  additional_policy_arns = {
    sqs_access = aws_iam_policy.sqs_argo_events[0].arn
  }

  associations = {
    argo_events = {
      cluster_name    = module.eks.cluster_name
      namespace       = local.argo_events_namespace
      service_account = local.argo_events_service_account
    }
  }
  tags = local.tags
}

#---------------------------------------------------------------
# IAM Policy for Argo Events SQS Access
#---------------------------------------------------------------
data "aws_iam_policy_document" "sqs_argo_events" {
  statement {
    sid       = "AllowReadingAndSendingSQSfromArgoEvents"
    effect    = "Allow"
    resources = ["*"]
    actions = [
      "sqs:ListQueues",
      "sqs:GetQueueUrl",
      "sqs:ListDeadLetterSourceQueues",
      "sqs:ListMessageMoveTasks",
      "sqs:ReceiveMessage",
      "sqs:SendMessage",
      "sqs:GetQueueAttributes",
      "sqs:ListQueueTags",
      "sqs:DeleteMessage"
    ]
  }
}

resource "aws_iam_policy" "sqs_argo_events" {
  count       = var.enable_argo_events ? 1 : 0
  name        = "ai-on-eks-argo-events-sqs-policy"
  description = "IAM policy for Argo Events SQS access"
  policy      = data.aws_iam_policy_document.sqs_argo_events.json
  tags        = local.tags
}

#---------------------------------------------------------------
# Argo Events Application
#---------------------------------------------------------------
resource "kubectl_manifest" "argo_events" {
  count = var.enable_argo_events ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/argo-events.yaml", {
    user_values_yaml = indent(10, yamlencode(local.argo_events_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
