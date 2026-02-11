#---------------------------------------------------------------
# Envoy AI Gateway - Bedrock Pod Identity
#---------------------------------------------------------------

# IAM Role for Bedrock Access
resource "aws_iam_role" "bedrock_role" {
  count = var.enable_envoy_ai_gateway ? 1 : 0

  name_prefix = "${local.name}-bedrock-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-bedrock-role"
    Purpose = "Bedrock Access for AI Gateway"
  })
}

# IAM Policy for Bedrock Access
resource "aws_iam_policy" "bedrock_policy" {
  count = var.enable_envoy_ai_gateway ? 1 : 0

  name_prefix = "${local.name}-bedrock-policy-"
  description = "Policy for Bedrock model access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-bedrock-policy"
    Purpose = "Bedrock Access for AI Gateway"
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "bedrock_policy_attachment" {
  count = var.enable_envoy_ai_gateway ? 1 : 0

  role       = aws_iam_role.bedrock_role[0].name
  policy_arn = aws_iam_policy.bedrock_policy[0].arn
}

# EKS Pod Identity Association for Bedrock Service Account (default namespace)
resource "aws_eks_pod_identity_association" "bedrock_pod_identity_default" {
  count = var.enable_envoy_ai_gateway ? 1 : 0

  cluster_name    = module.eks.cluster_name
  namespace       = "default"
  service_account = "ai-gateway-dataplane-aws"
  role_arn        = aws_iam_role.bedrock_role[0].arn

  tags = merge(local.tags, {
    Name    = "${local.name}-bedrock-pod-identity-default"
    Purpose = "Pod Identity for Bedrock AI Gateway (default namespace)"
  })
}

# EKS Pod Identity Association for Bedrock Service Account (envoy-gateway-system namespace)
resource "aws_eks_pod_identity_association" "bedrock_pod_identity_gateway" {
  count = var.enable_envoy_ai_gateway ? 1 : 0

  cluster_name    = module.eks.cluster_name
  namespace       = "envoy-gateway-system"
  service_account = "ai-gateway-dataplane-aws"
  role_arn        = aws_iam_role.bedrock_role[0].arn

  tags = merge(local.tags, {
    Name    = "${local.name}-bedrock-pod-identity-gateway"
    Purpose = "Pod Identity for Bedrock AI Gateway (envoy-gateway-system namespace)"
  })
}
