locals {
  jupyter_single_user_sa_name = "${module.eks.cluster_name}-jupyterhub-single-user"
  cognito_custom_domain       = var.cognito_custom_domain

  jupyterhub_values = templatefile("${path.module}/helm-values/jupyterhub-values-${var.jupyter_hub_auth_mechanism}.yaml", {
    ssl_cert_arn                = try(data.aws_acm_certificate.issued[0].arn, "")
    jupyterdomain               = try("https://${var.jupyterhub_domain}/hub/oauth_callback", "")
    authorize_url               = var.oauth_domain != "" ? "${var.oauth_domain}/auth" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/authorize", "")
    token_url                   = var.oauth_domain != "" ? "${var.oauth_domain}/token" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/token", "")
    userdata_url                = var.oauth_domain != "" ? "${var.oauth_domain}/userinfo" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/userInfo", "")
    username_key                = try(var.oauth_username_key, "")
    client_id                   = var.oauth_jupyter_client_id != "" ? var.oauth_jupyter_client_id : try(aws_cognito_user_pool_client.user_pool_client[0].id, "")
    client_secret               = var.oauth_jupyter_client_secret != "" ? var.oauth_jupyter_client_secret : try(aws_cognito_user_pool_client.user_pool_client[0].client_secret, "")
    user_pool_id                = try(aws_cognito_user_pool.pool[0].id, "")
    identity_pool_id            = try(aws_cognito_identity_pool.identity_pool[0].id, "")
    jupyter_single_user_sa_name = local.jupyter_single_user_sa_name
    region                      = var.region
  })
}
#-----------------------------------------------------------------------------------------
# JupyterHub Single User Pod Identity Policy, maybe that block could be incorporated in add-on registry
#-----------------------------------------------------------------------------------------
resource "kubernetes_namespace" "jupyterhub" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name = "jupyterhub"
  }
}

module "jupyterhub_pod_identity" {
  count   = var.enable_jupyterhub ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"

  name = "jupyterhub-pod-identity"

  additional_policy_arns = {
    jupyterhub_policy = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  }

  associations = {
    jupyterhub = {
      cluster_name    = module.eks.cluster_name
      namespace       = "jupyterhub"
      service_account = "${module.eks.cluster_name}-jupyterhub-single-user"
    }
  }
  tags = local.tags
}

resource "kubernetes_service_account_v1" "jupyterhub_single_user_sa" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = local.jupyter_single_user_sa_name
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
  }
}

#---------------------------------------
# EFS Configuration
#---------------------------------------
resource "aws_efs_access_point" "efs_persist_ap" {
  count          = var.enable_jupyterhub ? 1 : 0
  file_system_id = module.efs[0].id
  posix_user {
    gid            = 0
    uid            = 0
    secondary_gids = [100]
  }
  root_directory {
    path = "/home"
    creation_info {
      owner_gid   = 0
      owner_uid   = 0
      permissions = 700
    }
  }
  depends_on = [module.efs]
  tags       = local.tags
}
resource "aws_efs_access_point" "efs_shared_ap" {
  count          = var.enable_jupyterhub ? 1 : 0
  file_system_id = module.efs[0].id
  posix_user {
    gid            = 0
    uid            = 0
    secondary_gids = [100]
  }
  root_directory {
    path = "/shared"
    creation_info {
      owner_gid   = 0
      owner_uid   = 0
      permissions = 700
    }
  }
  depends_on = [module.efs]
  tags       = local.tags
}

resource "kubernetes_persistent_volume_v1" "efs_persist" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name = "efs-persist"
    labels = {
      "volume-name" = "efs-persist"
    }
  }
  spec {
    access_modes                     = ["ReadWriteMany"]
    capacity                         = { storage : "123Gi" }
    storage_class_name               = kubernetes_storage_class_v1.efs[count.index].metadata[0].name
    persistent_volume_reclaim_policy = "Retain"
    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${module.efs[0].id}::${aws_efs_access_point.efs_persist_ap[0].id}"
      }
    }
  }
  depends_on = [
    kubernetes_namespace.jupyterhub,
    module.efs
  ]
}

resource "kubernetes_persistent_volume_v1" "efs_shared" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name = "efs-persist-shared"
    labels = {
      "volume-name" = "efs-persist-shared"
    }
  }
  spec {
    access_modes                     = ["ReadWriteMany"]
    capacity                         = { storage : "123Gi" }
    storage_class_name               = kubernetes_storage_class_v1.efs[count.index].metadata[0].name
    persistent_volume_reclaim_policy = "Retain"
    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = "${module.efs[0].id}::${aws_efs_access_point.efs_shared_ap[0].id}"
      }
    }
  }
  depends_on = [
    kubernetes_namespace.jupyterhub,
    module.efs
  ]
}

resource "kubernetes_persistent_volume_claim_v1" "efs" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "efs-persist"
    namespace = "jupyterhub"
  }
  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs[count.index].metadata[0].name
    selector {
      match_labels = {
        "volume-name" = "efs-persist"
      }
    }
    resources {
      requests = {
        storage = "123Gi"
      }
    }
  }
  depends_on = [
    kubernetes_persistent_volume_v1.efs_persist
  ]
}

resource "kubernetes_persistent_volume_claim_v1" "efs_shared" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "efs-persist-shared"
    namespace = "jupyterhub"
  }
  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = kubernetes_storage_class_v1.efs[count.index].metadata[0].name
    selector {
      match_labels = {
        "volume-name" = "efs-persist-shared"
      }
    }
    resources {
      requests = {
        storage = "123Gi"
      }
    }
  }
  depends_on = [
    kubernetes_persistent_volume_v1.efs_shared
  ]
}

#---------------------------------------------------------------
# Additional Resources
#---------------------------------------------------------------
resource "kubernetes_secret_v1" "huggingface_token" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "hf-token"
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
  }

  data = {
    token = var.huggingface_token
  }
}

resource "kubernetes_config_map_v1" "notebook" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "notebook"
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
  }
}

resource "kubectl_manifest" "jupyterhub" {
  count = var.enable_jupyterhub ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/jupyterhub.yaml", {
    user_values_yaml = indent(8, local.jupyterhub_values)
  })

  depends_on = [
    helm_release.argocd,
    kubernetes_config_map_v1.notebook,
    kubernetes_persistent_volume_claim_v1.efs,
    kubernetes_persistent_volume_claim_v1.efs_shared,
    kubernetes_service_account_v1.jupyterhub_single_user_sa,
    aws_efs_access_point.efs_persist_ap,
    aws_efs_access_point.efs_shared_ap
  ]
}
