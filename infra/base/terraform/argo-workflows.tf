locals {
  argo_workflows_values = yamldecode(templatefile("${path.module}/helm-values/argo-workflows.yaml", {
  }))
}

resource "kubectl_manifest" "argo_workflows" {
  count = var.enable_argo_workflows ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/argo-workflows.yaml", {
    user_values_yaml = indent(10, yamlencode(local.argo_workflows_values))
  })

  depends_on = [
    helm_release.argocd,
  ]
}
