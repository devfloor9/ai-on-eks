################################################################################
# Observability Stack - Grafana Dashboards
# Grafana Operator is deployed in base terraform (enable_grafana_operator=true)
# Grafana credentials are managed by base terraform (grafana-admin-secret)
# This file manages workshop-specific dashboards
# Note: Namespace is configured via kube_prometheus_stack_namespace variable
################################################################################

resource "kubectl_manifest" "external_grafana" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: Grafana
    metadata:
      name: external-grafana
      namespace: ${var.kube_prometheus_stack_namespace}
      labels:
        dashboards: external-grafana
    spec:
      external:
        url: http://kube-prometheus-stack-grafana.${var.kube_prometheus_stack_namespace}.svc.cluster.local:${var.grafana_service_port}
        adminUser:
          name: grafana-admin-secret
          key: admin-user
        adminPassword:
          name: grafana-admin-secret
          key: admin-password
  YAML
}

resource "kubectl_manifest" "vllm_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: vllm-grafana-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      vllm-dashboard.json: ${jsonencode(file("${path.module}/grafana-dashboards/vllm-dashboard.json"))}
  YAML
}

resource "kubectl_manifest" "ray_default_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-default-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-default-grafana-dashboard.json: ${jsonencode(file("${path.module}/grafana-dashboards/ray-default-grafana-dashboard.json"))}
  YAML
}

resource "kubectl_manifest" "ray_serve_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-serve-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-serve-grafana-dashboard.json: ${jsonencode(file("${path.module}/grafana-dashboards/ray-serve-grafana-dashboard.json"))}
  YAML
}

resource "kubectl_manifest" "ray_serve_deployment_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-serve-deployment-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-serve-deployment-grafana-dashboard.json: ${jsonencode(file("${path.module}/grafana-dashboards/ray-serve-deployment-grafana-dashboard.json"))}
  YAML
}

resource "kubectl_manifest" "dcgm_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: dcgm-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      dcgm-grafana-dashboard.json: ${jsonencode(file("${path.module}/grafana-dashboards/dcgm-grafana-dashboard.json"))}
  YAML
}
