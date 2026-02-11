#---------------------------------------------------------------
# ETCD for TorchX
#---------------------------------------------------------------
data "http" "torchx_etcd_yaml" {
  url = "https://raw.githubusercontent.com/pytorch/torchx/main/resources/etcd.yaml"
}

data "kubectl_file_documents" "torchx_etcd_yaml" {
  content = replace(
    data.http.torchx_etcd_yaml.response_body,
    "image: quay.io/coreos/etcd:latest",
    "image: quay.io/coreos/etcd:v3.5.0"
  )
}

resource "kubectl_manifest" "torchx_etcd" {
  for_each   = var.enable_torchx_etcd ? data.kubectl_file_documents.torchx_etcd_yaml.manifests : {}
  yaml_body  = each.value
  depends_on = [module.eks.eks_cluster_id]
}
