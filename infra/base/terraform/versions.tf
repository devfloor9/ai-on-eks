terraform {
  required_version = ">= 1.3.2"

  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = ">=2.7.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.22"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.10"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.0"
    }
    http = {
      source  = "hashicorp/http"
      version = ">=3.5.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.6.0" # Replace with the appropriate version of the random provider
    }
  }
}
