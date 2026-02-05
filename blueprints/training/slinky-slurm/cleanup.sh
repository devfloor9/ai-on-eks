#!/bin/bash

# Uninstall Slurm Helm release and wait for resources to be deleted
helm uninstall slurm -n slurm --wait --timeout 5m 2>/dev/null || true

# Delete MariaDB (deployed separately via kubectl apply)
kubectl delete -f mariadb.yaml --timeout=60s 2>/dev/null || true

# Delete PVCs to ensure EBS volumes are properly cleaned up
kubectl delete pvc statesave-slurm-controller-0 storage-mariadb-0 -n slurm --timeout=60s 2>/dev/null || true

# Remove generated files
rm -f slurm-values.yaml slurm-login-service-patch.yaml

cd ../../../infra/slinky-slurm/terraform/_LOCAL/

./cleanup.sh
