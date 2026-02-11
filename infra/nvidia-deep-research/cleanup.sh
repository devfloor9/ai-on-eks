#!/bin/bash

# Clean up Terraform infrastructure
cd terraform/_LOCAL
source ./cleanup.sh

# Go back to infra directory
cd ../..

# Clean up configuration
echo "Cleaning up configuration files..."

# Clean up PID files
for pid_file in /tmp/.port-forward-*.pid; do
    if [ -f "$pid_file" ]; then
        rm "$pid_file"
    fi
done
echo "Configuration cleaned up"

echo ""
echo "============================================"
echo "Manual cleanup (if desired):"
echo "  - .env        (contains reusable API keys)"
echo "  - rag/        (RAG source code)"
echo "  - opensearch/ (OpenSearch integration files)"
echo ""
echo "To remove these, run:"
echo "  rm .env"
echo "  rm -r rag opensearch"
echo "============================================"
