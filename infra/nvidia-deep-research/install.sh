#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

# Copy all custom .tf files from terraform/ to _LOCAL/
cp ./terraform/*.tf ./terraform/_LOCAL/ 2>/dev/null || true

cd terraform/_LOCAL
source ./install.sh
