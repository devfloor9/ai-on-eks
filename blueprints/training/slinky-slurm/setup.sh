#!/bin/bash

# Default values
REPO_NAME="dlc-slurmd"
IMAGE_TAG="25.11.1-ubuntu24.04"
AWS_REGION=$(aws configure get region || echo "us-west-2")
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SKIP_IMAGE_BUILD=false

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-name)
      REPO_NAME="$2"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --region)
      AWS_REGION="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_IMAGE_BUILD=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "      --repo-name    Repository name (default: dlc-slurmd)"
      echo "      --tag          Image tag (default: 25.11.1-ubuntu24.04)"
      echo "      --region       AWS region (default: AWS CLI configured region or us-west-2)"
      echo "      --skip-build   Skip image build (use existing image in ECR)"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

IMAGE_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

if [[ "$SKIP_IMAGE_BUILD" == "true" ]]; then
  echo "Skipping image build..."

  # Check if image exists in ECR
  if ! aws ecr describe-images --repository-name ${REPO_NAME} --image-ids imageTag=${IMAGE_TAG} --region ${AWS_REGION} >/dev/null 2>&1; then
    echo "Error: Image ${IMAGE_REPOSITORY}:${IMAGE_TAG} not found in ECR"
    echo "Run without --skip-build flag to build and push the image first, add --help for help"
    exit 1
  fi

  echo "Found existing image: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"

else
    # Authenticate to DLC repo (Account 763104351884 is publicly known)
    echo "Authenticating to the DLC ECR repo ..."
    aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS \
    --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

    # Build the DLC Slurmd container image
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Building image $REPO_NAME:$IMAGE_TAG on macOS..."
        docker buildx build --platform linux/amd64 -t ${REPO_NAME}:${IMAGE_TAG} -f dlc-slurmd.Dockerfile .
    else
        # Linux
        echo "Building image $REPO_NAME:$IMAGE_TAG on Linux..."
        docker build -t ${REPO_NAME}:${IMAGE_TAG} -f dlc-slurmd.Dockerfile .
    fi

    # Create ECR repo
    echo "Creating ECR repo $REPO_NAME (if it doesn't exist)"
    aws ecr create-repository --no-cli-pager --repository-name $REPO_NAME --region $AWS_REGION || echo "Repository already exists, continuing..."

    # Authenticate to the repo
    echo "Authenticating to ECR in region $AWS_REGION..."
    aws ecr get-login-password --region $AWS_REGION \
    | docker login --username AWS \
    --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

    # Tag the image
    echo "Taging the image with $IMAGE_REPOSITORY:$IMAGE_TAG..."
    docker tag ${REPO_NAME}:${IMAGE_TAG} ${IMAGE_REPOSITORY}:${IMAGE_TAG}

    # Push image to ECR
    echo "Pushing the image $IMAGE_REPOSITORY:$IMAGE_TAG to ECR..."
    docker push ${IMAGE_REPOSITORY}:${IMAGE_TAG}
fi

# Make SSH Keys
if [[ ! -f ~/.ssh/id_ed25519_slurm ]]; then
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_slurm -C "slurm-login" -N ""
fi

# Get the public key
SSH_KEY="$(cat ~/.ssh/id_ed25519_slurm.pub)"

# Generate slurm-values.yaml from template
sed -e "s|\${image_repository}|${IMAGE_REPOSITORY}|g" \
    -e "s|\${image_tag}|${IMAGE_TAG}|g" \
    -e "s|\${ssh_key}|${SSH_KEY}|g" \
    slurm-values.yaml.template > slurm-values.yaml
