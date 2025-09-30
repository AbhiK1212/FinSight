#!/bin/bash
# Deploy FinSight to AWS App Runner

set -e

# Configuration
APP_NAME="finsight"
ECR_REPO_NAME="finsight"
REGION=${AWS_REGION:-"us-east-1"}

echo "🚀 Deploying FinSight to AWS App Runner"
echo "App Name: ${APP_NAME}"
echo "Region: ${REGION}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI first."
    echo "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure'"
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "📋 AWS Account: ${ACCOUNT_ID}"
echo "📦 ECR Repository: ${ECR_URI}"

# Create ECR repository if it doesn't exist
echo "🏗️  Creating ECR repository..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${REGION}

# Get ECR login token
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# Build and tag the image
echo "🔨 Building Docker image..."
docker build -t ${ECR_REPO_NAME} .
docker tag ${ECR_REPO_NAME}:latest ${ECR_URI}:latest

# Push to ECR
echo "📤 Pushing image to ECR..."
docker push ${ECR_URI}:latest

# Create apprunner.yaml for deployment
cat > apprunner.yaml << EOF
version: 1.0
runtime: docker
build:
  commands:
    build:
      - echo Build started on \`date\`
      - echo Build completed on \`date\`
run:
  runtime-version: latest
  command: uvicorn src.insight.api.app:app --host 0.0.0.0 --port 8000 --workers 2
  network:
    port: 8000
  env:
    - name: API_DEBUG
      value: "false"
    - name: LOG_LEVEL
      value: "INFO"
    - name: PORT
      value: "8000"
EOF

echo "📋 Created apprunner.yaml configuration"

# Instructions for manual deployment (App Runner doesn't have direct CLI deployment)
echo ""
echo "🎯 Next Steps - Complete deployment in AWS Console:"
echo "1. Go to AWS App Runner Console: https://console.aws.amazon.com/apprunner/"
echo "2. Click 'Create an App Runner service'"
echo "3. Select 'Container registry' as source"
echo "4. Use ECR URI: ${ECR_URI}:latest"
echo "5. Deployment trigger: Manual"
echo "6. Service settings:"
echo "   - Service name: ${APP_NAME}"
echo "   - Port: 8000"
echo "   - CPU: 1 vCPU"
echo "   - Memory: 2 GB"
echo "7. Auto scaling: 1-10 instances"
echo ""
echo "🔗 After deployment, your API will be available at:"
echo "   https://[random-id].${REGION}.awsapprunner.com"
