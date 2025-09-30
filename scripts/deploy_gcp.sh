#!/bin/bash
# Deploy FinSight to Google Cloud Run

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION="us-central1"
SERVICE_NAME="finsight"
IMAGE_NAME="gcr.io/${PROJECT_ID}/finsight"

echo "🚀 Deploying FinSight to Google Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK first."
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set the project
echo "📋 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and submit the container image
echo "🔨 Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --max-instances 10 \
    --set-env-vars="API_DEBUG=false,LOG_LEVEL=INFO" \
    --timeout=300

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "🎉 Deployment completed successfully!"
echo "📍 Service URL: ${SERVICE_URL}"
echo "🔗 API Documentation: ${SERVICE_URL}/docs"
echo "💓 Health Check: ${SERVICE_URL}/api/v1/health"
echo ""
echo "📊 Test your deployed API:"
echo "curl ${SERVICE_URL}/api/v1/health"
echo ""
echo "curl -X POST \"${SERVICE_URL}/api/v1/predict\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"text\": \"Apple reports strong earnings\"}'"
