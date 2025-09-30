#!/bin/bash
# Quick script to generate metrics and view dashboards

set -e

echo "🚀 FinSight Metrics Demo"
echo "========================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if API is running
echo -e "${BLUE}Checking if API is running...${NC}"
if ! curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  API is not running. Starting it now...${NC}"
    echo ""
    echo "Run this in another terminal:"
    echo "  uvicorn src.insight.api.app:app --reload"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo -e "${GREEN}✅ API is running${NC}"
echo ""

# Check if monitoring is running
echo -e "${BLUE}Checking if Prometheus/Grafana are running...${NC}"
if ! curl -s http://localhost:9090 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Prometheus is not running${NC}"
    echo "Start it with: docker compose up -d prometheus grafana"
    echo ""
fi

# Generate metrics
echo -e "${BLUE}Generating realistic traffic to populate metrics...${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Run the metrics generator
python scripts/generate_metrics.py --mode varied --duration 120

echo ""
echo -e "${GREEN}🎉 Done! Check your dashboards:${NC}"
echo ""
echo -e "${BLUE}📊 Prometheus:${NC} http://localhost:9090"
echo -e "${BLUE}📈 Grafana:${NC}    http://localhost:3000 (admin/admin)"
echo -e "${BLUE}📋 API Docs:${NC}   http://localhost:8000/docs"
echo -e "${BLUE}🔍 Metrics:${NC}    http://localhost:8000/metrics"
echo ""
