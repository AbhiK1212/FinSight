#!/bin/bash
# Verify that metrics are flowing from API → Prometheus → Grafana

set -e

echo "🔍 FinSight Metrics Verification"
echo "================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
ALL_OK=true

# Function to check service
check_service() {
    local name=$1
    local url=$2
    
    echo -n "Checking $name... "
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        ALL_OK=false
        return 1
    fi
}

echo "1️⃣  Checking Services"
echo "-------------------"
check_service "API Health" "http://localhost:8000/api/v1/health"
check_service "API Metrics Endpoint" "http://localhost:8000/metrics"
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"
echo ""

echo "2️⃣  Checking Prometheus Targets"
echo "------------------------------"
TARGET_STATUS=$(curl -s http://localhost:9090/api/v1/targets | python3 -c "
import sys, json
data = json.load(sys.stdin)
targets = data['data']['activeTargets']
api_target = [t for t in targets if t['labels']['job'] == 'finsight-api']
if api_target:
    t = api_target[0]
    print(f\"{t['health']}|{t['scrapeUrl']}|{t.get('lastError', '')}\")
else:
    print('not_found||')
" 2>/dev/null)

IFS='|' read -r health url error <<< "$TARGET_STATUS"

if [ "$health" = "up" ]; then
    echo -e "${GREEN}✅ Prometheus is scraping API successfully${NC}"
    echo "   URL: $url"
else
    echo -e "${RED}❌ Prometheus cannot scrape API${NC}"
    echo "   URL: $url"
    echo "   Error: $error"
    ALL_OK=false
fi
echo ""

echo "3️⃣  Checking Available Metrics"
echo "----------------------------"
METRIC_COUNT=$(curl -s http://localhost:8000/metrics | grep -c "^finsight_" || echo "0")
echo "Found $METRIC_COUNT FinSight metrics exposed by API"

if [ "$METRIC_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Metrics are being exposed${NC}"
    echo ""
    echo "Available metrics:"
    curl -s http://localhost:8000/metrics | grep "^# HELP finsight_" | sed 's/# HELP /  • /' | head -10
else
    echo -e "${RED}❌ No metrics found${NC}"
    ALL_OK=false
fi
echo ""

echo "4️⃣  Testing Prometheus Queries"
echo "----------------------------"

# Test query
QUERY_RESULT=$(curl -s -G 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=finsight_http_requests_total' | \
    python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data['data']['result']))" 2>/dev/null || echo "0")

if [ "$QUERY_RESULT" -gt 0 ]; then
    echo -e "${GREEN}✅ Prometheus has FinSight data ($QUERY_RESULT series)${NC}"
    
    # Show sample data
    echo ""
    echo "Sample query results:"
    curl -s -G 'http://localhost:9090/api/v1/query' \
        --data-urlencode 'query=finsight_http_requests_total' | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data['data']['result'][:5]:
    endpoint = r['metric'].get('endpoint', 'unknown')
    method = r['metric'].get('method', 'unknown')
    value = r['value'][1]
    print(f'  • {method} {endpoint}: {value}')
" 2>/dev/null
else
    echo -e "${YELLOW}⚠️  No data in Prometheus yet${NC}"
    echo "   Tip: Generate some traffic first"
    echo "   Run: python scripts/generate_metrics.py --mode realistic --duration 30"
fi
echo ""

echo "5️⃣  Grafana Data Source"
echo "---------------------"
# Note: This requires Grafana API authentication, so we'll just provide instructions
echo "To verify Grafana data source:"
echo "1. Open http://localhost:3000"
echo "2. Login: admin/admin"
echo "3. Go to Configuration → Data Sources"
echo "4. Check that Prometheus is configured and working"
echo ""

echo "================================="
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}🎉 All checks passed!${NC}"
    echo ""
    echo "📊 Your metrics stack is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Open Prometheus: http://localhost:9090"
    echo "2. Try query: rate(finsight_http_requests_total[5m])"
    echo "3. Open Grafana: http://localhost:3000"
    echo "4. Import dashboard from: infrastructure/monitoring/grafana/dashboards/finsight-dashboard.json"
else
    echo -e "${RED}❌ Some checks failed${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Make sure API is running: uvicorn src.insight.api.app:app --reload"
    echo "2. Check Prometheus config: infrastructure/monitoring/prometheus.yml"
    echo "3. Restart services: docker compose restart prometheus grafana"
    echo "4. Check logs: docker compose logs prometheus"
fi
echo "================================="
