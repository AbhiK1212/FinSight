#!/bin/bash
# Quick script to query Prometheus from the command line

PROMETHEUS_URL="http://localhost:9090"
QUERY="$1"

if [ -z "$QUERY" ]; then
    echo "📊 FinSight Prometheus Query Tool"
    echo "=================================="
    echo ""
    echo "Usage: $0 '<prometheus_query>'"
    echo ""
    echo "Example queries:"
    echo ""
    echo "1. Total requests:"
    echo "   $0 'finsight_http_requests_total'"
    echo ""
    echo "2. Request rate (last 5 minutes):"
    echo "   $0 'rate(finsight_http_requests_total[5m])'"
    echo ""
    echo "3. Predictions by sentiment:"
    echo "   $0 'sum by (sentiment) (finsight_predictions_total)'"
    echo ""
    echo "4. 95th percentile latency:"
    echo "   $0 'histogram_quantile(0.95, rate(finsight_http_request_duration_seconds_bucket[5m]))'"
    echo ""
    echo "5. Active requests:"
    echo "   $0 'finsight_active_requests'"
    echo ""
    echo "6. Cache hit rate:"
    echo "   $0 'rate(finsight_cache_hits_total[5m]) / (rate(finsight_cache_hits_total[5m]) + rate(finsight_cache_misses_total[5m])) * 100'"
    echo ""
    exit 1
fi

echo "Querying: $QUERY"
echo ""

# Query Prometheus
RESULT=$(curl -s -G "$PROMETHEUS_URL/api/v1/query" --data-urlencode "query=$QUERY")

# Check if jq is available
if command -v jq &> /dev/null; then
    echo "$RESULT" | jq '.data.result[] | {metric: .metric, value: .value[1]}'
else
    echo "$RESULT"
    echo ""
    echo "💡 Tip: Install 'jq' for prettier output: brew install jq"
fi
