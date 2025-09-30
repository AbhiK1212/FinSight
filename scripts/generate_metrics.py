#!/usr/bin/env python3
"""
Generate realistic API traffic to populate Prometheus metrics and Grafana dashboards.
This script simulates various API usage patterns to demonstrate monitoring capabilities.
"""

import requests
import random
import time
import concurrent.futures
from typing import List, Dict
import argparse
from datetime import datetime

# Sample financial headlines for testing
SAMPLE_HEADLINES = [
    # Positive sentiment
    "Apple reports record quarterly profits exceeding expectations",
    "Tesla stock surges on strong delivery numbers",
    "Microsoft announces breakthrough in AI technology",
    "Amazon revenue beats Wall Street estimates",
    "Google parent Alphabet reaches new market cap milestone",
    "NVIDIA stock rallies on AI chip demand",
    "Meta platforms revenue grows 25% year over year",
    "JPMorgan Chase reports strong quarterly earnings",
    "Goldman Sachs exceeds profit expectations",
    "Berkshire Hathaway reaches all-time high",
    
    # Negative sentiment
    "Boeing faces major production delays and cost overruns",
    "Tesla recalls thousands of vehicles due to safety concerns",
    "Netflix loses subscribers for third consecutive quarter",
    "Facebook faces major data breach affecting millions",
    "Twitter stock plummets on disappointing user growth",
    "Uber reports widening losses in quarterly report",
    "WeWork valuation crashes amid profitability concerns",
    "General Electric stock falls on earnings miss",
    "Ford announces major layoffs amid restructuring",
    "IBM revenue declines for fifth straight quarter",
    
    # Neutral sentiment
    "Federal Reserve maintains interest rates at current levels",
    "S&P 500 closes flat amid mixed economic signals",
    "Oil prices remain stable in Asian trading",
    "Dollar holds steady against major currencies",
    "Treasury yields unchanged in morning trading",
    "Gold prices trade in narrow range",
    "European markets open little changed",
    "Asian stocks mixed as investors await data",
    "Bitcoin trades sideways amid regulatory discussions",
    "Commodity markets show muted activity",
]


class MetricsGenerator:
    """Generate API traffic to populate metrics."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "start_time": time.time(),
        }
    
    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False
    
    def single_prediction(self, text: str) -> Dict:
        """Make a single prediction request."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict",
                json={"text": text, "return_confidence": True},
                timeout=10
            )
            self.stats["total_requests"] += 1
            
            if response.status_code == 200:
                self.stats["successful"] += 1
                return response.json()
            else:
                self.stats["failed"] += 1
                return {"error": f"Status {response.status_code}"}
        except Exception as e:
            self.stats["failed"] += 1
            return {"error": str(e)}
    
    def batch_prediction(self, texts: List[str]) -> Dict:
        """Make a batch prediction request."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict/batch",
                json={"texts": texts},
                timeout=30
            )
            self.stats["total_requests"] += 1
            
            if response.status_code == 200:
                self.stats["successful"] += 1
                return response.json()
            else:
                self.stats["failed"] += 1
                return {"error": f"Status {response.status_code}"}
        except Exception as e:
            self.stats["failed"] += 1
            return {"error": str(e)}
    
    def generate_realistic_traffic(self, duration_seconds: int = 60, requests_per_second: float = 2.0):
        """Generate realistic API traffic pattern."""
        print(f"🚀 Generating traffic for {duration_seconds} seconds at ~{requests_per_second} req/s")
        print(f"📊 Total expected requests: ~{int(duration_seconds * requests_per_second)}")
        print()
        
        end_time = time.time() + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            # Randomly choose between single and batch predictions (80% single, 20% batch)
            if random.random() < 0.8:
                # Single prediction
                headline = random.choice(SAMPLE_HEADLINES)
                result = self.single_prediction(headline)
                request_count += 1
                
                if "error" not in result:
                    sentiment = result.get("sentiment", "unknown")
                    confidence = result.get("confidence", 0)
                    print(f"✓ [{request_count}] Single: '{headline[:50]}...' → {sentiment} ({confidence:.3f})")
                else:
                    print(f"✗ [{request_count}] Single request failed: {result['error']}")
            else:
                # Batch prediction
                batch_size = random.randint(2, 5)
                batch_texts = random.sample(SAMPLE_HEADLINES, batch_size)
                result = self.batch_prediction(batch_texts)
                request_count += 1
                
                if "error" not in result:
                    predictions = result.get("predictions", [])
                    print(f"✓ [{request_count}] Batch: {batch_size} headlines → {len(predictions)} predictions")
                else:
                    print(f"✗ [{request_count}] Batch request failed: {result['error']}")
            
            # Sleep to control request rate
            time.sleep(1.0 / requests_per_second + random.uniform(-0.1, 0.1))
        
        self.print_summary()
    
    def generate_burst_traffic(self, num_requests: int = 50, max_workers: int = 5):
        """Generate burst traffic to test concurrent request handling."""
        print(f"💥 Generating burst traffic: {num_requests} concurrent requests")
        print()
        
        headlines = [random.choice(SAMPLE_HEADLINES) for _ in range(num_requests)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.single_prediction, headline) for headline in headlines]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                if "error" not in result:
                    sentiment = result.get("sentiment", "unknown")
                    print(f"✓ [{i}/{num_requests}] {sentiment}")
                else:
                    print(f"✗ [{i}/{num_requests}] Failed")
        
        print()
        self.print_summary()
    
    def generate_varied_patterns(self, duration_seconds: int = 120):
        """Generate varied traffic patterns to populate different metrics."""
        print(f"🎯 Generating varied traffic patterns for {duration_seconds} seconds")
        print()
        
        patterns = [
            ("Low traffic", 0.5, 20),
            ("Medium traffic", 2.0, 30),
            ("High traffic", 5.0, 20),
            ("Burst traffic", 10.0, 15),
            ("Cool down", 1.0, duration_seconds - 85),
        ]
        
        for pattern_name, rate, duration in patterns:
            if duration <= 0:
                continue
            print(f"\n{'='*60}")
            print(f"📈 Pattern: {pattern_name} ({rate} req/s for {duration}s)")
            print(f"{'='*60}\n")
            self.generate_realistic_traffic(duration, rate)
            time.sleep(2)  # Brief pause between patterns
        
        print("\n" + "="*60)
        print("🎉 All traffic patterns completed!")
        print("="*60)
        self.print_summary()
    
    def stress_test(self, duration_seconds: int = 30, max_rate: float = 10.0):
        """Stress test to generate high load metrics."""
        print(f"⚡ Stress testing: {max_rate} req/s for {duration_seconds} seconds")
        print()
        
        self.generate_realistic_traffic(duration_seconds, max_rate)
    
    def print_summary(self):
        """Print statistics summary."""
        elapsed = time.time() - self.stats["start_time"]
        avg_rate = self.stats["total_requests"] / elapsed if elapsed > 0 else 0
        success_rate = (self.stats["successful"] / self.stats["total_requests"] * 100) if self.stats["total_requests"] > 0 else 0
        
        print()
        print("="*60)
        print("📊 METRICS GENERATION SUMMARY")
        print("="*60)
        print(f"Total Requests:     {self.stats['total_requests']}")
        print(f"Successful:         {self.stats['successful']} ({success_rate:.1f}%)")
        print(f"Failed:             {self.stats['failed']}")
        print(f"Duration:           {elapsed:.1f}s")
        print(f"Average Rate:       {avg_rate:.2f} req/s")
        print("="*60)
        print()
        print("📍 Check your metrics at:")
        print(f"   • Prometheus: http://localhost:9090")
        print(f"   • Grafana:    http://localhost:3000")
        print(f"   • API Metrics: {self.base_url}/metrics")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate API traffic to populate Prometheus metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate realistic traffic for 60 seconds
  python scripts/generate_metrics.py --mode realistic --duration 60
  
  # Generate burst traffic
  python scripts/generate_metrics.py --mode burst --requests 100
  
  # Generate varied patterns
  python scripts/generate_metrics.py --mode varied --duration 120
  
  # Stress test
  python scripts/generate_metrics.py --mode stress --duration 30 --rate 10
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["realistic", "burst", "varied", "stress"],
        default="realistic",
        help="Traffic generation mode (default: realistic)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Requests per second for realistic/stress modes (default: 2.0)"
    )
    
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests for burst mode (default: 50)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Concurrent workers for burst mode (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MetricsGenerator(base_url=args.url)
    
    # Check API health
    print("🔍 Checking API health...")
    if not generator.check_health():
        print("\n❌ API is not accessible. Please ensure:")
        print("   1. API is running: uvicorn src.insight.api.app:app --reload")
        print("   2. API is accessible at:", args.url)
        return 1
    
    print("✅ API is healthy\n")
    
    # Generate traffic based on mode
    try:
        if args.mode == "realistic":
            generator.generate_realistic_traffic(args.duration, args.rate)
        elif args.mode == "burst":
            generator.generate_burst_traffic(args.requests, args.workers)
        elif args.mode == "varied":
            generator.generate_varied_patterns(args.duration)
        elif args.mode == "stress":
            generator.stress_test(args.duration, args.rate)
    except KeyboardInterrupt:
        print("\n\n⚠️  Traffic generation interrupted by user")
        generator.print_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
