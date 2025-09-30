#!/usr/bin/env python3
"""
FinSight Setup Script
Automates the integration of all external services
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"📋 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return None


def check_requirements():
    """Check if required tools are installed."""
    print("🔍 Checking requirements...")
    
    requirements = {
        "python3": "python3 --version",
        "docker": "docker --version", 
        "docker-compose": "docker-compose --version"
    }
    
    missing = []
    for tool, cmd in requirements.items():
        if not run_command(cmd, f"Checking {tool}"):
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing requirements: {', '.join(missing)}")
        print("Please install missing tools and run again.")
        sys.exit(1)
    
    print("✅ All requirements satisfied")


def setup_environment():
    """Set up Python environment and install dependencies."""
    print("\n🐍 Setting up Python environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        run_command("python3 -m venv venv", "Creating virtual environment")
    
    # Install dependencies
    pip_cmd = "./venv/bin/pip install -r requirements.txt"
    run_command(pip_cmd, "Installing Python dependencies")
    
    # Install dev dependencies
    pip_dev_cmd = "./venv/bin/pip install -r requirements-dev.txt"
    run_command(pip_dev_cmd, "Installing development dependencies")


def setup_config():
    """Set up configuration files."""
    print("\n⚙️  Setting up configuration...")
    
    # Copy env file if it doesn't exist
    if not os.path.exists(".env"):
        if os.path.exists("env.example"):
            run_command("cp env.example .env", "Creating .env file")
            print("📝 Edit .env file with your API keys:")
            print("   - Get NewsAPI key: https://newsapi.org/register")
            print("   - Get Alpha Vantage key: https://www.alphavantage.co/support/#api-key")
        else:
            print("⚠️  env.example not found, creating basic .env")
            with open(".env", "w") as f:
                f.write("""API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
DATABASE_URL=postgresql://finsight:password@localhost:5432/finsight
REDIS_URL=redis://localhost:6379/0
MODEL_NAME=distilbert-base-uncased
MODEL_PATH=./data/models/sentiment_model
LOG_LEVEL=INFO
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
""")
    
    # Create necessary directories
    dirs = ["data/raw", "data/processed", "data/models", "logs", "mlruns"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")


def start_infrastructure():
    """Start Docker infrastructure services."""
    print("\n🐳 Starting infrastructure services...")
    
    # Stop any existing containers
    run_command("docker-compose down", "Stopping existing containers")
    
    # Start new containers
    run_command("docker-compose up -d", "Starting Docker services")
    
    # Wait for services to be ready
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check service health
    services = [
        ("PostgreSQL", "docker exec finsight-postgres pg_isready -U finsight"),
        ("Redis", "docker exec finsight-redis redis-cli ping"),
    ]
    
    for service_name, cmd in services:
        if run_command(cmd, f"Checking {service_name}"):
            print(f"✅ {service_name} is ready")
        else:
            print(f"❌ {service_name} failed to start")


def test_system():
    """Test the complete system."""
    print("\n🧪 Testing system components...")
    
    # Test demo script
    run_command("./venv/bin/python demo.py", "Testing core sentiment analysis")
    
    # Test database setup
    run_command("./venv/bin/python scripts/setup_database.py", "Setting up database schema")
    
    # Start API in background and test
    print("🚀 Starting API server...")
    api_process = subprocess.Popen([
        "./venv/bin/python", "-m", "uvicorn", 
        "src.insight.api.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])
    
    # Wait for API to start
    time.sleep(5)
    
    # Test API endpoints
    test_commands = [
        ("curl -s http://localhost:8000/api/v1/health", "API health check"),
        ("curl -s http://localhost:8000/docs", "API documentation"),
    ]
    
    for cmd, desc in test_commands:
        if run_command(cmd, desc):
            print(f"✅ {desc} working")
    
    # Stop API
    api_process.terminate()
    
    print("🎉 System test completed!")


def main():
    """Main setup orchestration."""
    print("🚀 FinSight Setup Starting...")
    print("="*50)
    
    try:
        # Step 1: Check requirements
        check_requirements()
        
        # Step 2: Set up Python environment  
        setup_environment()
        
        # Step 3: Set up configuration
        setup_config()
        
        # Step 4: Start infrastructure
        start_infrastructure()
        
        # Step 5: Test system
        test_system()
        
        print("\n" + "="*50)
        print("🎉 FinSight setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python scripts/full_pipeline.py")
        print("3. Start API: uvicorn src.insight.api.app:app --reload")
        print("4. Open: http://localhost:8000/docs")
        print("\n🔗 Access points:")
        print("- API Docs: http://localhost:8000/docs")
        print("- MLflow: http://localhost:5000") 
        print("- Grafana: http://localhost:3000 (admin/admin)")
        print("- Prometheus: http://localhost:9090")
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
