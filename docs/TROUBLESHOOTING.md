# Troubleshooting Guide

## 📋 Overview

This guide provides solutions to common issues encountered when using, developing, or deploying the Stock Analysis System. Issues are organized by category with step-by-step resolution procedures.

## 🚀 Installation and Setup Issues

### Docker Installation Problems

#### Issue: Docker services fail to start
```bash
Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution:**
```bash
# Check Docker daemon status
sudo systemctl status docker

# Start Docker daemon
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
newgrp docker

# Test Docker installation
docker run hello-world
```

#### Issue: Port conflicts (5432, 6379, 8000 already in use)
```bash
Error: bind: address already in use
```

**Solution:**
```bash
# Check which processes are using the ports
sudo netstat -tlnp | grep :5432
sudo netstat -tlnp | grep :6379
sudo netstat -tlnp | grep :8000

# Kill conflicting processes (if safe to do so)
sudo kill -9 <PID>

# Or modify docker-compose.yml to use different ports
# Change "5432:5432" to "15432:5432"
# Update .env file accordingly
```

#### Issue: Docker Compose version compatibility
```bash
Error: Unsupported Compose file version
```

**Solution:**
```bash
# Check Docker Compose version
docker-compose --version

# Update Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Or use Docker Compose V2
docker compose up -d
```

### Python Environment Issues

#### Issue: Python version incompatibility
```bash
Error: Python 3.9+ required, found Python 3.8
```

**Solution:**
```bash
# Install Python 3.12 (Ubuntu/Debian)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv python3.12-dev

# Create virtual environment with specific Python version
python3.12 -m venv venv
source venv/bin/activate
```

#### Issue: Package installation failures
```bash
Error: Failed building wheel for some-package
```

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu/Debian)
sudo apt install build-essential python3-dev libpq-dev

# Install system dependencies (CentOS/RHEL)
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel postgresql-devel

# Clear pip cache and retry
pip cache purge
pip install -r requirements.txt
```

#### Issue: Virtual environment activation problems
```bash
Error: venv/bin/activate: No such file or directory
```

**Solution:**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation
which python
python --version
```

## 🗄️ Database Issues

### PostgreSQL Connection Problems

#### Issue: Database connection refused
```bash
Error: could not connect to server: Connection refused
```

**Solution:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check if PostgreSQL is listening on correct port
sudo netstat -tlnp | grep :5432

# Check PostgreSQL configuration
sudo -u postgres psql -c "SHOW port;"
sudo -u postgres psql -c "SHOW listen_addresses;"

# Test connection manually
psql -h localhost -U postgres -d stock_analysis
```

#### Issue: Authentication failed
```bash
Error: FATAL: password authentication failed for user "postgres"
```

**Solution:**
```bash
# Reset PostgreSQL password
sudo -u postgres psql
ALTER USER postgres PASSWORD 'new_password';
\q

# Update .env file with correct password
DB_PASSWORD=new_password

# Or use peer authentication for local connections
sudo -u postgres psql -c "CREATE USER $USER SUPERUSER;"
```

#### Issue: Database does not exist
```bash
Error: FATAL: database "stock_analysis" does not exist
```

**Solution:**
```bash
# Create database manually
sudo -u postgres psql
CREATE DATABASE stock_analysis;
CREATE USER stockapp WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE stock_analysis TO stockapp;
\q

# Or run setup script
python scripts/setup_dev.py
```

### Database Migration Issues

#### Issue: Migration fails with constraint errors
```bash
Error: duplicate key value violates unique constraint
```

**Solution:**
```bash
# Check current migration status
alembic current

# Show migration history
alembic history

# Rollback to previous migration
alembic downgrade -1

# Clean up problematic data
sudo -u postgres psql stock_analysis
DELETE FROM table_name WHERE condition;

# Retry migration
alembic upgrade head
```

#### Issue: Alembic revision conflicts
```bash
Error: Multiple heads detected
```

**Solution:**
```bash
# Show all heads
alembic heads

# Merge heads
alembic merge -m "merge heads" head1 head2

# Apply merged migration
alembic upgrade head
```

### Redis Connection Issues

#### Issue: Redis connection timeout
```bash
Error: Redis connection timeout
```

**Solution:**
```bash
# Check Redis status
sudo systemctl status redis-server

# Start Redis
sudo systemctl start redis-server

# Test Redis connection
redis-cli ping

# Check Redis configuration
redis-cli config get timeout
redis-cli config get maxclients

# Increase timeout if needed
redis-cli config set timeout 300
```

## 🌐 API and Backend Issues

### FastAPI Server Problems

#### Issue: Server fails to start
```bash
Error: Address already in use
```

**Solution:**
```bash
# Find process using port 8000
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or use different port
uvicorn stock_analysis_system.api.main:app --port 8001

# Check for import errors
python -c "from stock_analysis_system.api.main import app; print('Import successful')"
```

#### Issue: Import errors on startup
```bash
Error: ModuleNotFoundError: No module named 'stock_analysis_system'
```

**Solution:**
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or install package in development mode
pip install -e .

# Verify Python path
python -c "import sys; print(sys.path)"

# Check if __init__.py files exist
find stock_analysis_system -name "__init__.py"
```

#### Issue: Database session errors
```bash
Error: Session is not bound to a connection
```

**Solution:**
```bash
# Check database connection in code
from stock_analysis_system.core.database import get_session

async def test_db():
    async for session in get_session():
        result = await session.execute("SELECT 1")
        print(result.scalar())

# Ensure proper session management
# Use dependency injection in FastAPI endpoints
```

### Celery Worker Issues

#### Issue: Celery workers not starting
```bash
Error: consumer: Cannot connect to redis://localhost:6379/1
```

**Solution:**
```bash
# Check Redis connection
redis-cli -n 1 ping

# Check Celery configuration
python -c "from stock_analysis_system.etl.celery_app import celery_app; print(celery_app.conf.broker_url)"

# Start worker with debug info
celery -A stock_analysis_system.etl.celery_app worker --loglevel=debug

# Check worker status
celery -A stock_analysis_system.etl.celery_app status
```

#### Issue: Tasks not executing
```bash
Error: Task never received by worker
```

**Solution:**
```bash
# Check task registration
celery -A stock_analysis_system.etl.celery_app inspect registered

# Check queue status
celery -A stock_analysis_system.etl.celery_app inspect active_queues

# Purge stuck tasks
celery -A stock_analysis_system.etl.celery_app purge

# Monitor tasks in real-time
celery -A stock_analysis_system.etl.celery_app events
```

## 🎨 Frontend Issues

### React Development Server Problems

#### Issue: Frontend fails to start
```bash
Error: react-scripts: not found
```

**Solution:**
```bash
cd frontend

# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Check Node.js version
node --version  # Should be 16+
npm --version   # Should be 8+

# Update Node.js if needed (using nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

#### Issue: TypeScript compilation errors
```bash
Error: Type 'string' is not assignable to type 'number'
```

**Solution:**
```bash
# Check TypeScript configuration
cat frontend/tsconfig.json

# Fix type errors in code
# Example: Use proper type assertions
const value = response.data as ExpectedType;

# Or update type definitions
npm install --save-dev @types/node @types/react @types/react-dom

# Run type checking
cd frontend
npx tsc --noEmit
```

#### Issue: API connection failures
```bash
Error: Network Error / CORS policy
```

**Solution:**
```bash
# Check if backend is running
curl http://localhost:8000/health

# Verify proxy configuration in package.json
"proxy": "http://localhost:8000"

# Or configure CORS in backend
# In stock_analysis_system/api/main.py:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Chart Rendering Issues

#### Issue: Plotly charts not displaying
```bash
Error: Plotly is not defined
```

**Solution:**
```bash
cd frontend

# Reinstall Plotly
npm uninstall plotly.js
npm install plotly.js@latest

# Check import statements
# Correct import:
import Plotly from 'plotly.js-dist';

# Clear browser cache
# Chrome: Ctrl+Shift+R
# Firefox: Ctrl+F5
```

#### Issue: Chart export not working
```bash
Error: Failed to export chart
```

**Solution:**
```bash
# Install kaleido for server-side export
pip install kaleido

# Check if kaleido is working
python -c "import kaleido; print('Kaleido installed successfully')"

# For client-side export, ensure proper configuration
# In chart component:
const config = {
  toImageButtonOptions: {
    format: 'png',
    filename: 'chart',
    height: 800,
    width: 1200,
    scale: 2
  }
};
```

## 📊 Data and Analysis Issues

### Data Source Problems

#### Issue: External API rate limiting
```bash
Error: Rate limit exceeded for API
```

**Solution:**
```bash
# Check current rate limits
python -c "
from stock_analysis_system.data.data_source_manager import DataSourceManager
manager = DataSourceManager()
print(manager.get_rate_limit_status())
"

# Implement exponential backoff
# Already implemented in DataSourceManager

# Use multiple API keys if available
# Configure in .env:
TUSHARE_TOKEN_1=token1
TUSHARE_TOKEN_2=token2

# Monitor API usage
tail -f logs/app.log | grep "rate_limit"
```

#### Issue: Data quality validation failures
```bash
Error: Data quality score below threshold
```

**Solution:**
```bash
# Check data quality report
python -c "
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine
engine = EnhancedDataQualityEngine()
# Run quality check on your data
"

# Adjust quality thresholds if needed
# In config/settings.py:
DATA_QUALITY_THRESHOLD = 0.6  # Lower threshold

# Clean data manually
python scripts/clean_data.py --symbol 000001.SZ --start-date 2024-01-01
```

### Analysis Engine Issues

#### Issue: Spring Festival analysis fails
```bash
Error: Insufficient data for analysis
```

**Solution:**
```bash
# Check available data
python -c "
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalAlignmentEngine
engine = SpringFestivalAlignmentEngine()
# Check data availability for symbol
"

# Ensure minimum data requirements
# Need at least 3 years of data
# Need data around Spring Festival dates

# Download missing data
python scripts/download_historical_data.py --symbol 000001.SZ --years 2020,2021,2022,2023,2024
```

#### Issue: Memory errors during analysis
```bash
Error: MemoryError: Unable to allocate array
```

**Solution:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Use Dask for large datasets
from stock_analysis_system.analysis.parallel_spring_festival_engine import ParallelSpringFestivalEngine
engine = ParallelSpringFestivalEngine()

# Reduce batch size
# In config/settings.py:
DASK_CHUNK_SIZE = 50  # Reduce from default 100

# Increase system memory or use swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 🔧 Performance Issues

### Slow Query Performance

#### Issue: Database queries taking too long
```bash
Query execution time: 30+ seconds
```

**Solution:**
```bash
# Check database indexes
sudo -u postgres psql stock_analysis
\d+ stock_daily_data
\di

# Create missing indexes
CREATE INDEX CONCURRENTLY idx_stock_daily_data_symbol_date 
ON stock_daily_data(stock_code, trade_date);

# Analyze query performance
EXPLAIN ANALYZE SELECT * FROM stock_daily_data WHERE stock_code = '000001.SZ';

# Update table statistics
ANALYZE stock_daily_data;

# Consider partitioning large tables
# See database optimization guide
```

#### Issue: High memory usage
```bash
Memory usage consistently above 80%
```

**Solution:**
```bash
# Monitor memory usage
htop
free -h

# Check for memory leaks
python -m memory_profiler your_script.py

# Optimize pandas operations
# Use chunking for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Configure garbage collection
import gc
gc.collect()

# Limit worker memory
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

### API Response Time Issues

#### Issue: API endpoints responding slowly
```bash
Response time > 5 seconds
```

**Solution:**
```bash
# Enable API profiling
# In .env:
ENABLE_PROFILING=true

# Check slow queries
tail -f logs/app.log | grep "slow_query"

# Implement caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param):
    # Expensive computation
    return result

# Use Redis caching
import redis
r = redis.Redis()
r.setex("key", 3600, "value")  # Cache for 1 hour

# Optimize database queries
# Use select_related and prefetch_related
# Implement pagination for large results
```

## 🔒 Security Issues

### Authentication Problems

#### Issue: JWT token validation fails
```bash
Error: Invalid token signature
```

**Solution:**
```bash
# Check JWT secret key
python -c "
from config.settings import get_settings
settings = get_settings()
print(f'JWT Secret length: {len(settings.jwt_secret_key)}')
"

# Ensure secret key is consistent across restarts
# Store in environment variable, not generated randomly

# Check token expiration
python -c "
import jwt
from datetime import datetime
token = 'your_token_here'
decoded = jwt.decode(token, verify=False)
exp = datetime.fromtimestamp(decoded['exp'])
print(f'Token expires: {exp}')
"

# Regenerate token if needed
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

#### Issue: CORS errors in browser
```bash
Error: Access to fetch blocked by CORS policy
```

**Solution:**
```bash
# Configure CORS in FastAPI
# In stock_analysis_system/api/main.py:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# For development, allow all origins (not for production)
allow_origins=["*"]
```

## 🚀 Deployment Issues

### Docker Deployment Problems

#### Issue: Container fails to start in production
```bash
Error: Container exited with code 1
```

**Solution:**
```bash
# Check container logs
docker logs container_name

# Run container interactively for debugging
docker run -it --entrypoint /bin/bash your_image

# Check environment variables
docker exec container_name env

# Verify file permissions
docker exec container_name ls -la /app

# Check health status
docker inspect container_name | grep Health
```

#### Issue: Database connection fails in container
```bash
Error: could not translate host name "postgres" to address
```

**Solution:**
```bash
# Check Docker network
docker network ls
docker network inspect bridge

# Ensure containers are on same network
# In docker-compose.yml:
networks:
  app-network:
    driver: bridge

services:
  postgres:
    networks:
      - app-network
  api:
    networks:
      - app-network

# Use service names for internal communication
DATABASE_URL=postgresql://postgres:password@postgres:5432/stock_analysis
```

### SSL/TLS Issues

#### Issue: SSL certificate problems
```bash
Error: SSL certificate verify failed
```

**Solution:**
```bash
# Check certificate validity
openssl x509 -in certificate.crt -text -noout

# Renew Let's Encrypt certificate
sudo certbot renew

# Check certificate chain
openssl s_client -connect yourdomain.com:443 -showcerts

# Update certificate in nginx
sudo nginx -t
sudo systemctl reload nginx
```

## 📊 Monitoring and Logging

### Log Analysis

#### Issue: Finding specific errors in logs
```bash
Need to find specific error patterns
```

**Solution:**
```bash
# Search for errors
grep -i "error" logs/app.log

# Search for specific patterns
grep -E "(error|exception|failed)" logs/app.log

# Follow logs in real-time
tail -f logs/app.log

# Use structured log analysis
jq '.level == "ERROR"' logs/app.log

# Set up log rotation
sudo logrotate -f /etc/logrotate.d/stockapp
```

### Performance Monitoring

#### Issue: Identifying performance bottlenecks
```bash
Need to identify slow components
```

**Solution:**
```bash
# Monitor system resources
htop
iotop
nethogs

# Profile Python application
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(10)"

# Monitor database performance
sudo -u postgres psql stock_analysis
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

# Set up application monitoring
# Use tools like Prometheus, Grafana, or New Relic
```

## 🆘 Emergency Procedures

### System Recovery

#### Issue: Complete system failure
```bash
All services down, need emergency recovery
```

**Solution:**
```bash
# 1. Check system resources
df -h  # Disk space
free -h  # Memory
uptime  # System load

# 2. Restart core services
sudo systemctl restart postgresql
sudo systemctl restart redis-server
sudo systemctl restart nginx

# 3. Check Docker services
docker-compose down
docker-compose up -d

# 4. Restore from backup if needed
# See backup restoration procedures in deployment guide

# 5. Verify system health
curl http://localhost:8000/health
```

### Data Recovery

#### Issue: Data corruption or loss
```bash
Critical data corruption detected
```

**Solution:**
```bash
# 1. Stop all services immediately
docker-compose down
sudo systemctl stop stockapp-*

# 2. Assess damage
sudo -u postgres psql stock_analysis
\dt  # List tables
SELECT count(*) FROM stock_daily_data;

# 3. Restore from latest backup
# See backup procedures in deployment guide
gunzip -c backup_file.sql.gz | sudo -u postgres psql stock_analysis

# 4. Verify data integrity
python scripts/verify_data_integrity.py

# 5. Restart services
docker-compose up -d
```

## 📞 Getting Help

### Before Contacting Support

1. **Check this troubleshooting guide** for your specific issue
2. **Review system logs** for error messages
3. **Verify system requirements** are met
4. **Try basic solutions** like restarting services
5. **Document the issue** with steps to reproduce

### Information to Provide

When contacting support, include:

- **System information**: OS, Python version, Docker version
- **Error messages**: Complete error text and stack traces
- **Configuration**: Relevant parts of .env and config files
- **Steps to reproduce**: Detailed steps that cause the issue
- **Expected vs actual behavior**: What should happen vs what happens
- **Recent changes**: Any recent system or code changes

### Support Channels

- **GitHub Issues**: https://github.com/your-org/stock-analysis-system/issues
- **Documentation**: https://docs.stockanalysis.com
- **Community Forum**: https://community.stockanalysis.com
- **Email Support**: support@stockanalysis.com

## 📋 Quick Reference

### Common Commands

```bash
# System health check
curl http://localhost:8000/health

# View logs
tail -f logs/app.log

# Restart services
docker-compose restart

# Database connection test
psql -h localhost -U postgres -d stock_analysis

# Redis connection test
redis-cli ping

# Python environment check
python -c "import stock_analysis_system; print('OK')"

# Run tests
pytest tests/

# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
ps aux | grep -E "(python|postgres|redis)"
```

### Emergency Contacts

- **System Administrator**: admin@company.com
- **Database Administrator**: dba@company.com
- **DevOps Team**: devops@company.com
- **On-call Engineer**: +1-555-0123

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Maintained By**: Support Team