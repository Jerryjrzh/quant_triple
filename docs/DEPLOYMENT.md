# Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the Stock Analysis System in various environments, from development to production. The system supports multiple deployment strategies including Docker, traditional server deployment, and cloud platforms.

## 🏗️ Deployment Architecture Options

### 1. Development Deployment
- **Purpose**: Local development and testing
- **Components**: Docker Compose with all services
- **Scalability**: Single machine
- **Availability**: Development only

### 2. Production Deployment
- **Purpose**: Production workloads
- **Components**: Separate containers/services
- **Scalability**: Horizontal scaling
- **Availability**: High availability with redundancy

### 3. Cloud Deployment
- **Purpose**: Cloud-native deployment
- **Components**: Managed services where possible
- **Scalability**: Auto-scaling
- **Availability**: Multi-zone redundancy

## 🐳 Docker Deployment (Recommended)

### Prerequisites

```bash
# Required software
Docker >= 20.10.0
Docker Compose >= 2.0.0
Git >= 2.30.0

# System requirements
RAM: 4GB minimum, 8GB recommended
CPU: 2 cores minimum, 4 cores recommended
Storage: 20GB minimum, 50GB recommended
```

### Quick Start Deployment

```bash
# 1. Clone the repository
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system

# 2. Copy environment configuration
cp .env.example .env

# 3. Edit configuration (see Configuration section)
nano .env

# 4. Start all services
docker-compose up -d

# 5. Run database migrations
docker-compose exec api alembic upgrade head

# 6. Verify deployment
curl http://localhost:8000/health
```

### Docker Compose Configuration

#### Development Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: stock_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stock_analysis
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A stock_analysis_system.etl.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stock_analysis
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs

  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A stock_analysis_system.etl.celery_app beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stock_analysis
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

#### Production Configuration (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: stock_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - db_password
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  api:
    image: stock-analysis-system:latest
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stock_analysis
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
      - ENVIRONMENT=production
    ports:
      - "8000:8000"
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery:
    image: stock-analysis-system:latest
    command: celery -A stock_analysis_system.etl.celery_app worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@postgres:5432/stock_analysis
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/1
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - frontend

secrets:
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres_data:
  redis_data:
```

### Dockerfile Configuration

#### Backend Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Add local Python packages to PATH
ENV PATH=/home/app/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "stock_analysis_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
```

## ⚙️ Configuration Management

### Environment Variables

#### Core Configuration (`.env`)

```bash
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
API_VERSION=v1

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/stock_analysis
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_analysis
DB_USER=postgres
DB_PASSWORD=your-db-password

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
CELERY_TIMEZONE=Asia/Shanghai

# Data Sources
TUSHARE_TOKEN=your-tushare-token
AKSHARE_TIMEOUT=30
DATA_REQUESTS_PER_MINUTE=200

# Security
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/app.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
KEEP_ALIVE=2
```

#### Production Secrets

```bash
# Store in secure location (e.g., Docker secrets, Kubernetes secrets)
echo "your-secure-db-password" > secrets/db_password.txt
echo "your-secure-redis-password" > secrets/redis_password.txt
echo "your-secure-jwt-secret" > secrets/jwt_secret.txt
echo "your-tushare-token" > secrets/tushare_token.txt
```

### Configuration Validation

```python
# config/settings.py
from pydantic import BaseSettings, validator
from typing import Optional

class Settings(BaseSettings):
    environment: str = "development"
    debug: bool = False
    secret_key: str
    
    # Database
    database_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # Redis
    redis_url: str
    redis_max_connections: int = 100
    
    @validator('secret_key')
    def secret_key_must_be_strong(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters')
        return v
    
    @validator('database_url')
    def database_url_must_be_postgresql(cls, v):
        if not v.startswith('postgresql://'):
            raise ValueError('Database URL must be PostgreSQL')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

## 🚀 Production Deployment

### Server Requirements

#### Minimum Requirements
- **CPU**: 4 cores (2.0 GHz+)
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **Network**: 100 Mbps
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

#### Recommended Requirements
- **CPU**: 8 cores (2.5 GHz+)
- **RAM**: 16 GB
- **Storage**: 500 GB SSD
- **Network**: 1 Gbps
- **OS**: Ubuntu 22.04 LTS

### Traditional Server Deployment

#### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    postgresql-15 \
    redis-server \
    nginx \
    supervisor \
    git \
    curl \
    htop

# Create application user
sudo useradd -m -s /bin/bash stockapp
sudo usermod -aG sudo stockapp
```

#### 2. Database Setup

```bash
# Configure PostgreSQL
sudo -u postgres psql << EOF
CREATE DATABASE stock_analysis;
CREATE USER stockapp WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE stock_analysis TO stockapp;
ALTER USER stockapp CREATEDB;
\q
EOF

# Configure Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Secure Redis
echo "requirepass your_redis_password" | sudo tee -a /etc/redis/redis.conf
sudo systemctl restart redis-server
```

#### 3. Application Deployment

```bash
# Switch to application user
sudo su - stockapp

# Clone repository
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Run database migrations
alembic upgrade head

# Test application
python -m pytest tests/

# Create systemd service files
sudo tee /etc/systemd/system/stockapp-api.service << EOF
[Unit]
Description=Stock Analysis API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=stockapp
Group=stockapp
WorkingDirectory=/home/stockapp/stock-analysis-system
Environment=PATH=/home/stockapp/stock-analysis-system/venv/bin
ExecStart=/home/stockapp/stock-analysis-system/venv/bin/uvicorn stock_analysis_system.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/stockapp-celery.service << EOF
[Unit]
Description=Stock Analysis Celery Worker
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=stockapp
Group=stockapp
WorkingDirectory=/home/stockapp/stock-analysis-system
Environment=PATH=/home/stockapp/stock-analysis-system/venv/bin
ExecStart=/home/stockapp/stock-analysis-system/venv/bin/celery -A stock_analysis_system.etl.celery_app worker --loglevel=info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable stockapp-api stockapp-celery
sudo systemctl start stockapp-api stockapp-celery
```

#### 4. Nginx Configuration

```nginx
# /etc/nginx/sites-available/stockapp
server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # API Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health Check
    location /health {
        proxy_pass http://127.0.0.1:8000;
        access_log off;
    }
    
    # Static Files
    location / {
        root /home/stockapp/stock-analysis-system/frontend/build;
        try_files $uri $uri/ /index.html;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
}
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Architecture Overview

```mermaid
graph TB
    subgraph "AWS Cloud"
        subgraph "VPC"
            subgraph "Public Subnets"
                ALB[Application Load Balancer]
                NAT[NAT Gateway]
            end
            
            subgraph "Private Subnets"
                ECS[ECS Fargate Cluster]
                RDS[(RDS PostgreSQL)]
                REDIS[(ElastiCache Redis)]
            end
        end
        
        subgraph "Services"
            ECR[ECR Container Registry]
            S3[S3 Storage]
            LOGS[CloudWatch Logs]
            SECRETS[Secrets Manager]
        end
    end
    
    USERS[Users] --> ALB
    ALB --> ECS
    ECS --> RDS
    ECS --> REDIS
    ECS --> S3
    ECS --> LOGS
    ECS --> SECRETS
```

#### ECS Task Definition

```json
{
    "family": "stock-analysis-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "api",
            "image": "account.dkr.ecr.region.amazonaws.com/stock-analysis:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "ENVIRONMENT",
                    "value": "production"
                }
            ],
            "secrets": [
                {
                    "name": "DATABASE_URL",
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"
                },
                {
                    "name": "REDIS_URL", 
                    "valueFrom": "arn:aws:secretsmanager:region:account:secret:redis-url"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/stock-analysis",
                    "awslogs-region": "us-west-2",
                    "awslogs-stream-prefix": "api"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
```

#### Terraform Configuration

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "stock-analysis-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
  }
}

# RDS
resource "aws_db_instance" "postgres" {
  identifier = "stock-analysis-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp2"
  storage_encrypted    = true
  
  db_name  = "stock_analysis"
  username = "postgres"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "stock-analysis-final-snapshot"
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "stock-analysis-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "stock-analysis-redis"
  description                = "Redis cluster for stock analysis"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = var.redis_password
  
  tags = {
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "stock-analysis"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Environment = var.environment
  }
}
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: stock-analysis
  labels:
    name: stock-analysis

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: stock-analysis-config
  namespace: stock-analysis
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_VERSION: "v1"
  CELERY_TIMEZONE: "Asia/Shanghai"
```

#### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: stock-analysis-secrets
  namespace: stock-analysis
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  TUSHARE_TOKEN: <base64-encoded-tushare-token>
```

#### API Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-analysis-api
  namespace: stock-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stock-analysis-api
  template:
    metadata:
      labels:
        app: stock-analysis-api
    spec:
      containers:
      - name: api
        image: stock-analysis:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: stock-analysis-config
        - secretRef:
            name: stock-analysis-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: stock-analysis-api-service
  namespace: stock-analysis
spec:
  selector:
    app: stock-analysis-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: stock-analysis-ingress
  namespace: stock-analysis
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.stockanalysis.com
    secretName: stock-analysis-tls
  rules:
  - host: api.stockanalysis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: stock-analysis-api-service
            port:
              number: 80
```

#### Celery Worker Deployment

```yaml
# celery-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-analysis-celery
  namespace: stock-analysis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: stock-analysis-celery
  template:
    metadata:
      labels:
        app: stock-analysis-celery
    spec:
      containers:
      - name: celery
        image: stock-analysis:latest
        command: ["celery"]
        args: ["-A", "stock_analysis_system.etl.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
        envFrom:
        - configMapRef:
            name: stock-analysis-config
        - secretRef:
            name: stock-analysis-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 📊 Monitoring and Observability

### Health Checks

```python
# health_check.py
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
from datetime import datetime

async def check_database():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("SELECT 1")
        await conn.close()
        return {"status": "healthy", "response_time": "< 10ms"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_redis():
    try:
        r = redis.from_url(REDIS_URL)
        await r.ping()
        await r.close()
        return {"status": "healthy", "response_time": "< 5ms"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_external_apis():
    results = {}
    async with aiohttp.ClientSession() as session:
        # Check Tushare
        try:
            async with session.get("https://api.tushare.pro", timeout=5) as resp:
                results["tushare"] = {"status": "healthy", "response_code": resp.status}
        except Exception as e:
            results["tushare"] = {"status": "unhealthy", "error": str(e)}
    
    return results
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    # Create formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('/app/logs/app.log')
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')
CELERY_TASKS = Counter('celery_tasks_total', 'Total Celery tasks', ['task_name', 'status'])

def start_metrics_server():
    start_http_server(9090)

@REQUEST_DURATION.time()
def track_request_duration():
    # Decorator for timing requests
    pass
```

## 🔒 Security Considerations

### SSL/TLS Configuration

```bash
# Generate SSL certificate with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Security Headers

```nginx
# Security headers in Nginx
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
```

## 🔄 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="stock_analysis"

# Create backup
pg_dump -h localhost -U postgres -d $DB_NAME | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +7 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/db_backup_$DATE.sql.gz s3://your-backup-bucket/database/
```

### Application Data Backup

```bash
#!/bin/bash
# app_backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup logs
tar -czf $BACKUP_DIR/logs_backup_$DATE.tar.gz /app/logs/

# Backup configuration
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz /app/config/

# Backup uploaded files
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz /app/data/
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
sudo systemctl stop stockapp-api stockapp-celery

# Restore database
gunzip -c $BACKUP_FILE | psql -h localhost -U postgres -d stock_analysis

# Start services
sudo systemctl start stockapp-api stockapp-celery

echo "Restore completed"
```

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] Environment configuration reviewed
- [ ] Secrets properly configured
- [ ] Database migrations tested
- [ ] SSL certificates obtained
- [ ] Firewall rules configured
- [ ] Monitoring setup completed
- [ ] Backup procedures tested
- [ ] Load testing performed
- [ ] Security scan completed

### Deployment

- [ ] Services deployed in correct order
- [ ] Health checks passing
- [ ] Database connectivity verified
- [ ] Redis connectivity verified
- [ ] External API access verified
- [ ] Frontend-backend integration tested
- [ ] SSL/TLS working correctly
- [ ] Monitoring alerts configured

### Post-Deployment

- [ ] Application functionality verified
- [ ] Performance metrics within acceptable range
- [ ] Log aggregation working
- [ ] Backup procedures scheduled
- [ ] Documentation updated
- [ ] Team notified of deployment
- [ ] Rollback plan documented

## 🚨 Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT version();"

# Check connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis-server
redis-cli ping

# Check memory usage
redis-cli info memory
```

#### Application Not Starting
```bash
# Check logs
sudo journalctl -u stockapp-api -f
tail -f /app/logs/app.log

# Check port usage
sudo netstat -tlnp | grep :8000
```

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
valgrind --tool=memcheck python app.py
```

### Performance Optimization

```bash
# Database optimization
sudo -u postgres psql stock_analysis << EOF
ANALYZE;
REINDEX DATABASE stock_analysis;
VACUUM ANALYZE;
EOF

# Redis optimization
redis-cli config set maxmemory-policy allkeys-lru
redis-cli config rewrite
```

## 📋 Summary

This deployment guide provides comprehensive instructions for deploying the Stock Analysis System across various environments:

- ✅ **Docker Deployment**: Containerized deployment with Docker Compose
- ✅ **Production Deployment**: Traditional server deployment with systemd
- ✅ **Cloud Deployment**: AWS and Kubernetes deployment options
- ✅ **Security**: SSL/TLS, firewall, and security headers configuration
- ✅ **Monitoring**: Health checks, logging, and metrics collection
- ✅ **Backup**: Database and application data backup procedures
- ✅ **Troubleshooting**: Common issues and performance optimization

The deployment strategies are designed to be scalable, secure, and maintainable, supporting both development and production workloads.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Maintained By**: DevOps Team