# Stock Analysis System - Operational Procedures

## Table of Contents

1. [Overview](#overview)
2. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
3. [Capacity Planning and Scaling](#capacity-planning-and-scaling)
4. [Incident Response and Escalation](#incident-response-and-escalation)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Security Operations](#security-operations)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Emergency Contacts](#emergency-contacts)

## Overview

This document outlines the operational procedures for the Stock Analysis System, providing step-by-step guidance for system administrators, DevOps engineers, and on-call personnel to maintain, monitor, and troubleshoot the production environment.

### System Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Nginx Proxy   │────│    Frontend     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   API Gateway   │
                       └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ API Server  │ │ API Server  │ │ API Server  │
        └─────────────┘ └─────────────┘ └─────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
                ┌───────────────────────────────────┐
                │         Shared Services           │
                │  ┌─────────────┐ ┌─────────────┐  │
                │  │ PostgreSQL  │ │    Redis    │  │
                │  └─────────────┘ └─────────────┘  │
                │  ┌─────────────┐ ┌─────────────┐  │
                │  │   Celery    │ │ Monitoring  │  │
                │  │   Workers   │ │   Stack     │  │
                │  └─────────────┘ └─────────────┘  │
                └───────────────────────────────────┘
```

## Backup and Disaster Recovery

### 1. Database Backup Procedures

#### Daily Automated Backups

**Frequency**: Daily at 2:00 AM UTC  
**Retention**: 30 days for daily, 12 weeks for weekly, 12 months for monthly

```bash
#!/bin/bash
# Database backup script

BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="stock_analysis"
DB_USER="stock_analysis"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup
pg_dump -h postgresql-service -U "$DB_USER" -d "$DB_NAME" \
    --verbose --clean --no-owner --no-privileges \
    --format=custom > "$BACKUP_DIR/backup_${DATE}.dump"

# Compress backup
gzip "$BACKUP_DIR/backup_${DATE}.dump"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/backup_${DATE}.dump.gz" \
    s3://stock-analysis-backups/postgresql/daily/

# Clean up local backups older than 7 days
find "$BACKUP_DIR" -name "backup_*.dump.gz" -mtime +7 -delete

# Verify backup integrity
pg_restore --list "$BACKUP_DIR/backup_${DATE}.dump.gz" > /dev/null
if [ $? -eq 0 ]; then
    echo "Backup verification successful"
else
    echo "Backup verification failed" | mail -s "Backup Alert" ops@company.com
fi
```

#### Database Restore Procedure

```bash
#!/bin/bash
# Database restore script

BACKUP_FILE="$1"
DB_NAME="stock_analysis"
DB_USER="stock_analysis"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application services
kubectl scale deployment stock-analysis-api --replicas=0 -n stock-analysis-system
kubectl scale deployment celery-worker --replicas=0 -n stock-analysis-system

# Drop and recreate database
psql -h postgresql-service -U "$DB_USER" -c "DROP DATABASE IF EXISTS ${DB_NAME};"
psql -h postgresql-service -U "$DB_USER" -c "CREATE DATABASE ${DB_NAME};"

# Restore from backup
pg_restore -h postgresql-service -U "$DB_USER" -d "$DB_NAME" \
    --verbose --clean --no-owner --no-privileges "$BACKUP_FILE"

# Run database migrations
kubectl run migration-job --image=stock-analysis-system:latest \
    --rm -i --restart=Never -- alembic upgrade head

# Restart application services
kubectl scale deployment stock-analysis-api --replicas=3 -n stock-analysis-system
kubectl scale deployment celery-worker --replicas=3 -n stock-analysis-system

echo "Database restore completed"
```

### 2. Application Data Backup

#### Configuration Backup

```bash
#!/bin/bash
# Backup Kubernetes configurations

BACKUP_DIR="/backups/k8s-configs"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup all Kubernetes resources
kubectl get all,configmaps,secrets,pvc,ingress -n stock-analysis-system \
    -o yaml > "$BACKUP_DIR/k8s-backup-${DATE}.yaml"

# Backup custom resources
kubectl get crd -o yaml > "$BACKUP_DIR/crd-backup-${DATE}.yaml"

# Upload to cloud storage
tar -czf "$BACKUP_DIR/k8s-backup-${DATE}.tar.gz" "$BACKUP_DIR"/*.yaml
aws s3 cp "$BACKUP_DIR/k8s-backup-${DATE}.tar.gz" \
    s3://stock-analysis-backups/k8s-configs/
```

### 3. Disaster Recovery Plan

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 1 hour

#### DR Site Setup

1. **Primary Site Failure Detection**
   ```bash
   # Health check script
   #!/bin/bash
   PRIMARY_URL="https://api.stockanalysis.com/health"
   
   if ! curl -f "$PRIMARY_URL" --max-time 30; then
       echo "Primary site is down, initiating DR procedures"
       # Trigger DR automation
       ./scripts/activate-dr-site.sh
   fi
   ```

2. **DR Site Activation**
   ```bash
   #!/bin/bash
   # DR site activation script
   
   # Update DNS to point to DR site
   aws route53 change-resource-record-sets \
       --hosted-zone-id Z123456789 \
       --change-batch file://dns-failover.json
   
   # Scale up DR environment
   kubectl scale deployment stock-analysis-api --replicas=5 -n stock-analysis-system-dr
   kubectl scale deployment celery-worker --replicas=5 -n stock-analysis-system-dr
   
   # Restore latest backup
   ./scripts/restore-database.sh /backups/latest-backup.dump.gz
   
   # Verify DR site health
   sleep 60
   curl -f https://dr.stockanalysis.com/health
   ```

## Capacity Planning and Scaling

### 1. Resource Monitoring

#### Key Metrics to Monitor

- **CPU Utilization**: Target < 70% average
- **Memory Usage**: Target < 80% average
- **Disk Usage**: Target < 85% for data volumes
- **Network I/O**: Monitor for bottlenecks
- **Database Connections**: Target < 80% of max connections
- **API Response Time**: Target < 500ms for 95th percentile

#### Monitoring Queries

```promql
# CPU utilization by deployment
avg(rate(container_cpu_usage_seconds_total[5m])) by (pod) * 100

# Memory utilization by deployment
avg(container_memory_usage_bytes / container_spec_memory_limit_bytes) by (pod) * 100

# API response time 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Database connection count
pg_stat_database_numbackends{datname="stock_analysis"}
```

### 2. Scaling Procedures

#### Horizontal Pod Autoscaling (HPA)

HPA is configured for automatic scaling based on CPU and memory metrics:

```yaml
# API Server HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: stock-analysis-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stock-analysis-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Manual Scaling

```bash
# Scale API servers
kubectl scale deployment stock-analysis-api --replicas=5 -n stock-analysis-system

# Scale Celery workers
kubectl scale deployment celery-worker --replicas=8 -n stock-analysis-system

# Scale frontend
kubectl scale deployment stock-analysis-frontend --replicas=4 -n stock-analysis-system
```

#### Database Scaling

```bash
# Scale PostgreSQL (if using a scalable solution like PostgreSQL Operator)
kubectl patch postgresql stock-analysis-db \
    --type='merge' -p='{"spec":{"numberOfInstances":3}}'

# Scale Redis (if using Redis Cluster)
kubectl patch rediscluster stock-analysis-redis \
    --type='merge' -p='{"spec":{"nodes":6}}'
```

### 3. Capacity Planning Guidelines

#### Growth Projections

- **User Growth**: Plan for 20% monthly growth
- **Data Growth**: Plan for 15% monthly growth in database size
- **API Requests**: Plan for 25% monthly growth in request volume

#### Resource Allocation

| Component | Current | 6 Months | 12 Months |
|-----------|---------|----------|-----------|
| API Pods | 3 | 5 | 8 |
| Worker Pods | 3 | 6 | 10 |
| Database CPU | 2 cores | 4 cores | 8 cores |
| Database Memory | 8GB | 16GB | 32GB |
| Storage | 100GB | 200GB | 500GB |

## Incident Response and Escalation

### 1. Incident Classification

#### Severity Levels

**P0 - Critical**
- Complete system outage
- Data corruption or loss
- Security breach
- Response Time: 15 minutes

**P1 - High**
- Partial system outage
- Significant performance degradation
- Critical feature unavailable
- Response Time: 1 hour

**P2 - Medium**
- Minor performance issues
- Non-critical feature unavailable
- Response Time: 4 hours

**P3 - Low**
- Cosmetic issues
- Enhancement requests
- Response Time: 24 hours

### 2. Incident Response Procedures

#### Initial Response (First 15 minutes)

1. **Acknowledge the incident**
   ```bash
   # Update incident status
   curl -X POST https://api.pagerduty.com/incidents/{id}/acknowledge \
        -H "Authorization: Token token=YOUR_API_KEY"
   ```

2. **Assess the situation**
   - Check monitoring dashboards
   - Review recent deployments
   - Check system logs

3. **Communicate**
   - Update status page
   - Notify stakeholders
   - Create incident channel

#### Investigation and Resolution

1. **Gather information**
   ```bash
   # Check pod status
   kubectl get pods -n stock-analysis-system
   
   # Check recent events
   kubectl get events -n stock-analysis-system --sort-by='.lastTimestamp'
   
   # Check logs
   kubectl logs -f deployment/stock-analysis-api -n stock-analysis-system
   ```

2. **Implement immediate fixes**
   - Restart failed services
   - Scale resources if needed
   - Apply hotfixes

3. **Monitor recovery**
   - Verify system health
   - Monitor key metrics
   - Confirm user impact resolution

#### Post-Incident Activities

1. **Document the incident**
2. **Conduct post-mortem**
3. **Implement preventive measures**
4. **Update runbooks**

### 3. Escalation Matrix

| Role | Contact | Escalation Time |
|------|---------|----------------|
| On-Call Engineer | Slack: @oncall | Immediate |
| DevOps Lead | Phone: +1-xxx-xxx-xxxx | 30 minutes |
| Engineering Manager | Email: eng-mgr@company.com | 1 hour |
| CTO | Phone: +1-xxx-xxx-xxxx | 2 hours (P0 only) |

### 4. Communication Templates

#### Initial Incident Notification

```
🚨 INCIDENT ALERT - P{severity}

Title: {incident_title}
Status: Investigating
Impact: {user_impact_description}
Started: {timestamp}

We are investigating reports of {issue_description}. 
Updates will be provided every 30 minutes.

Incident Commander: {name}
```

#### Resolution Notification

```
✅ INCIDENT RESOLVED - P{severity}

Title: {incident_title}
Status: Resolved
Duration: {duration}
Root Cause: {brief_cause}

The issue has been resolved. All systems are operating normally.
A detailed post-mortem will be published within 48 hours.
```

## Monitoring and Alerting

### 1. Alert Rules

#### Critical Alerts

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value }} errors per second"

# Database down
- alert: DatabaseDown
  expr: up{job="postgresql"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "PostgreSQL database is down"

# High memory usage
- alert: HighMemoryUsage
  expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High memory usage detected"
```

### 2. Dashboard Configuration

#### Key Dashboards

1. **System Overview Dashboard**
   - Overall system health
   - Request rates and response times
   - Error rates
   - Resource utilization

2. **Application Performance Dashboard**
   - API endpoint performance
   - Database query performance
   - Cache hit rates
   - Background job status

3. **Infrastructure Dashboard**
   - Kubernetes cluster health
   - Node resource usage
   - Network metrics
   - Storage metrics

### 3. Log Management

#### Log Aggregation

```yaml
# Fluentd configuration for log collection
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type kubernetes_metadata
      @id kubernetes_metadata
    </source>
    
    <filter kubernetes.**>
      @type parser
      key_name log
      <parse>
        @type json
        time_key timestamp
        time_format %Y-%m-%dT%H:%M:%S.%L%z
      </parse>
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch-service
      port 9200
      index_name kubernetes-logs
    </match>
```

## Maintenance Procedures

### 1. Scheduled Maintenance

#### Monthly Maintenance Window

**Schedule**: First Sunday of each month, 2:00 AM - 6:00 AM UTC

**Pre-maintenance Checklist**:
- [ ] Notify users 48 hours in advance
- [ ] Create maintenance branch
- [ ] Prepare rollback plan
- [ ] Backup all critical data
- [ ] Test procedures in staging

**Maintenance Tasks**:
1. Apply security patches
2. Update dependencies
3. Database maintenance (VACUUM, REINDEX)
4. Certificate renewals
5. Log rotation and cleanup

#### Database Maintenance

```sql
-- Monthly database maintenance
VACUUM ANALYZE;
REINDEX DATABASE stock_analysis;

-- Update statistics
ANALYZE;

-- Check for unused indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
```

### 2. Security Updates

#### Patch Management Process

1. **Vulnerability Assessment**
   ```bash
   # Scan for vulnerabilities
   trivy image stock-analysis-system:latest
   ```

2. **Patch Testing**
   - Apply patches in staging environment
   - Run automated tests
   - Perform manual verification

3. **Production Deployment**
   - Deploy during maintenance window
   - Monitor for issues
   - Rollback if necessary

## Security Operations

### 1. Security Monitoring

#### Security Alerts

```yaml
# Failed authentication attempts
- alert: HighFailedLogins
  expr: rate(auth_failed_total[5m]) > 10
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High number of failed login attempts"

# Suspicious API activity
- alert: SuspiciousAPIActivity
  expr: rate(http_requests_total{status="403"}[5m]) > 5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Suspicious API activity detected"
```

### 2. Access Control

#### User Access Review

**Frequency**: Quarterly

**Process**:
1. Review all user accounts
2. Verify access permissions
3. Remove inactive accounts
4. Update role assignments

#### Certificate Management

```bash
# Check certificate expiration
kubectl get certificates -n stock-analysis-system

# Renew certificates (if using cert-manager)
kubectl annotate certificate stock-analysis-tls \
    cert-manager.io/issue-temporary-certificate="true" \
    -n stock-analysis-system
```

## Performance Optimization

### 1. Database Optimization

#### Query Performance Monitoring

```sql
-- Find slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
```

#### Index Optimization

```sql
-- Create missing indexes
CREATE INDEX CONCURRENTLY idx_stocks_symbol_date 
ON stock_prices (symbol, date);

-- Remove unused indexes
DROP INDEX IF EXISTS idx_unused_index;
```

### 2. Application Performance

#### Cache Optimization

```python
# Redis cache monitoring
import redis

r = redis.Redis(host='redis-service', port=6379)

# Check cache hit rate
info = r.info()
hit_rate = info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses'])
print(f"Cache hit rate: {hit_rate:.2%}")
```

#### API Performance Tuning

```python
# Enable connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

## Troubleshooting Guide

### 1. Common Issues

#### API Server Not Responding

**Symptoms**: HTTP 503 errors, timeouts

**Investigation Steps**:
```bash
# Check pod status
kubectl get pods -l app=stock-analysis-api -n stock-analysis-system

# Check pod logs
kubectl logs -f deployment/stock-analysis-api -n stock-analysis-system

# Check resource usage
kubectl top pods -n stock-analysis-system
```

**Resolution**:
1. Restart pods if memory/CPU issues
2. Scale up if high load
3. Check database connectivity

#### Database Connection Issues

**Symptoms**: Connection timeouts, pool exhaustion

**Investigation Steps**:
```bash
# Check database pod
kubectl get pods -l app=postgresql -n stock-analysis-system

# Check database logs
kubectl logs -f deployment/postgresql -n stock-analysis-system

# Check connections
kubectl exec -it postgresql-pod -- psql -U stock_analysis -c "SELECT count(*) FROM pg_stat_activity;"
```

**Resolution**:
1. Restart database pod
2. Increase connection pool size
3. Check for long-running queries

#### High Memory Usage

**Symptoms**: OOMKilled pods, slow performance

**Investigation Steps**:
```bash
# Check memory usage
kubectl top pods -n stock-analysis-system

# Check memory limits
kubectl describe pod pod-name -n stock-analysis-system
```

**Resolution**:
1. Increase memory limits
2. Optimize application code
3. Add more replicas

### 2. Diagnostic Commands

#### System Health Check

```bash
#!/bin/bash
# Comprehensive health check script

echo "=== Kubernetes Cluster Health ==="
kubectl cluster-info
kubectl get nodes

echo "=== Namespace Resources ==="
kubectl get all -n stock-analysis-system

echo "=== Pod Status ==="
kubectl get pods -n stock-analysis-system -o wide

echo "=== Service Endpoints ==="
kubectl get endpoints -n stock-analysis-system

echo "=== Recent Events ==="
kubectl get events -n stock-analysis-system --sort-by='.lastTimestamp' | tail -20

echo "=== Resource Usage ==="
kubectl top pods -n stock-analysis-system
kubectl top nodes
```

#### Performance Analysis

```bash
#!/bin/bash
# Performance analysis script

echo "=== API Response Times ==="
curl -w "@curl-format.txt" -o /dev/null -s "http://api.stockanalysis.com/health"

echo "=== Database Performance ==="
kubectl exec -it postgresql-pod -- psql -U stock_analysis -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 5;"

echo "=== Cache Performance ==="
kubectl exec -it redis-pod -- redis-cli info stats | grep -E "(keyspace_hits|keyspace_misses)"
```

## Emergency Contacts

### Primary Contacts

| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| On-Call Engineer | Current Rotation | +1-xxx-xxx-xxxx | oncall@company.com | @oncall |
| DevOps Lead | John Smith | +1-xxx-xxx-xxxx | john.smith@company.com | @jsmith |
| Database Admin | Jane Doe | +1-xxx-xxx-xxxx | jane.doe@company.com | @jdoe |
| Security Lead | Bob Wilson | +1-xxx-xxx-xxxx | bob.wilson@company.com | @bwilson |

### Escalation Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Engineering Manager | Alice Johnson | +1-xxx-xxx-xxxx | alice.johnson@company.com |
| VP Engineering | Mike Brown | +1-xxx-xxx-xxxx | mike.brown@company.com |
| CTO | Sarah Davis | +1-xxx-xxx-xxxx | sarah.davis@company.com |

### External Vendors

| Service | Contact | Phone | Support Portal |
|---------|---------|-------|----------------|
| Cloud Provider | AWS Support | +1-xxx-xxx-xxxx | https://console.aws.amazon.com/support/ |
| Monitoring | DataDog Support | +1-xxx-xxx-xxxx | https://help.datadoghq.com/ |
| CDN | CloudFlare Support | +1-xxx-xxx-xxxx | https://support.cloudflare.com/ |

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-20  
**Next Review**: 2024-04-20  
**Owner**: DevOps Team