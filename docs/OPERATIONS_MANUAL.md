# 爬虫接口集成系统运维手册

## 目录

1. [系统概述](#系统概述)
2. [部署架构](#部署架构)
3. [日常运维](#日常运维)
4. [监控告警](#监控告警)
5. [故障处理](#故障处理)
6. [备份恢复](#备份恢复)
7. [性能优化](#性能优化)
8. [安全管理](#安全管理)

## 系统概述

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   负载均衡器    │    │   API网关       │    │   应用服务器    │
│   (Nginx)       │───▶│   (FastAPI)     │───▶│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   监控系统      │    │   缓存服务      │    │   数据库        │
│ (Prometheus)    │    │   (Redis)       │    │ (PostgreSQL)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

- **API服务**: FastAPI应用，提供RESTful接口
- **数据库**: PostgreSQL，存储业务数据
- **缓存**: Redis，提供高速缓存
- **消息队列**: Celery + Redis，处理异步任务
- **监控**: Prometheus + Grafana，系统监控
- **日志**: ELK Stack，日志收集和分析

## 部署架构

### 生产环境部署

#### Docker Compose部署

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

#### Kubernetes部署

```bash
# 部署应用
kubectl apply -f k8s/

# 查看部署状态
kubectl get pods -n stock-analysis

# 查看服务
kubectl get svc -n stock-analysis

# 查看日志
kubectl logs -f deployment/stock-analysis-api -n stock-analysis
```

### 环境配置

#### 生产环境变量

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@db-host:5432/stockdb
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis配置
REDIS_URL=redis://redis-host:6379/0
REDIS_MAX_CONNECTIONS=50

# API配置
SECRET_KEY=your-secret-key
DEBUG=false
LOG_LEVEL=INFO

# 数据源配置
TUSHARE_API_KEY=your-tushare-key
AKSHARE_API_KEY=your-akshare-key
WIND_API_KEY=your-wind-key

# 监控配置
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```## 日常
运维

### 服务启动和停止

#### 启动服务

```bash
# Docker环境
docker-compose up -d

# Kubernetes环境
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 直接启动
python start_server.py --env production
```

#### 停止服务

```bash
# Docker环境
docker-compose down

# Kubernetes环境
kubectl delete -f k8s/

# 直接停止
pkill -f "python start_server.py"
```

### 健康检查

#### 自动健康检查

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ 服务正常"
    exit 0
else
    echo "❌ 服务异常: HTTP $RESPONSE"
    exit 1
fi
```

#### 手动健康检查

```bash
# 检查API服务
curl http://localhost:8000/health

# 检查数据库连接
psql -h localhost -U username -d stockdb -c "SELECT 1;"

# 检查Redis连接
redis-cli ping

# 检查系统资源
htop
df -h
free -h
```

### 日志管理

#### 日志位置

```
/var/log/stock-analysis/
├── api.log              # API服务日志
├── celery.log           # Celery任务日志
├── error.log            # 错误日志
├── access.log           # 访问日志
└── system.log           # 系统日志
```

#### 日志轮转配置

```bash
# /etc/logrotate.d/stock-analysis
/var/log/stock-analysis/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 app app
    postrotate
        systemctl reload stock-analysis
    endscript
}
```

#### 日志查看命令

```bash
# 查看实时日志
tail -f /var/log/stock-analysis/api.log

# 查看错误日志
grep ERROR /var/log/stock-analysis/api.log

# 查看最近1小时的日志
journalctl --since "1 hour ago" -u stock-analysis

# 使用ELK查看日志
# 访问 http://kibana-host:5601
```

### 数据库维护

#### 数据库备份

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backup/database"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="stockdb_backup_$DATE.sql"

# 创建备份
pg_dump -h localhost -U username stockdb > $BACKUP_DIR/$BACKUP_FILE

# 压缩备份
gzip $BACKUP_DIR/$BACKUP_FILE

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "数据库备份完成: $BACKUP_FILE.gz"
```

#### 数据库恢复

```bash
# 恢复数据库
gunzip -c /backup/database/stockdb_backup_20240101_120000.sql.gz | \
psql -h localhost -U username stockdb
```

#### 数据库维护

```sql
-- 更新统计信息
ANALYZE;

-- 重建索引
REINDEX DATABASE stockdb;

-- 清理死元组
VACUUM FULL;

-- 检查数据库大小
SELECT pg_size_pretty(pg_database_size('stockdb'));

-- 检查表大小
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 监控告警

### Prometheus配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'stock-analysis-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['localhost:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 告警规则

```yaml
# alert_rules.yml
groups:
- name: stock-analysis-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time"
      description: "95th percentile response time is {{ $value }}s"

  - alert: DatabaseConnectionHigh
    expr: pg_stat_activity_count > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database connections"
      description: "Database has {{ $value }} active connections"

  - alert: RedisMemoryHigh
    expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Redis memory usage high"
      description: "Redis memory usage is {{ $value | humanizePercentage }}"
```### Gr
afana仪表板

#### 系统概览仪表板

```json
{
  "dashboard": {
    "title": "Stock Analysis System Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_activity_count",
            "legendFormat": "Active Connections"
          }
        ]
      }
    ]
  }
}
```

### 告警通知配置

#### Alertmanager配置

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@stockanalysis.com'
  smtp_auth_username: 'alerts@stockanalysis.com'
  smtp_auth_password: 'your-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@stockanalysis.com'
    subject: '[ALERT] {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  
  webhook_configs:
  - url: 'http://slack-webhook-url'
    send_resolved: true
```

## 故障处理

### 常见故障处理流程

#### 1. 服务无响应

```bash
# 检查进程状态
ps aux | grep python

# 检查端口占用
netstat -tlnp | grep 8000

# 检查系统资源
top
df -h

# 重启服务
systemctl restart stock-analysis
```

#### 2. 数据库连接失败

```bash
# 检查数据库状态
systemctl status postgresql

# 检查连接数
psql -c "SELECT count(*) FROM pg_stat_activity;"

# 检查慢查询
psql -c "SELECT query, query_start FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# 重启数据库
systemctl restart postgresql
```

#### 3. Redis缓存问题

```bash
# 检查Redis状态
redis-cli ping

# 检查内存使用
redis-cli info memory

# 清理缓存
redis-cli flushdb

# 重启Redis
systemctl restart redis
```

#### 4. 高CPU使用率

```bash
# 查看CPU使用情况
htop

# 查看进程CPU使用
ps aux --sort=-%cpu | head -10

# 查看系统负载
uptime

# 分析性能瓶颈
perf top -p $(pgrep python)
```

### 故障恢复检查清单

- [ ] 服务进程正常运行
- [ ] 数据库连接正常
- [ ] Redis缓存可用
- [ ] API接口响应正常
- [ ] 监控指标恢复正常
- [ ] 日志无错误信息
- [ ] 用户访问正常

## 备份恢复

### 自动备份脚本

```bash
#!/bin/bash
# auto_backup.sh

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)

# 数据库备份
echo "开始数据库备份..."
pg_dump -h localhost -U username stockdb | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# 配置文件备份
echo "备份配置文件..."
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz /etc/stock-analysis/

# 日志备份
echo "备份日志文件..."
tar -czf $BACKUP_DIR/logs_backup_$DATE.tar.gz /var/log/stock-analysis/

# 清理旧备份
find $BACKUP_DIR -name "*backup*" -mtime +7 -delete

echo "备份完成: $DATE"
```

### 灾难恢复流程

1. **评估损坏程度**
   - 确定故障范围
   - 评估数据丢失情况
   - 制定恢复计划

2. **恢复基础设施**
   - 重建服务器环境
   - 安装必要软件
   - 配置网络和安全

3. **恢复数据**
   - 恢复数据库
   - 恢复配置文件
   - 验证数据完整性

4. **恢复服务**
   - 启动应用服务
   - 验证功能正常
   - 恢复监控告警

5. **验证和测试**
   - 执行功能测试
   - 验证数据一致性
   - 确认服务可用性

## 性能优化

### 数据库优化

```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_stock_data_symbol_date 
ON stock_data(symbol, trade_date);

-- 分区表
CREATE TABLE stock_data_2024 PARTITION OF stock_data
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- 连接池配置
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### 应用优化

```python
# 连接池配置
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30

# 缓存配置
CACHE_TTL = {
    'realtime_data': 60,      # 1分钟
    'daily_data': 3600,       # 1小时
    'historical_data': 86400  # 1天
}

# 异步处理
CELERY_WORKER_CONCURRENCY = 4
CELERY_TASK_SOFT_TIME_LIMIT = 300
```

### 系统优化

```bash
# 内核参数优化
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
echo 'fs.file-max = 100000' >> /etc/sysctl.conf

# 应用系统参数
sysctl -p

# 文件描述符限制
echo '* soft nofile 65535' >> /etc/security/limits.conf
echo '* hard nofile 65535' >> /etc/security/limits.conf
```

## 安全管理

### 访问控制

```bash
# 防火墙配置
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # 直接API访问
ufw enable

# SSL证书更新
certbot renew --nginx
```

### 安全监控

```bash
# 检查异常登录
grep "Failed password" /var/log/auth.log

# 检查异常访问
grep "404\|403\|500" /var/log/nginx/access.log

# 检查系统完整性
aide --check
```

### 定期安全任务

- [ ] 更新系统补丁
- [ ] 更新SSL证书
- [ ] 检查访问日志
- [ ] 更新密码策略
- [ ] 审计用户权限
- [ ] 备份安全配置

## 运维自动化

### 监控脚本

```bash
#!/bin/bash
# monitor.sh

# 检查服务状态
check_service() {
    if systemctl is-active --quiet $1; then
        echo "✅ $1 is running"
    else
        echo "❌ $1 is not running"
        systemctl restart $1
    fi
}

check_service stock-analysis
check_service postgresql
check_service redis
check_service nginx

# 检查磁盘空间
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️ Disk usage is ${DISK_USAGE}%"
    # 清理日志
    find /var/log -name "*.log" -mtime +7 -delete
fi

# 检查内存使用
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ $MEMORY_USAGE -gt 80 ]; then
    echo "⚠️ Memory usage is ${MEMORY_USAGE}%"
fi
```

### 部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

echo "开始部署..."

# 拉取最新代码
git pull origin main

# 构建Docker镜像
docker build -t stock-analysis:latest .

# 更新服务
docker-compose down
docker-compose up -d

# 等待服务启动
sleep 30

# 健康检查
if curl -f http://localhost:8000/health; then
    echo "✅ 部署成功"
else
    echo "❌ 部署失败，回滚..."
    docker-compose down
    docker-compose up -d
    exit 1
fi
```

## 联系信息

- **运维团队**: ops@stockanalysis.com
- **紧急联系**: +86-xxx-xxxx-xxxx
- **技术支持**: support@stockanalysis.com
- **监控告警**: alerts@stockanalysis.com