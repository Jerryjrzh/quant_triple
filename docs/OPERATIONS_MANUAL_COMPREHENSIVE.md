# 股票分析系统运维手册

## 目录

1. [系统概述](#系统概述)
2. [部署架构](#部署架构)
3. [日常运维操作](#日常运维操作)
4. [监控和告警](#监控和告警)
5. [备份和恢复](#备份和恢复)
6. [故障排除](#故障排除)
7. [系统维护](#系统维护)
8. [升级流程](#升级流程)
9. [应急响应](#应急响应)
10. [性能优化](#性能优化)

## 系统概述

### 系统架构

股票分析系统采用微服务架构，主要包括以下组件：

- **API服务**: FastAPI应用，提供REST API接口
- **前端服务**: React应用，提供用户界面
- **数据处理服务**: Celery工作节点，处理后台任务
- **数据库**: PostgreSQL，存储业务数据
- **缓存**: Redis，提供缓存服务
- **消息队列**: Redis，作为Celery的消息代理
- **反向代理**: Nginx，负载均衡和SSL终止

### 技术栈

- **后端**: Python 3.12, FastAPI, SQLAlchemy, Celery
- **前端**: React, TypeScript, Ant Design
- **数据库**: PostgreSQL 15
- **缓存**: Redis 7
- **容器化**: Docker, Docker Compose
- **编排**: Kubernetes
- **监控**: Prometheus, Grafana, ELK Stack
- **CI/CD**: GitHub Actions

## 部署架构

### 生产环境架构

```
Internet
    |
[Load Balancer]
    |
[Nginx Ingress]
    |
+-- [Frontend Pods] (3 replicas)
    |
+-- [API Pods] (5 replicas)
    |
+-- [Celery Pods] (3 replicas)
    |
+-- [PostgreSQL] (Master-Slave)
    |
+-- [Redis Cluster] (3 nodes)
```

### 资源配置

#### API服务
- CPU: 1000m (1 core)
- Memory: 2Gi
- 副本数: 5
- 自动扩缩容: 3-10 副本

#### 前端服务
- CPU: 100m
- Memory: 256Mi
- 副本数: 3

#### Celery服务
- CPU: 500m
- Memory: 1Gi
- 副本数: 3

#### 数据库
- CPU: 2000m (2 cores)
- Memory: 4Gi
- 存储: 100Gi SSD

#### Redis
- CPU: 500m
- Memory: 1Gi
- 存储: 10Gi

## 日常运维操作

### 1. 系统状态检查

#### 检查所有Pod状态
```bash
kubectl get pods -n stock-analysis
```

#### 检查服务状态
```bash
kubectl get services -n stock-analysis
```

#### 检查Ingress状态
```bash
kubectl get ingress -n stock-analysis
```

#### 检查节点资源使用
```bash
kubectl top nodes
kubectl top pods -n stock-analysis
```

### 2. 日志查看

#### 查看API服务日志
```bash
kubectl logs -f deployment/stock-analysis-api -n stock-analysis
```

#### 查看Celery服务日志
```bash
kubectl logs -f deployment/stock-analysis-celery -n stock-analysis
```

#### 查看数据库日志
```bash
kubectl logs -f statefulset/postgresql -n stock-analysis
```

### 3. 数据库操作

#### 连接数据库
```bash
kubectl exec -it postgresql-0 -n stock-analysis -- psql -U postgres -d stock_analysis
```

#### 查看数据库连接数
```sql
SELECT count(*) FROM pg_stat_activity;
```

#### 查看慢查询
```sql
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

### 4. 缓存操作

#### 连接Redis
```bash
kubectl exec -it redis-0 -n stock-analysis -- redis-cli
```

#### 查看Redis信息
```bash
INFO memory
INFO stats
```

#### 清理缓存
```bash
FLUSHDB
```

## 监控和告警

### 监控指标

#### 系统级指标
- CPU使用率
- 内存使用率
- 磁盘使用率
- 网络I/O
- 磁盘I/O

#### 应用级指标
- API响应时间
- API错误率
- 请求QPS
- 数据库连接数
- 缓存命中率
- Celery任务队列长度

#### 业务级指标
- 用户活跃数
- 数据更新频率
- 分析任务成功率
- 数据质量分数

### Prometheus配置

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

  - job_name: 'stock-analysis-api'
    static_configs:
      - targets: ['stock-analysis-api:8000']
    metrics_path: '/metrics'

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
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for the last 5 minutes"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: DatabaseConnectionHigh
        expr: pg_stat_activity_count > 80
        for: 2m
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

      - alert: CeleryQueueHigh
        expr: celery_queue_length > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Celery queue length high"
          description: "Celery queue has {{ $value }} pending tasks"
```

### Grafana仪表板

#### 系统概览仪表板
- 系统健康状态
- 关键性能指标
- 错误率趋势
- 用户活跃度

#### API性能仪表板
- 请求量和响应时间
- 错误率分布
- 端点性能排行
- 状态码分布

#### 基础设施仪表板
- 节点资源使用
- Pod状态和重启次数
- 网络和存储I/O
- 集群容量规划

## 备份和恢复

### 数据库备份

#### 自动备份脚本
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="stock_analysis_backup_${DATE}.sql"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 执行备份
kubectl exec postgresql-0 -n stock-analysis -- pg_dump -U postgres stock_analysis > $BACKUP_DIR/$BACKUP_FILE

# 压缩备份文件
gzip $BACKUP_DIR/$BACKUP_FILE

# 删除7天前的备份
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "Database backup completed: $BACKUP_FILE.gz"
```

#### 数据库恢复
```bash
#!/bin/bash
# restore_database.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# 解压备份文件
gunzip $BACKUP_FILE

# 停止应用服务
kubectl scale deployment stock-analysis-api --replicas=0 -n stock-analysis
kubectl scale deployment stock-analysis-celery --replicas=0 -n stock-analysis

# 恢复数据库
kubectl exec -i postgresql-0 -n stock-analysis -- psql -U postgres -d stock_analysis < ${BACKUP_FILE%.gz}

# 重启应用服务
kubectl scale deployment stock-analysis-api --replicas=5 -n stock-analysis
kubectl scale deployment stock-analysis-celery --replicas=3 -n stock-analysis

echo "Database restore completed"
```

### Redis备份

#### Redis备份脚本
```bash
#!/bin/bash
# backup_redis.sh

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# 触发Redis保存
kubectl exec redis-0 -n stock-analysis -- redis-cli BGSAVE

# 等待保存完成
sleep 10

# 复制RDB文件
kubectl cp stock-analysis/redis-0:/data/dump.rdb $BACKUP_DIR/redis_backup_${DATE}.rdb

echo "Redis backup completed: redis_backup_${DATE}.rdb"
```

### 配置文件备份

#### 备份Kubernetes配置
```bash
#!/bin/bash
# backup_k8s_configs.sh

BACKUP_DIR="/backups/k8s"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/$DATE

# 备份所有配置
kubectl get all -n stock-analysis -o yaml > $BACKUP_DIR/$DATE/all-resources.yaml
kubectl get configmaps -n stock-analysis -o yaml > $BACKUP_DIR/$DATE/configmaps.yaml
kubectl get secrets -n stock-analysis -o yaml > $BACKUP_DIR/$DATE/secrets.yaml
kubectl get pvc -n stock-analysis -o yaml > $BACKUP_DIR/$DATE/pvc.yaml

# 压缩备份
tar -czf $BACKUP_DIR/k8s_backup_${DATE}.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

echo "Kubernetes configs backup completed: k8s_backup_${DATE}.tar.gz"
```

## 故障排除

### 常见问题和解决方案

#### 1. API服务无响应

**症状**: API请求超时或返回502错误

**排查步骤**:
```bash
# 检查Pod状态
kubectl get pods -n stock-analysis -l app=stock-analysis-api

# 查看Pod日志
kubectl logs -f deployment/stock-analysis-api -n stock-analysis

# 检查资源使用
kubectl top pods -n stock-analysis -l app=stock-analysis-api

# 检查服务端点
kubectl get endpoints -n stock-analysis
```

**可能原因和解决方案**:
- **内存不足**: 增加内存限制或优化代码
- **CPU限制**: 增加CPU限制或优化算法
- **数据库连接池耗尽**: 检查数据库连接配置
- **依赖服务不可用**: 检查Redis和PostgreSQL状态

#### 2. 数据库连接问题

**症状**: 应用无法连接数据库

**排查步骤**:
```bash
# 检查PostgreSQL Pod状态
kubectl get pods -n stock-analysis -l app=postgresql

# 查看数据库日志
kubectl logs -f statefulset/postgresql -n stock-analysis

# 测试数据库连接
kubectl exec -it postgresql-0 -n stock-analysis -- psql -U postgres -c "SELECT 1"
```

**解决方案**:
- 检查数据库密码配置
- 验证网络策略
- 检查存储卷状态
- 重启数据库Pod（谨慎操作）

#### 3. Redis缓存问题

**症状**: 缓存命中率低或Redis连接失败

**排查步骤**:
```bash
# 检查Redis状态
kubectl get pods -n stock-analysis -l app=redis

# 查看Redis日志
kubectl logs -f statefulset/redis -n stock-analysis

# 检查Redis内存使用
kubectl exec redis-0 -n stock-analysis -- redis-cli INFO memory
```

**解决方案**:
- 增加Redis内存限制
- 优化缓存策略
- 检查网络连接
- 清理过期键

#### 4. Celery任务堆积

**症状**: 后台任务处理缓慢，队列长度持续增长

**排查步骤**:
```bash
# 检查Celery Worker状态
kubectl get pods -n stock-analysis -l app=stock-analysis-celery

# 查看Celery日志
kubectl logs -f deployment/stock-analysis-celery -n stock-analysis

# 检查队列长度
kubectl exec redis-0 -n stock-analysis -- redis-cli LLEN celery
```

**解决方案**:
- 增加Celery Worker数量
- 优化任务处理逻辑
- 检查任务失败原因
- 清理失败任务

## 系统维护

### 定期维护任务

#### 每日维护
- [ ] 检查系统健康状态
- [ ] 查看监控告警
- [ ] 检查日志错误
- [ ] 验证备份完成
- [ ] 检查磁盘空间

#### 每周维护
- [ ] 分析性能趋势
- [ ] 检查资源使用情况
- [ ] 清理旧日志文件
- [ ] 更新安全补丁
- [ ] 验证备份恢复

#### 每月维护
- [ ] 容量规划评估
- [ ] 性能优化分析
- [ ] 安全审计
- [ ] 文档更新
- [ ] 灾难恢复演练

### 维护脚本

#### 日志清理脚本
```bash
#!/bin/bash
# cleanup_logs.sh

# 清理应用日志（保留30天）
find /var/log/stock-analysis -name "*.log" -mtime +30 -delete

# 清理Docker日志
docker system prune -f --filter "until=720h"

# 清理Kubernetes事件日志
kubectl delete events --all-namespaces --field-selector reason!=Normal

echo "Log cleanup completed"
```

#### 系统健康检查脚本
```bash
#!/bin/bash
# health_check.sh

echo "=== System Health Check ==="
echo "Date: $(date)"
echo

# 检查Pod状态
echo "Pod Status:"
kubectl get pods -n stock-analysis --no-headers | awk '{print $1, $3}' | grep -v Running && echo "Some pods are not running!" || echo "All pods are running"
echo

# 检查服务状态
echo "Service Status:"
kubectl get services -n stock-analysis
echo

# 检查资源使用
echo "Resource Usage:"
kubectl top nodes
echo

# 检查存储使用
echo "Storage Usage:"
kubectl get pvc -n stock-analysis
echo

# 检查最近的告警
echo "Recent Alerts:"
# 这里可以集成Alertmanager API查询
echo "Check Grafana dashboard for detailed alerts"
echo

echo "=== Health Check Completed ==="
```

## 升级流程

### 应用升级

#### 1. 准备阶段
```bash
# 创建备份
./backup_database.sh
./backup_k8s_configs.sh

# 检查当前版本
kubectl get deployment stock-analysis-api -n stock-analysis -o jsonpath='{.spec.template.spec.containers[0].image}'
```

#### 2. 滚动升级
```bash
# 更新API服务
kubectl set image deployment/stock-analysis-api stock-analysis-api=stock-analysis:v2.0.0 -n stock-analysis

# 监控升级进度
kubectl rollout status deployment/stock-analysis-api -n stock-analysis

# 更新Celery服务
kubectl set image deployment/stock-analysis-celery stock-analysis-celery=stock-analysis:v2.0.0 -n stock-analysis
kubectl rollout status deployment/stock-analysis-celery -n stock-analysis
```

#### 3. 验证升级
```bash
# 检查Pod状态
kubectl get pods -n stock-analysis

# 验证API功能
curl -f http://api.stockanalysis.com/health

# 检查日志
kubectl logs -f deployment/stock-analysis-api -n stock-analysis --tail=100
```

#### 4. 回滚（如需要）
```bash
# 回滚到上一版本
kubectl rollout undo deployment/stock-analysis-api -n stock-analysis
kubectl rollout undo deployment/stock-analysis-celery -n stock-analysis

# 验证回滚
kubectl rollout status deployment/stock-analysis-api -n stock-analysis
```

### 数据库升级

#### 1. 准备阶段
```bash
# 完整备份
./backup_database.sh

# 停止写入服务
kubectl scale deployment stock-analysis-api --replicas=0 -n stock-analysis
kubectl scale deployment stock-analysis-celery --replicas=0 -n stock-analysis
```

#### 2. 执行升级
```bash
# 运行数据库迁移
kubectl exec -it postgresql-0 -n stock-analysis -- psql -U postgres -d stock_analysis -f /migrations/upgrade.sql
```

#### 3. 验证和恢复
```bash
# 验证数据完整性
kubectl exec -it postgresql-0 -n stock-analysis -- psql -U postgres -d stock_analysis -c "SELECT COUNT(*) FROM stocks;"

# 恢复服务
kubectl scale deployment stock-analysis-api --replicas=5 -n stock-analysis
kubectl scale deployment stock-analysis-celery --replicas=3 -n stock-analysis
```

## 应急响应

### 应急响应流程

#### 1. 事件分级

**P0 - 紧急**
- 系统完全不可用
- 数据丢失或损坏
- 安全漏洞被利用

**P1 - 高优先级**
- 核心功能不可用
- 性能严重下降
- 部分用户无法访问

**P2 - 中优先级**
- 非核心功能异常
- 性能轻微下降
- 监控告警

**P3 - 低优先级**
- 界面问题
- 文档错误
- 优化建议

#### 2. 应急联系人

```
角色                电话            邮箱                    备注
系统管理员          138-xxxx-xxxx   admin@company.com       24小时待命
开发负责人          139-xxxx-xxxx   dev@company.com         工作时间
数据库管理员        137-xxxx-xxxx   dba@company.com         24小时待命
网络管理员          136-xxxx-xxxx   network@company.com     工作时间
```

#### 3. 应急处理步骤

**步骤1: 事件确认**
- 确认事件影响范围
- 评估事件严重程度
- 通知相关人员

**步骤2: 快速响应**
- 执行临时修复措施
- 启用备用系统（如有）
- 通知用户（如需要）

**步骤3: 根本原因分析**
- 收集日志和监控数据
- 分析问题根本原因
- 制定永久解决方案

**步骤4: 恢复和验证**
- 实施永久修复
- 验证系统功能
- 更新文档和流程

### 常见应急场景

#### 场景1: 数据库主节点故障

**应急步骤**:
```bash
# 1. 确认主节点状态
kubectl get pods -n stock-analysis -l app=postgresql

# 2. 切换到从节点
kubectl patch service postgresql -n stock-analysis -p '{"spec":{"selector":{"role":"slave"}}}'

# 3. 更新应用配置
kubectl set env deployment/stock-analysis-api DATABASE_URL=postgresql://user:pass@postgresql-slave:5432/db -n stock-analysis

# 4. 验证服务恢复
curl -f http://api.stockanalysis.com/health
```

#### 场景2: Redis集群故障

**应急步骤**:
```bash
# 1. 检查Redis集群状态
kubectl exec redis-0 -n stock-analysis -- redis-cli cluster nodes

# 2. 重启故障节点
kubectl delete pod redis-1 -n stock-analysis

# 3. 等待集群自愈
kubectl exec redis-0 -n stock-analysis -- redis-cli cluster info

# 4. 如需要，清空缓存重新构建
kubectl exec redis-0 -n stock-analysis -- redis-cli FLUSHALL
```

#### 场景3: API服务大量错误

**应急步骤**:
```bash
# 1. 快速扩容
kubectl scale deployment stock-analysis-api --replicas=10 -n stock-analysis

# 2. 检查资源限制
kubectl describe deployment stock-analysis-api -n stock-analysis

# 3. 如需要，重启服务
kubectl rollout restart deployment/stock-analysis-api -n stock-analysis

# 4. 监控恢复情况
kubectl get pods -n stock-analysis -w
```

## 性能优化

### 数据库优化

#### 查询优化
```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_stocks_symbol ON stocks(symbol);
CREATE INDEX CONCURRENTLY idx_daily_data_date ON daily_data(date);

-- 分析查询计划
EXPLAIN ANALYZE SELECT * FROM stocks WHERE symbol = '000001.SZ';

-- 更新统计信息
ANALYZE;
```

#### 连接池优化
```python
# 数据库连接池配置
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

### 缓存优化

#### Redis配置优化
```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### 应用缓存策略
```python
# 缓存配置
CACHE_CONFIG = {
    'realtime_data': {'ttl': 60, 'preload': True},
    'daily_data': {'ttl': 3600, 'compress': True},
    'analysis_results': {'ttl': 1800, 'preload': False}
}
```

### 应用优化

#### 异步处理
```python
# 使用异步处理提高并发
@app.get("/api/v1/analysis/{symbol}")
async def get_analysis(symbol: str):
    # 异步获取数据
    data = await get_stock_data(symbol)
    # 异步分析
    result = await analyze_data(data)
    return result
```

#### 批量处理
```python
# 批量处理减少数据库访问
async def batch_update_stocks(symbols: List[str]):
    # 批量获取数据
    data_list = await get_multiple_stock_data(symbols)
    # 批量插入数据库
    await bulk_insert_data(data_list)
```

---

## 附录

### A. 监控指标参考

#### 系统指标
- CPU使用率: < 80%
- 内存使用率: < 85%
- 磁盘使用率: < 90%
- 网络延迟: < 100ms

#### 应用指标
- API响应时间: P95 < 2s
- API错误率: < 1%
- 数据库连接数: < 80
- 缓存命中率: > 90%

### B. 常用命令速查

```bash
# Kubernetes
kubectl get pods -n stock-analysis
kubectl logs -f deployment/stock-analysis-api -n stock-analysis
kubectl exec -it pod-name -n stock-analysis -- bash
kubectl scale deployment stock-analysis-api --replicas=5 -n stock-analysis

# Docker
docker ps
docker logs container-name
docker exec -it container-name bash
docker system prune -f

# 数据库
psql -U postgres -d stock_analysis
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

# Redis
redis-cli
INFO memory
MONITOR
FLUSHDB
```

### C. 联系信息

如有问题或建议，请联系：

- 技术支持: support@company.com
- 运维团队: ops@company.com
- 开发团队: dev@company.com

---

*本文档最后更新时间: 2025-08-08*
*版本: v1.0*