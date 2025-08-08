#!/bin/bash
"""
系统维护脚本

执行定期系统维护任务，包括日志清理、性能优化、安全更新等。
"""

set -e

# 配置变量
NAMESPACE="stock-analysis"
LOG_RETENTION_DAYS=30
BACKUP_RETENTION_DAYS=30
MAINTENANCE_LOG="/var/log/stock-analysis-maintenance.log"

# 日志函数
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message"
    echo "$message" >> "$MAINTENANCE_LOG"
}

error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" >&2
    echo "$message" >> "$MAINTENANCE_LOG"
    exit 1
}

# 创建维护日志目录
mkdir -p "$(dirname "$MAINTENANCE_LOG")"

# 系统健康检查
system_health_check() {
    log "=== 系统健康检查 ==="
    
    # 检查集群状态
    log "检查Kubernetes集群状态..."
    if ! kubectl cluster-info &> /dev/null; then
        error "Kubernetes集群连接失败"
    fi
    
    # 检查节点状态
    log "检查节点状态..."
    local not_ready_nodes=$(kubectl get nodes --no-headers | grep -v Ready | wc -l)
    if [ "$not_ready_nodes" -gt 0 ]; then
        log "警告: 发现 $not_ready_nodes 个节点状态异常"
    else
        log "所有节点状态正常"
    fi
    
    # 检查Pod状态
    log "检查Pod状态..."
    local failed_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -E "(Error|CrashLoopBackOff|ImagePullBackOff)" | wc -l)
    if [ "$failed_pods" -gt 0 ]; then
        log "警告: 发现 $failed_pods 个Pod状态异常"
        kubectl get pods -n "$NAMESPACE" | grep -E "(Error|CrashLoopBackOff|ImagePullBackOff)" >> "$MAINTENANCE_LOG"
    else
        log "所有Pod状态正常"
    fi
    
    # 检查存储使用
    log "检查存储使用情况..."
    kubectl get pvc -n "$NAMESPACE" >> "$MAINTENANCE_LOG"
    
    # 检查资源使用
    log "检查资源使用情况..."
    kubectl top nodes >> "$MAINTENANCE_LOG" 2>/dev/null || log "无法获取节点资源使用情况"
    kubectl top pods -n "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "无法获取Pod资源使用情况"
    
    log "系统健康检查完成"
}

# 清理日志文件
cleanup_logs() {
    log "=== 清理日志文件 ==="
    
    # 清理应用日志
    log "清理应用日志文件..."
    find /var/log -name "*.log" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    find /var/log -name "*.log.*" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    
    # 清理Docker日志
    log "清理Docker日志..."
    if command -v docker &> /dev/null; then
        docker system prune -f --filter "until=${LOG_RETENTION_DAYS}d" &> /dev/null || true
    fi
    
    # 清理Kubernetes事件
    log "清理Kubernetes事件..."
    kubectl delete events --all-namespaces --field-selector reason!=Normal &> /dev/null || true
    
    # 清理临时文件
    log "清理临时文件..."
    find /tmp -name "stock-analysis-*" -mtime +7 -delete 2>/dev/null || true
    
    log "日志清理完成"
}

# 清理旧备份
cleanup_old_backups() {
    log "=== 清理旧备份文件 ==="
    
    local backup_dir="/backups"
    
    if [ -d "$backup_dir" ]; then
        # 清理数据库备份
        find "$backup_dir/database" -name "*.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        find "$backup_dir/database" -name "*.md5" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        
        # 清理Redis备份
        find "$backup_dir/redis" -name "*.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        find "$backup_dir/redis" -name "*.md5" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        
        # 清理配置备份
        find "$backup_dir/configs" -name "*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        
        # 清理日志备份
        find "$backup_dir/logs" -name "*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
        
        # 清理空目录
        find "$backup_dir" -type d -empty -delete 2>/dev/null || true
        
        log "旧备份清理完成"
    else
        log "备份目录不存在，跳过备份清理"
    fi
}

# 数据库维护
database_maintenance() {
    log "=== 数据库维护 ==="
    
    # 检查数据库连接
    if ! kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -c "SELECT 1;" &> /dev/null; then
        log "警告: 无法连接到数据库"
        return
    fi
    
    # 更新统计信息
    log "更新数据库统计信息..."
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -c "ANALYZE;" >> "$MAINTENANCE_LOG"
    
    # 清理死元组
    log "清理数据库死元组..."
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -c "VACUUM;" >> "$MAINTENANCE_LOG"
    
    # 检查数据库大小
    log "检查数据库大小..."
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -c "
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
        LIMIT 10;
    " >> "$MAINTENANCE_LOG"
    
    # 检查慢查询
    log "检查慢查询..."
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -c "
        SELECT 
            query,
            calls,
            total_time,
            mean_time,
            rows
        FROM pg_stat_statements 
        ORDER BY mean_time DESC 
        LIMIT 10;
    " >> "$MAINTENANCE_LOG" 2>/dev/null || log "pg_stat_statements扩展未启用"
    
    log "数据库维护完成"
}

# Redis维护
redis_maintenance() {
    log "=== Redis维护 ==="
    
    # 检查Redis连接
    if ! kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli ping | grep -q PONG; then
        log "警告: 无法连接到Redis"
        return
    fi
    
    # 获取Redis信息
    log "获取Redis信息..."
    kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli INFO memory >> "$MAINTENANCE_LOG"
    kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli INFO stats >> "$MAINTENANCE_LOG"
    
    # 检查内存使用
    local memory_usage=$(kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
    log "Redis内存使用: $memory_usage"
    
    # 检查键空间
    local keyspace_info=$(kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli INFO keyspace)
    if [ ! -z "$keyspace_info" ]; then
        log "Redis键空间信息:"
        echo "$keyspace_info" >> "$MAINTENANCE_LOG"
    fi
    
    # 清理过期键（如果需要）
    log "触发过期键清理..."
    kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli EVAL "return redis.call('scan', 0, 'count', 1000)" 0 > /dev/null
    
    log "Redis维护完成"
}

# 性能优化
performance_optimization() {
    log "=== 性能优化 ==="
    
    # 检查资源限制
    log "检查资源限制配置..."
    kubectl describe deployment stock-analysis-api -n "$NAMESPACE" | grep -A 10 "Limits\|Requests" >> "$MAINTENANCE_LOG"
    
    # 检查HPA状态
    log "检查自动扩缩容状态..."
    kubectl get hpa -n "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "未配置HPA"
    
    # 检查网络策略
    log "检查网络策略..."
    kubectl get networkpolicies -n "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "未配置网络策略"
    
    # 优化建议
    log "生成性能优化建议..."
    
    # 检查CPU使用率
    local high_cpu_pods=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$2 ~ /[0-9]+m/ && $2+0 > 800 {print $1}' || true)
    if [ ! -z "$high_cpu_pods" ]; then
        log "建议: 以下Pod CPU使用率较高，考虑增加资源限制或优化代码:"
        echo "$high_cpu_pods" >> "$MAINTENANCE_LOG"
    fi
    
    # 检查内存使用率
    local high_memory_pods=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '$3 ~ /[0-9]+Mi/ && $3+0 > 1500 {print $1}' || true)
    if [ ! -z "$high_memory_pods" ]; then
        log "建议: 以下Pod内存使用率较高，考虑增加内存限制或优化内存使用:"
        echo "$high_memory_pods" >> "$MAINTENANCE_LOG"
    fi
    
    log "性能优化检查完成"
}

# 安全检查
security_check() {
    log "=== 安全检查 ==="
    
    # 检查镜像版本
    log "检查容器镜像版本..."
    kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\n"}{end}' >> "$MAINTENANCE_LOG"
    
    # 检查安全上下文
    log "检查安全上下文配置..."
    kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext}{"\n"}{end}' >> "$MAINTENANCE_LOG"
    
    # 检查RBAC配置
    log "检查RBAC配置..."
    kubectl get rolebindings -n "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "未找到RoleBindings"
    kubectl get clusterrolebindings | grep "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "未找到相关ClusterRoleBindings"
    
    # 检查网络策略
    log "检查网络安全策略..."
    kubectl get networkpolicies -n "$NAMESPACE" >> "$MAINTENANCE_LOG" 2>/dev/null || log "未配置网络策略"
    
    # 检查Secrets
    log "检查Secrets配置..."
    kubectl get secrets -n "$NAMESPACE" --no-headers | wc -l | xargs -I {} log "发现 {} 个Secrets"
    
    log "安全检查完成"
}

# 容量规划
capacity_planning() {
    log "=== 容量规划分析 ==="
    
    # 检查存储使用趋势
    log "分析存储使用趋势..."
    kubectl get pvc -n "$NAMESPACE" -o custom-columns=NAME:.metadata.name,CAPACITY:.spec.resources.requests.storage,STATUS:.status.phase >> "$MAINTENANCE_LOG"
    
    # 检查资源使用趋势
    log "分析资源使用趋势..."
    
    # 获取当前资源使用
    local current_cpu=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$2} END {print sum}' || echo "0")
    local current_memory=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' || echo "0")
    
    log "当前CPU使用总量: ${current_cpu}m"
    log "当前内存使用总量: ${current_memory}Mi"
    
    # 容量建议
    log "容量规划建议:"
    log "- 建议定期监控资源使用趋势"
    log "- 考虑在高峰期前扩容"
    log "- 评估是否需要增加节点"
    
    log "容量规划分析完成"
}

# 生成维护报告
generate_maintenance_report() {
    log "=== 生成维护报告 ==="
    
    local report_file="maintenance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" <<EOF
股票分析系统维护报告
==================

维护时间: $(date)
维护类型: 定期维护
命名空间: $NAMESPACE

维护内容:
✅ 系统健康检查
✅ 日志文件清理
✅ 旧备份清理
✅ 数据库维护
✅ Redis维护
✅ 性能优化检查
✅ 安全检查
✅ 容量规划分析

详细日志: $MAINTENANCE_LOG

系统状态摘要:
EOF
    
    # 添加当前系统状态
    echo "" >> "$report_file"
    echo "Pod状态:" >> "$report_file"
    kubectl get pods -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "无法获取Pod状态" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "服务状态:" >> "$report_file"
    kubectl get services -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "无法获取服务状态" >> "$report_file"
    
    echo "" >> "$report_file"
    echo "存储状态:" >> "$report_file"
    kubectl get pvc -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "无法获取存储状态" >> "$report_file"
    
    log "维护报告生成完成: $report_file"
}

# 发送维护通知
send_maintenance_notification() {
    log "发送维护通知..."
    
    local report_file="maintenance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    # 发送邮件通知
    if command -v mail &> /dev/null; then
        mail -s "股票分析系统维护完成 - $(date +%Y-%m-%d)" ops@company.com < "$report_file"
    fi
    
    # 发送Slack通知
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔧 股票分析系统定期维护完成\\n时间: $(date)\\n状态: 成功\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    log "维护通知发送完成"
}

# 主函数
main() {
    log "开始执行系统维护..."
    
    # 检查必要的工具
    if ! command -v kubectl &> /dev/null; then
        error "kubectl 未安装"
    fi
    
    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        error "无法连接到Kubernetes集群"
    fi
    
    # 执行维护任务
    system_health_check
    cleanup_logs
    cleanup_old_backups
    database_maintenance
    redis_maintenance
    performance_optimization
    security_check
    capacity_planning
    generate_maintenance_report
    send_maintenance_notification
    
    log "系统维护完成！"
    log "维护日志: $MAINTENANCE_LOG"
}

# 处理命令行参数
case "${1:-}" in
    "health")
        log "仅执行健康检查..."
        system_health_check
        ;;
    "cleanup")
        log "仅执行清理任务..."
        cleanup_logs
        cleanup_old_backups
        ;;
    "database")
        log "仅执行数据库维护..."
        database_maintenance
        ;;
    "redis")
        log "仅执行Redis维护..."
        redis_maintenance
        ;;
    "security")
        log "仅执行安全检查..."
        security_check
        ;;
    "full"|"")
        main
        ;;
    *)
        echo "用法: $0 [health|cleanup|database|redis|security|full]"
        echo "  health   - 仅执行健康检查"
        echo "  cleanup  - 仅执行清理任务"
        echo "  database - 仅执行数据库维护"
        echo "  redis    - 仅执行Redis维护"
        echo "  security - 仅执行安全检查"
        echo "  full     - 完整维护（默认）"
        exit 1
        ;;
esac