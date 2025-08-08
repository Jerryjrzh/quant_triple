#!/bin/bash
"""
系统备份脚本

执行完整的系统备份，包括数据库、配置文件、和应用数据。
支持增量备份和完整备份。
"""

set -e

# 配置变量
BACKUP_BASE_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACE="stock-analysis"
RETENTION_DAYS=30

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

# 创建备份目录
create_backup_dirs() {
    log "创建备份目录..."
    
    mkdir -p "$BACKUP_BASE_DIR/database/$DATE"
    mkdir -p "$BACKUP_BASE_DIR/redis/$DATE"
    mkdir -p "$BACKUP_BASE_DIR/configs/$DATE"
    mkdir -p "$BACKUP_BASE_DIR/logs/$DATE"
    
    log "备份目录创建完成"
}

# 备份PostgreSQL数据库
backup_database() {
    log "开始备份PostgreSQL数据库..."
    
    local db_backup_file="$BACKUP_BASE_DIR/database/$DATE/stock_analysis_db_$DATE.sql"
    
    # 执行数据库备份
    kubectl exec -n $NAMESPACE postgresql-0 -- pg_dump -U postgres stock_analysis > "$db_backup_file"
    
    if [ $? -eq 0 ]; then
        # 压缩备份文件
        gzip "$db_backup_file"
        log "数据库备份完成: ${db_backup_file}.gz"
        
        # 生成校验和
        md5sum "${db_backup_file}.gz" > "${db_backup_file}.gz.md5"
        
        # 备份数据库配置
        kubectl get configmap postgresql-config -n $NAMESPACE -o yaml > "$BACKUP_BASE_DIR/database/$DATE/postgresql-config.yaml"
        
    else
        error "数据库备份失败"
    fi
}

# 备份Redis数据
backup_redis() {
    log "开始备份Redis数据..."
    
    local redis_backup_file="$BACKUP_BASE_DIR/redis/$DATE/redis_dump_$DATE.rdb"
    
    # 触发Redis保存
    kubectl exec -n $NAMESPACE redis-0 -- redis-cli BGSAVE
    
    # 等待保存完成
    sleep 10
    
    # 检查保存状态
    local save_status=$(kubectl exec -n $NAMESPACE redis-0 -- redis-cli LASTSAVE)
    log "Redis LASTSAVE: $save_status"
    
    # 复制RDB文件
    kubectl cp $NAMESPACE/redis-0:/data/dump.rdb "$redis_backup_file"
    
    if [ $? -eq 0 ]; then
        # 压缩备份文件
        gzip "$redis_backup_file"
        log "Redis备份完成: ${redis_backup_file}.gz"
        
        # 生成校验和
        md5sum "${redis_backup_file}.gz" > "${redis_backup_file}.gz.md5"
        
        # 备份Redis配置
        kubectl get configmap redis-config -n $NAMESPACE -o yaml > "$BACKUP_BASE_DIR/redis/$DATE/redis-config.yaml"
        
    else
        error "Redis备份失败"
    fi
}

# 备份Kubernetes配置
backup_k8s_configs() {
    log "开始备份Kubernetes配置..."
    
    local config_dir="$BACKUP_BASE_DIR/configs/$DATE"
    
    # 备份所有资源
    kubectl get all -n $NAMESPACE -o yaml > "$config_dir/all-resources.yaml"
    kubectl get configmaps -n $NAMESPACE -o yaml > "$config_dir/configmaps.yaml"
    kubectl get secrets -n $NAMESPACE -o yaml > "$config_dir/secrets.yaml"
    kubectl get pvc -n $NAMESPACE -o yaml > "$config_dir/pvc.yaml"
    kubectl get ingress -n $NAMESPACE -o yaml > "$config_dir/ingress.yaml"
    kubectl get networkpolicies -n $NAMESPACE -o yaml > "$config_dir/networkpolicies.yaml"
    
    # 备份Helm charts（如果使用）
    if command -v helm &> /dev/null; then
        helm list -n $NAMESPACE -o yaml > "$config_dir/helm-releases.yaml"
    fi
    
    # 压缩配置文件
    tar -czf "$config_dir.tar.gz" -C "$BACKUP_BASE_DIR/configs" "$DATE"
    rm -rf "$config_dir"
    
    log "Kubernetes配置备份完成: $config_dir.tar.gz"
}

# 备份应用日志
backup_logs() {
    log "开始备份应用日志..."
    
    local log_dir="$BACKUP_BASE_DIR/logs/$DATE"
    
    # 获取所有Pod的日志
    for pod in $(kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
        log "备份Pod日志: $pod"
        kubectl logs -n $NAMESPACE "$pod" --all-containers=true > "$log_dir/${pod}.log" 2>/dev/null || true
    done
    
    # 压缩日志文件
    tar -czf "$log_dir.tar.gz" -C "$BACKUP_BASE_DIR/logs" "$DATE"
    rm -rf "$log_dir"
    
    log "应用日志备份完成: $log_dir.tar.gz"
}

# 备份持久化卷数据
backup_persistent_volumes() {
    log "开始备份持久化卷数据..."
    
    local pv_backup_dir="$BACKUP_BASE_DIR/volumes/$DATE"
    mkdir -p "$pv_backup_dir"
    
    # 获取所有PVC
    for pvc in $(kubectl get pvc -n $NAMESPACE -o jsonpath='{.items[*].metadata.name}'); do
        log "备份PVC: $pvc"
        
        # 创建临时Pod来访问PVC
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: backup-$pvc
  namespace: $NAMESPACE
spec:
  containers:
  - name: backup
    image: busybox
    command: ['sleep', '3600']
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: $pvc
  restartPolicy: Never
EOF
        
        # 等待Pod启动
        kubectl wait --for=condition=Ready pod/backup-$pvc -n $NAMESPACE --timeout=300s
        
        # 备份数据
        kubectl exec -n $NAMESPACE backup-$pvc -- tar -czf - /data > "$pv_backup_dir/${pvc}.tar.gz"
        
        # 清理临时Pod
        kubectl delete pod backup-$pvc -n $NAMESPACE
        
        log "PVC备份完成: $pvc"
    done
    
    log "持久化卷备份完成"
}

# 清理旧备份
cleanup_old_backups() {
    log "清理旧备份文件..."
    
    # 清理数据库备份
    find "$BACKUP_BASE_DIR/database" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_BASE_DIR/database" -name "*.md5" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_BASE_DIR/database" -type d -empty -delete
    
    # 清理Redis备份
    find "$BACKUP_BASE_DIR/redis" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_BASE_DIR/redis" -name "*.md5" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_BASE_DIR/redis" -type d -empty -delete
    
    # 清理配置备份
    find "$BACKUP_BASE_DIR/configs" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    
    # 清理日志备份
    find "$BACKUP_BASE_DIR/logs" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    
    # 清理卷备份
    find "$BACKUP_BASE_DIR/volumes" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_BASE_DIR/volumes" -type d -empty -delete
    
    log "旧备份清理完成"
}

# 生成备份报告
generate_backup_report() {
    log "生成备份报告..."
    
    local report_file="$BACKUP_BASE_DIR/backup_report_$DATE.txt"
    
    cat > "$report_file" <<EOF
股票分析系统备份报告
==================

备份时间: $(date)
备份类型: 完整备份
命名空间: $NAMESPACE

备份内容:
EOF
    
    # 统计备份文件
    if [ -f "$BACKUP_BASE_DIR/database/$DATE/stock_analysis_db_$DATE.sql.gz" ]; then
        local db_size=$(du -h "$BACKUP_BASE_DIR/database/$DATE/stock_analysis_db_$DATE.sql.gz" | cut -f1)
        echo "- 数据库备份: $db_size" >> "$report_file"
    fi
    
    if [ -f "$BACKUP_BASE_DIR/redis/$DATE/redis_dump_$DATE.rdb.gz" ]; then
        local redis_size=$(du -h "$BACKUP_BASE_DIR/redis/$DATE/redis_dump_$DATE.rdb.gz" | cut -f1)
        echo "- Redis备份: $redis_size" >> "$report_file"
    fi
    
    if [ -f "$BACKUP_BASE_DIR/configs/$DATE.tar.gz" ]; then
        local config_size=$(du -h "$BACKUP_BASE_DIR/configs/$DATE.tar.gz" | cut -f1)
        echo "- 配置备份: $config_size" >> "$report_file"
    fi
    
    if [ -f "$BACKUP_BASE_DIR/logs/$DATE.tar.gz" ]; then
        local log_size=$(du -h "$BACKUP_BASE_DIR/logs/$DATE.tar.gz" | cut -f1)
        echo "- 日志备份: $log_size" >> "$report_file"
    fi
    
    # 计算总大小
    local total_size=$(du -sh "$BACKUP_BASE_DIR" | cut -f1)
    echo "" >> "$report_file"
    echo "总备份大小: $total_size" >> "$report_file"
    echo "备份位置: $BACKUP_BASE_DIR" >> "$report_file"
    echo "保留天数: $RETENTION_DAYS 天" >> "$report_file"
    
    log "备份报告生成完成: $report_file"
}

# 发送备份通知
send_backup_notification() {
    log "发送备份通知..."
    
    local report_file="$BACKUP_BASE_DIR/backup_report_$DATE.txt"
    
    # 发送邮件通知（需要配置邮件服务）
    if command -v mail &> /dev/null; then
        mail -s "股票分析系统备份完成 - $DATE" ops@company.com < "$report_file"
    fi
    
    # 发送Slack通知（需要配置Webhook）
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"📦 股票分析系统备份完成\\n时间: $DATE\\n状态: 成功\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    log "备份通知发送完成"
}

# 主函数
main() {
    log "开始执行系统备份..."
    
    # 检查必要的工具
    if ! command -v kubectl &> /dev/null; then
        error "kubectl 未安装"
    fi
    
    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        error "无法连接到Kubernetes集群"
    fi
    
    # 执行备份步骤
    create_backup_dirs
    backup_database
    backup_redis
    backup_k8s_configs
    backup_logs
    backup_persistent_volumes
    cleanup_old_backups
    generate_backup_report
    send_backup_notification
    
    log "系统备份完成！"
    log "备份位置: $BACKUP_BASE_DIR"
    log "备份时间: $DATE"
}

# 处理命令行参数
case "${1:-}" in
    "database")
        log "仅备份数据库..."
        create_backup_dirs
        backup_database
        ;;
    "redis")
        log "仅备份Redis..."
        create_backup_dirs
        backup_redis
        ;;
    "configs")
        log "仅备份配置..."
        create_backup_dirs
        backup_k8s_configs
        ;;
    "full"|"")
        main
        ;;
    *)
        echo "用法: $0 [database|redis|configs|full]"
        echo "  database - 仅备份数据库"
        echo "  redis    - 仅备份Redis"
        echo "  configs  - 仅备份配置"
        echo "  full     - 完整备份（默认）"
        exit 1
        ;;
esac