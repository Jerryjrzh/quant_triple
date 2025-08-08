#!/bin/bash
"""
灾难恢复脚本

用于在灾难情况下快速恢复股票分析系统。
支持完整恢复和部分恢复。
"""

set -e

# 配置变量
BACKUP_BASE_DIR="/backups"
NAMESPACE="stock-analysis"
RECOVERY_MODE="${1:-full}"
BACKUP_DATE="${2:-latest}"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
    exit 1
}

warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >&2
}

# 获取最新备份日期
get_latest_backup() {
    if [ "$BACKUP_DATE" = "latest" ]; then
        BACKUP_DATE=$(ls -1 "$BACKUP_BASE_DIR/database" | sort -r | head -n1)
        if [ -z "$BACKUP_DATE" ]; then
            error "未找到可用的备份"
        fi
        log "使用最新备份: $BACKUP_DATE"
    fi
    
    # 验证备份存在
    if [ ! -d "$BACKUP_BASE_DIR/database/$BACKUP_DATE" ]; then
        error "备份目录不存在: $BACKUP_BASE_DIR/database/$BACKUP_DATE"
    fi
}

# 预检查
pre_recovery_checks() {
    log "执行恢复前检查..."
    
    # 检查kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl 未安装"
    fi
    
    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        error "无法连接到Kubernetes集群"
    fi
    
    # 检查命名空间
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "创建命名空间: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # 检查备份文件完整性
    check_backup_integrity
    
    log "预检查完成"
}

# 检查备份完整性
check_backup_integrity() {
    log "检查备份文件完整性..."
    
    local db_backup="$BACKUP_BASE_DIR/database/$BACKUP_DATE/stock_analysis_db_$BACKUP_DATE.sql.gz"
    local db_checksum="$BACKUP_BASE_DIR/database/$BACKUP_DATE/stock_analysis_db_$BACKUP_DATE.sql.gz.md5"
    
    if [ -f "$db_backup" ] && [ -f "$db_checksum" ]; then
        if ! md5sum -c "$db_checksum" &> /dev/null; then
            error "数据库备份文件校验失败"
        fi
        log "数据库备份文件校验通过"
    else
        warning "数据库备份文件或校验和不存在"
    fi
    
    local redis_backup="$BACKUP_BASE_DIR/redis/$BACKUP_DATE/redis_dump_$BACKUP_DATE.rdb.gz"
    local redis_checksum="$BACKUP_BASE_DIR/redis/$BACKUP_DATE/redis_dump_$BACKUP_DATE.rdb.gz.md5"
    
    if [ -f "$redis_backup" ] && [ -f "$redis_checksum" ]; then
        if ! md5sum -c "$redis_checksum" &> /dev/null; then
            error "Redis备份文件校验失败"
        fi
        log "Redis备份文件校验通过"
    else
        warning "Redis备份文件或校验和不存在"
    fi
}

# 恢复Kubernetes配置
restore_k8s_configs() {
    log "恢复Kubernetes配置..."
    
    local config_backup="$BACKUP_BASE_DIR/configs/$BACKUP_DATE.tar.gz"
    
    if [ ! -f "$config_backup" ]; then
        error "配置备份文件不存在: $config_backup"
    fi
    
    # 解压配置文件
    local temp_dir=$(mktemp -d)
    tar -xzf "$config_backup" -C "$temp_dir"
    
    # 恢复配置（按顺序）
    local config_dir="$temp_dir/$BACKUP_DATE"
    
    # 1. 恢复PVC（必须先恢复）
    if [ -f "$config_dir/pvc.yaml" ]; then
        log "恢复持久化卷声明..."
        kubectl apply -f "$config_dir/pvc.yaml"
        
        # 等待PVC绑定
        kubectl wait --for=condition=Bound pvc --all -n "$NAMESPACE" --timeout=300s
    fi
    
    # 2. 恢复ConfigMaps和Secrets
    if [ -f "$config_dir/configmaps.yaml" ]; then
        log "恢复ConfigMaps..."
        kubectl apply -f "$config_dir/configmaps.yaml"
    fi
    
    if [ -f "$config_dir/secrets.yaml" ]; then
        log "恢复Secrets..."
        kubectl apply -f "$config_dir/secrets.yaml"
    fi
    
    # 3. 恢复网络策略
    if [ -f "$config_dir/networkpolicies.yaml" ]; then
        log "恢复网络策略..."
        kubectl apply -f "$config_dir/networkpolicies.yaml"
    fi
    
    # 4. 恢复Ingress
    if [ -f "$config_dir/ingress.yaml" ]; then
        log "恢复Ingress..."
        kubectl apply -f "$config_dir/ingress.yaml"
    fi
    
    # 清理临时目录
    rm -rf "$temp_dir"
    
    log "Kubernetes配置恢复完成"
}

# 恢复数据库
restore_database() {
    log "恢复PostgreSQL数据库..."
    
    local db_backup="$BACKUP_BASE_DIR/database/$BACKUP_DATE/stock_analysis_db_$BACKUP_DATE.sql.gz"
    
    if [ ! -f "$db_backup" ]; then
        error "数据库备份文件不存在: $db_backup"
    fi
    
    # 部署PostgreSQL（如果不存在）
    if ! kubectl get statefulset postgresql -n "$NAMESPACE" &> /dev/null; then
        log "部署PostgreSQL..."
        kubectl apply -f k8s/postgresql.yaml
        
        # 等待PostgreSQL就绪
        kubectl wait --for=condition=Ready pod/postgresql-0 -n "$NAMESPACE" --timeout=600s
    fi
    
    # 停止依赖服务
    log "停止依赖服务..."
    kubectl scale deployment stock-analysis-api --replicas=0 -n "$NAMESPACE" 2>/dev/null || true
    kubectl scale deployment stock-analysis-celery --replicas=0 -n "$NAMESPACE" 2>/dev/null || true
    
    # 等待Pod停止
    sleep 30
    
    # 恢复数据库
    log "开始恢复数据库数据..."
    
    # 解压备份文件
    local temp_sql=$(mktemp)
    gunzip -c "$db_backup" > "$temp_sql"
    
    # 删除现有数据库（如果存在）
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -c "DROP DATABASE IF EXISTS stock_analysis;" || true
    
    # 创建新数据库
    kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -c "CREATE DATABASE stock_analysis;"
    
    # 恢复数据
    kubectl exec -i -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis < "$temp_sql"
    
    # 清理临时文件
    rm -f "$temp_sql"
    
    # 验证恢复
    local table_count=$(kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
    
    if [ "$table_count" -gt 0 ]; then
        log "数据库恢复成功，共恢复 $table_count 个表"
    else
        error "数据库恢复失败，未找到任何表"
    fi
}

# 恢复Redis
restore_redis() {
    log "恢复Redis数据..."
    
    local redis_backup="$BACKUP_BASE_DIR/redis/$BACKUP_DATE/redis_dump_$BACKUP_DATE.rdb.gz"
    
    if [ ! -f "$redis_backup" ]; then
        warning "Redis备份文件不存在: $redis_backup"
        return
    fi
    
    # 部署Redis（如果不存在）
    if ! kubectl get statefulset redis -n "$NAMESPACE" &> /dev/null; then
        log "部署Redis..."
        kubectl apply -f k8s/redis.yaml
        
        # 等待Redis就绪
        kubectl wait --for=condition=Ready pod/redis-0 -n "$NAMESPACE" --timeout=300s
    fi
    
    # 停止Redis
    kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli SHUTDOWN NOSAVE || true
    
    # 等待Redis停止
    sleep 10
    
    # 解压并复制RDB文件
    local temp_rdb=$(mktemp)
    gunzip -c "$redis_backup" > "$temp_rdb"
    
    # 复制RDB文件到Redis容器
    kubectl cp "$temp_rdb" "$NAMESPACE/redis-0:/data/dump.rdb"
    
    # 清理临时文件
    rm -f "$temp_rdb"
    
    # 重启Redis Pod
    kubectl delete pod redis-0 -n "$NAMESPACE"
    kubectl wait --for=condition=Ready pod/redis-0 -n "$NAMESPACE" --timeout=300s
    
    # 验证恢复
    local key_count=$(kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli DBSIZE)
    log "Redis恢复成功，共恢复 $key_count 个键"
}

# 恢复应用服务
restore_applications() {
    log "恢复应用服务..."
    
    # 恢复所有应用
    local config_backup="$BACKUP_BASE_DIR/configs/$BACKUP_DATE.tar.gz"
    
    if [ -f "$config_backup" ]; then
        local temp_dir=$(mktemp -d)
        tar -xzf "$config_backup" -C "$temp_dir"
        
        local config_dir="$temp_dir/$BACKUP_DATE"
        
        # 恢复所有资源（除了已经恢复的）
        if [ -f "$config_dir/all-resources.yaml" ]; then
            log "恢复应用资源..."
            
            # 过滤掉PVC、ConfigMap、Secret等已恢复的资源
            kubectl apply -f "$config_dir/all-resources.yaml" || true
        fi
        
        rm -rf "$temp_dir"
    fi
    
    # 等待应用启动
    log "等待应用服务启动..."
    
    # 等待API服务
    if kubectl get deployment stock-analysis-api -n "$NAMESPACE" &> /dev/null; then
        kubectl wait --for=condition=Available deployment/stock-analysis-api -n "$NAMESPACE" --timeout=600s
        log "API服务启动完成"
    fi
    
    # 等待Celery服务
    if kubectl get deployment stock-analysis-celery -n "$NAMESPACE" &> /dev/null; then
        kubectl wait --for=condition=Available deployment/stock-analysis-celery -n "$NAMESPACE" --timeout=600s
        log "Celery服务启动完成"
    fi
    
    # 等待前端服务
    if kubectl get deployment stock-analysis-frontend -n "$NAMESPACE" &> /dev/null; then
        kubectl wait --for=condition=Available deployment/stock-analysis-frontend -n "$NAMESPACE" --timeout=600s
        log "前端服务启动完成"
    fi
}

# 验证恢复结果
verify_recovery() {
    log "验证恢复结果..."
    
    # 检查Pod状态
    log "检查Pod状态..."
    kubectl get pods -n "$NAMESPACE"
    
    # 检查服务状态
    log "检查服务状态..."
    kubectl get services -n "$NAMESPACE"
    
    # 健康检查
    local api_service=$(kubectl get service stock-analysis-api -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [ ! -z "$api_service" ]; then
        # 等待服务就绪
        sleep 30
        
        # 测试API健康检查
        if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -n "$NAMESPACE" -- curl -f "http://$api_service:8000/health" &> /dev/null; then
            log "✅ API健康检查通过"
        else
            warning "❌ API健康检查失败"
        fi
    fi
    
    # 检查数据库连接
    if kubectl exec -n "$NAMESPACE" postgresql-0 -- psql -U postgres -d stock_analysis -c "SELECT 1;" &> /dev/null; then
        log "✅ 数据库连接正常"
    else
        warning "❌ 数据库连接失败"
    fi
    
    # 检查Redis连接
    if kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli ping | grep -q PONG; then
        log "✅ Redis连接正常"
    else
        warning "❌ Redis连接失败"
    fi
    
    log "恢复验证完成"
}

# 生成恢复报告
generate_recovery_report() {
    log "生成恢复报告..."
    
    local report_file="disaster_recovery_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" <<EOF
股票分析系统灾难恢复报告
======================

恢复时间: $(date)
恢复模式: $RECOVERY_MODE
使用备份: $BACKUP_DATE
命名空间: $NAMESPACE

恢复内容:
- Kubernetes配置: $([ -f "$BACKUP_BASE_DIR/configs/$BACKUP_DATE.tar.gz" ] && echo "✅ 已恢复" || echo "❌ 未恢复")
- PostgreSQL数据库: $([ -f "$BACKUP_BASE_DIR/database/$BACKUP_DATE/stock_analysis_db_$BACKUP_DATE.sql.gz" ] && echo "✅ 已恢复" || echo "❌ 未恢复")
- Redis缓存: $([ -f "$BACKUP_BASE_DIR/redis/$BACKUP_DATE/redis_dump_$BACKUP_DATE.rdb.gz" ] && echo "✅ 已恢复" || echo "❌ 未恢复")

系统状态:
EOF
    
    # 添加Pod状态
    echo "" >> "$report_file"
    echo "Pod状态:" >> "$report_file"
    kubectl get pods -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "无法获取Pod状态" >> "$report_file"
    
    # 添加服务状态
    echo "" >> "$report_file"
    echo "服务状态:" >> "$report_file"
    kubectl get services -n "$NAMESPACE" >> "$report_file" 2>/dev/null || echo "无法获取服务状态" >> "$report_file"
    
    log "恢复报告生成完成: $report_file"
}

# 发送恢复通知
send_recovery_notification() {
    log "发送恢复通知..."
    
    # 发送邮件通知
    if command -v mail &> /dev/null; then
        echo "股票分析系统灾难恢复完成。恢复时间: $(date)，使用备份: $BACKUP_DATE" | \
            mail -s "股票分析系统灾难恢复完成" ops@company.com
    fi
    
    # 发送Slack通知
    if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚨 股票分析系统灾难恢复完成\\n时间: $(date)\\n备份: $BACKUP_DATE\\n状态: 成功\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    log "恢复通知发送完成"
}

# 完整恢复
full_recovery() {
    log "开始完整灾难恢复..."
    
    get_latest_backup
    pre_recovery_checks
    restore_k8s_configs
    restore_database
    restore_redis
    restore_applications
    verify_recovery
    generate_recovery_report
    send_recovery_notification
    
    log "完整灾难恢复完成！"
}

# 数据库恢复
database_recovery() {
    log "开始数据库恢复..."
    
    get_latest_backup
    pre_recovery_checks
    restore_database
    verify_recovery
    
    log "数据库恢复完成！"
}

# Redis恢复
redis_recovery() {
    log "开始Redis恢复..."
    
    get_latest_backup
    pre_recovery_checks
    restore_redis
    verify_recovery
    
    log "Redis恢复完成！"
}

# 配置恢复
config_recovery() {
    log "开始配置恢复..."
    
    get_latest_backup
    pre_recovery_checks
    restore_k8s_configs
    restore_applications
    verify_recovery
    
    log "配置恢复完成！"
}

# 主函数
main() {
    case "$RECOVERY_MODE" in
        "full")
            full_recovery
            ;;
        "database")
            database_recovery
            ;;
        "redis")
            redis_recovery
            ;;
        "config")
            config_recovery
            ;;
        *)
            echo "用法: $0 [full|database|redis|config] [backup_date]"
            echo "  full     - 完整恢复（默认）"
            echo "  database - 仅恢复数据库"
            echo "  redis    - 仅恢复Redis"
            echo "  config   - 仅恢复配置"
            echo ""
            echo "backup_date: 备份日期（格式：YYYYMMDD_HHMMSS），默认为latest"
            echo ""
            echo "示例:"
            echo "  $0 full                    # 使用最新备份进行完整恢复"
            echo "  $0 database 20250808_120000 # 恢复指定日期的数据库备份"
            exit 1
            ;;
    esac
}

# 确认操作
if [ "$RECOVERY_MODE" != "help" ] && [ "$RECOVERY_MODE" != "--help" ]; then
    echo "⚠️  警告：此操作将恢复股票分析系统，可能会覆盖现有数据！"
    echo "恢复模式: $RECOVERY_MODE"
    echo "备份日期: $BACKUP_DATE"
    echo ""
    read -p "确认继续？(yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "操作已取消"
        exit 0
    fi
fi

# 执行恢复
main