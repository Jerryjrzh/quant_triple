#!/bin/bash

# Stock Analysis System Deployment Script
# This script provides deployment automation for different environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="stock-analysis-system"
REGISTRY="ghcr.io"
IMAGE_NAME="stock-analysis-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Stock Analysis System Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker images
    deploy          Deploy to Kubernetes
    rollback        Rollback to previous version
    status          Check deployment status
    logs            View application logs
    cleanup         Clean up resources

Options:
    -e, --environment   Target environment (staging|production) [default: staging]
    -t, --tag          Docker image tag [default: latest]
    -n, --namespace    Kubernetes namespace [default: stock-analysis-system]
    -d, --dry-run      Show what would be done without executing
    -v, --verbose      Enable verbose output
    -h, --help         Show this help message

Examples:
    $0 build
    $0 deploy -e production -t v1.2.3
    $0 rollback -e staging
    $0 status -e production
    $0 logs -e staging api
    $0 cleanup -e staging

EOF
}

# Parse command line arguments
ENVIRONMENT="staging"
IMAGE_TAG="latest"
DRY_RUN=false
VERBOSE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        build|deploy|rollback|status|logs|cleanup)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed (for build command)
    if [[ "$COMMAND" == "build" ]] && ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build backend image
    log_info "Building backend image..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ."
    else
        docker build -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" .
        docker tag "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:latest"
    fi
    
    # Build frontend image
    log_info "Building frontend image..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: docker build -t ${REGISTRY}/${IMAGE_NAME}-frontend:${IMAGE_TAG} frontend/"
    else
        docker build -t "${REGISTRY}/${IMAGE_NAME}-frontend:${IMAGE_TAG}" frontend/
        docker tag "${REGISTRY}/${IMAGE_NAME}-frontend:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}-frontend:latest"
    fi
    
    log_success "Docker images built successfully"
}

# Push Docker images
push_images() {
    log_info "Pushing Docker images..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would push images to registry"
        return
    fi
    
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    docker push "${REGISTRY}/${IMAGE_NAME}-frontend:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}-frontend:latest"
    
    log_success "Docker images pushed successfully"
}

# Deploy to Kubernetes
deploy_to_k8s() {
    log_info "Deploying to Kubernetes environment: $ENVIRONMENT"
    
    cd "$PROJECT_ROOT"
    
    # Create namespace if it doesn't exist
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create namespace: $NAMESPACE"
    else
        kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Update image tags in deployment files
    local temp_dir=$(mktemp -d)
    cp -r k8s/* "$temp_dir/"
    
    # Replace image tags
    sed -i "s|stock-analysis-system:latest|${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" "$temp_dir/api-deployment.yaml"
    sed -i "s|stock-analysis-frontend:latest|${REGISTRY}/${IMAGE_NAME}-frontend:${IMAGE_TAG}|g" "$temp_dir/frontend-deployment.yaml"
    sed -i "s|stock-analysis-system:latest|${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" "$temp_dir/celery-deployment.yaml"
    
    # Apply environment-specific configurations
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Production-specific configurations
        sed -i "s|replicas: 3|replicas: 5|g" "$temp_dir/api-deployment.yaml"
        sed -i "s|replicas: 2|replicas: 3|g" "$temp_dir/frontend-deployment.yaml"
        sed -i "s|replicas: 3|replicas: 5|g" "$temp_dir/celery-deployment.yaml"
    fi
    
    # Deploy components in order
    local components=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "postgresql.yaml"
        "redis.yaml"
        "api-deployment.yaml"
        "frontend-deployment.yaml"
        "nginx-deployment.yaml"
        "celery-deployment.yaml"
        "monitoring.yaml"
        "hpa.yaml"
        "network-policy.yaml"
        "rbac.yaml"
    )
    
    for component in "${components[@]}"; do
        log_info "Deploying $component..."
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would apply: kubectl apply -f $temp_dir/$component -n $NAMESPACE"
        else
            kubectl apply -f "$temp_dir/$component" -n "$NAMESPACE"
        fi
    done
    
    # Wait for deployments to be ready
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Waiting for deployments to be ready..."
        kubectl rollout status deployment/stock-analysis-api -n "$NAMESPACE" --timeout=600s
        kubectl rollout status deployment/stock-analysis-frontend -n "$NAMESPACE" --timeout=600s
        kubectl rollout status deployment/nginx-proxy -n "$NAMESPACE" --timeout=300s
        kubectl rollout status deployment/celery-worker -n "$NAMESPACE" --timeout=300s
    fi
    
    # Cleanup temp directory
    rm -rf "$temp_dir"
    
    log_success "Deployment completed successfully"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment in environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback deployments"
        return
    fi
    
    # Rollback main deployments
    kubectl rollout undo deployment/stock-analysis-api -n "$NAMESPACE"
    kubectl rollout undo deployment/stock-analysis-frontend -n "$NAMESPACE"
    kubectl rollout undo deployment/celery-worker -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/stock-analysis-api -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/stock-analysis-frontend -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/celery-worker -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed successfully"
}

# Check deployment status
check_status() {
    log_info "Checking deployment status in environment: $ENVIRONMENT"
    
    echo
    echo "=== Namespace Status ==="
    kubectl get namespace "$NAMESPACE" 2>/dev/null || log_warning "Namespace $NAMESPACE not found"
    
    echo
    echo "=== Pod Status ==="
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo
    echo "=== Service Status ==="
    kubectl get services -n "$NAMESPACE"
    
    echo
    echo "=== Deployment Status ==="
    kubectl get deployments -n "$NAMESPACE"
    
    echo
    echo "=== HPA Status ==="
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || log_info "No HPA found"
    
    echo
    echo "=== Ingress Status ==="
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || log_info "No Ingress found"
    
    # Check if services are healthy
    echo
    echo "=== Health Check ==="
    local api_pod=$(kubectl get pods -n "$NAMESPACE" -l app=stock-analysis-api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$api_pod" ]]; then
        kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f http://localhost:8000/health 2>/dev/null && \
            log_success "API health check passed" || \
            log_warning "API health check failed"
    else
        log_warning "No API pod found"
    fi
}

# View application logs
view_logs() {
    local component="${1:-api}"
    log_info "Viewing logs for component: $component in environment: $ENVIRONMENT"
    
    case $component in
        api)
            kubectl logs -f deployment/stock-analysis-api -n "$NAMESPACE"
            ;;
        frontend)
            kubectl logs -f deployment/stock-analysis-frontend -n "$NAMESPACE"
            ;;
        worker)
            kubectl logs -f deployment/celery-worker -n "$NAMESPACE"
            ;;
        beat)
            kubectl logs -f deployment/celery-beat -n "$NAMESPACE"
            ;;
        nginx)
            kubectl logs -f deployment/nginx-proxy -n "$NAMESPACE"
            ;;
        postgres)
            kubectl logs -f deployment/postgresql -n "$NAMESPACE"
            ;;
        redis)
            kubectl logs -f deployment/redis -n "$NAMESPACE"
            ;;
        *)
            log_error "Unknown component: $component"
            log_info "Available components: api, frontend, worker, beat, nginx, postgres, redis"
            exit 1
            ;;
    esac
}

# Cleanup resources
cleanup_resources() {
    log_info "Cleaning up resources in environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would delete namespace: $NAMESPACE"
        return
    fi
    
    read -p "Are you sure you want to delete namespace '$NAMESPACE' and all its resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        log_success "Cleanup completed successfully"
    else
        log_info "Cleanup cancelled"
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local api_service=$(kubectl get service stock-analysis-api -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    if [[ -n "$api_service" ]]; then
        # Port forward for health check
        kubectl port-forward service/stock-analysis-api 8080:8000 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        
        # Run health checks
        if curl -f http://localhost:8080/health &>/dev/null; then
            log_success "API health check passed"
        else
            log_error "API health check failed"
        fi
        
        if curl -f http://localhost:8080/ready &>/dev/null; then
            log_success "API readiness check passed"
        else
            log_error "API readiness check failed"
        fi
        
        # Cleanup port forward
        kill $port_forward_pid 2>/dev/null || true
    else
        log_warning "API service not found"
    fi
}

# Main execution
main() {
    log_info "Starting deployment script for Stock Analysis System"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    log_info "Command: $COMMAND"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    check_prerequisites
    
    case $COMMAND in
        build)
            build_images
            ;;
        deploy)
            build_images
            push_images
            deploy_to_k8s
            run_health_checks
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            check_status
            ;;
        logs)
            view_logs "$2"
            ;;
        cleanup)
            cleanup_resources
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Script completed successfully"
}

# Run main function
main "$@"