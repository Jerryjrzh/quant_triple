#!/usr/bin/env python3
"""
部署和配置验证脚本

验证Docker容器化部署、Kubernetes集群部署和生产环境配置的正确性。
"""

import subprocess
import json
import time
import requests
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """部署验证器"""
    
    def __init__(self):
        self.results = []
    
    def validate_docker_deployment(self):
        """验证Docker部署"""
        logger.info("🐳 验证Docker部署")
        
        try:
            # 检查Docker是否运行
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self._record_result("Docker安装", True, result.stdout.strip())
            else:
                self._record_result("Docker安装", False, "Docker未安装或未运行")
                return
            
            # 验证Dockerfile存在
            dockerfile_path = Path('Dockerfile')
            if dockerfile_path.exists():
                self._record_result("Dockerfile存在", True)
                
                # 验证Dockerfile内容
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                
                if 'FROM python:' in dockerfile_content:
                    self._record_result("Dockerfile-基础镜像", True)
                else:
                    self._record_result("Dockerfile-基础镜像", False, "未使用Python基础镜像")
                
                if 'COPY requirements.txt' in dockerfile_content:
                    self._record_result("Dockerfile-依赖安装", True)
                else:
                    self._record_result("Dockerfile-依赖安装", False, "未复制requirements.txt")
                    
                if 'EXPOSE' in dockerfile_content:
                    self._record_result("Dockerfile-端口暴露", True)
                else:
                    self._record_result("Dockerfile-端口暴露", False, "未暴露端口")
            else:
                self._record_result("Dockerfile存在", False, "Dockerfile不存在")
            
            # 验证docker-compose.yml
            compose_path = Path('docker-compose.yml')
            if compose_path.exists():
                self._record_result("Docker Compose配置", True)
                
                try:
                    with open(compose_path, 'r') as f:
                        compose_content = f.read()
                    
                    if 'services:' in compose_content:
                        self._record_result("Docker Compose-服务定义", True)
                    else:
                        self._record_result("Docker Compose-服务定义", False, "未定义服务")
                        
                except Exception as e:
                    self._record_result("Docker Compose-配置解析", False, str(e))
            else:
                self._record_result("Docker Compose配置", False, "docker-compose.yml不存在")
                
        except Exception as e:
            self._record_result("Docker部署验证", False, str(e))
    
    def validate_kubernetes_deployment(self):
        """验证Kubernetes部署"""
        logger.info("☸️ 验证Kubernetes部署")
        
        try:
            # 验证K8s配置文件存在性和格式
            k8s_files = [
                'k8s/api-deployment.yaml',
                'k8s/frontend-deployment.yaml', 
                'k8s/celery-deployment.yaml',
                'k8s/nginx-deployment.yaml',
                'k8s/postgresql.yaml',
                'k8s/redis.yaml',
                'k8s/configmap.yaml',
                'k8s/secrets.yaml'
            ]
            
            valid_files = 0
            for file_path in k8s_files:
                if Path(file_path).exists():
                    try:
                        with open(file_path, 'r') as f:
                            # 处理多文档YAML
                            yaml_documents = list(yaml.safe_load_all(f))
                        
                        # 验证至少有一个有效的K8s资源
                        valid_docs = 0
                        for doc in yaml_documents:
                            if doc and 'apiVersion' in doc and 'kind' in doc:
                                valid_docs += 1
                        
                        if valid_docs > 0:
                            self._record_result(f"K8s配置-{Path(file_path).name}", True, f"包含{valid_docs}个资源")
                            valid_files += 1
                        else:
                            self._record_result(f"K8s配置-{Path(file_path).name}", False, "缺少必要字段")
                            
                    except yaml.YAMLError as e:
                        self._record_result(f"K8s配置-{Path(file_path).name}", False, f"YAML格式错误: {e}")
                else:
                    self._record_result(f"K8s配置-{Path(file_path).name}", False, "文件不存在")
            
            # 验证整体配置完整性
            if valid_files >= 6:
                self._record_result("K8s配置完整性", True, f"有效配置文件: {valid_files}/{len(k8s_files)}")
            else:
                self._record_result("K8s配置完整性", False, f"配置文件不足: {valid_files}/{len(k8s_files)}")
                    
        except Exception as e:
            self._record_result("Kubernetes部署验证", False, str(e))
    
    def validate_production_config(self):
        """验证生产环境配置"""
        logger.info("🏭 验证生产环境配置")
        
        # 检查环境变量配置
        required_env_vars = [
            'DATABASE_URL', 'REDIS_URL', 'SECRET_KEY',
            'TUSHARE_API_KEY', 'AKSHARE_API_KEY'
        ]
        
        env_file = Path('.env')
        env_example_file = Path('.env.example')
        
        if env_file.exists():
            self._record_result("环境配置文件", True, ".env文件存在")
            
            with open(env_file, 'r') as f:
                env_content = f.read()
            
            configured_vars = 0
            for var in required_env_vars:
                if var in env_content and f"{var}=" in env_content:
                    # 检查是否有实际值（不是空的）
                    lines = env_content.split('\n')
                    for line in lines:
                        if line.startswith(f"{var}=") and len(line.split('=', 1)) > 1:
                            value = line.split('=', 1)[1].strip()
                            if value and value != '""' and value != "''":
                                self._record_result(f"环境变量-{var}", True, "已配置")
                                configured_vars += 1
                                break
                    else:
                        self._record_result(f"环境变量-{var}", False, "已定义但无值")
                else:
                    self._record_result(f"环境变量-{var}", False, "未配置")
            
            # 验证配置完整性
            if configured_vars >= 3:  # 至少配置了3个关键变量
                self._record_result("环境配置完整性", True, f"已配置: {configured_vars}/{len(required_env_vars)}")
            else:
                self._record_result("环境配置完整性", False, f"配置不足: {configured_vars}/{len(required_env_vars)}")
        else:
            self._record_result("环境配置文件", False, ".env文件不存在")
        
        # 检查示例配置文件
        if env_example_file.exists():
            self._record_result("环境配置示例", True, ".env.example文件存在")
        else:
            self._record_result("环境配置示例", False, ".env.example文件不存在")
        
        # 验证配置文件结构
        config_files = [
            'config/settings.py',
            'alembic.ini',
            'pyproject.toml',
            'requirements.txt'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                self._record_result(f"配置文件-{Path(config_file).name}", True)
            else:
                self._record_result(f"配置文件-{Path(config_file).name}", False, "文件不存在")
    
    def _record_result(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        result = {
            'name': test_name,
            'passed': passed,
            'details': details
        }
        self.results.append(result)
        
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
    
    def generate_report(self):
        """生成验证报告"""
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n部署验证完成: {success_rate:.1f}% ({passed}/{total})")
        return success_rate >= 80

def main():
    validator = DeploymentValidator()
    
    validator.validate_docker_deployment()
    validator.validate_kubernetes_deployment()
    validator.validate_production_config()
    
    success = validator.generate_report()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())