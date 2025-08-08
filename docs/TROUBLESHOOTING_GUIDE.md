# 爬虫接口集成系统故障排除指南

## 目录

1. [常见问题快速诊断](#常见问题快速诊断)
2. [连接和认证问题](#连接和认证问题)
3. [数据获取问题](#数据获取问题)
4. [性能问题](#性能问题)
5. [系统监控和诊断](#系统监控和诊断)
6. [错误代码参考](#错误代码参考)
7. [日志分析](#日志分析)
8. [调试工具](#调试工具)

## 常见问题快速诊断

### 问题诊断流程图

```
开始
  ↓
能否访问健康检查接口？
  ├─ 否 → 检查服务是否启动 → 检查端口和防火墙
  ↓
能否正常登录？
  ├─ 否 → 检查用户名密码 → 检查认证服务
  ↓
能否获取数据？
  ├─ 否 → 检查数据源状态 → 检查网络连接
  ↓
数据是否及时更新？
  ├─ 否 → 检查缓存设置 → 检查数据源频率
  ↓
问题解决
```

### 快速检查清单

- [ ] 服务是否正常运行 (`curl http://localhost:8000/health`)
- [ ] 数据库连接是否正常
- [ ] Redis缓存是否可用
- [ ] 外部数据源是否可访问
- [ ] 认证令牌是否有效
- [ ] 网络连接是否稳定
- [ ] 系统资源是否充足

## 连接和认证问题

### 1. 无法连接到服务

**症状**: 请求超时或连接拒绝

**可能原因**:
- 服务未启动
- 端口被占用或防火墙阻止
- 网络配置问题

**解决方案**:

```bash
# 检查服务状态
ps aux | grep python
netstat -tlnp | grep 8000

# 检查端口占用
lsof -i :8000

# 启动服务
python start_server.py

# 检查防火墙
sudo ufw status
sudo iptables -L
```### 2
. 认证失败

**症状**: 401 Unauthorized 错误

**可能原因**:
- 用户名或密码错误
- 令牌过期
- 认证服务异常

**解决方案**:

```python
# 测试登录
import requests

response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    'username': 'admin',
    'password': 'password'
})

if response.status_code == 401:
    print("用户名或密码错误")
elif response.status_code == 500:
    print("认证服务异常")
else:
    print("登录成功")
```

**检查步骤**:
1. 验证用户名和密码
2. 检查令牌是否过期
3. 查看认证服务日志
4. 重新获取令牌

### 3. 令牌过期

**症状**: 请求返回 401，提示令牌无效

**解决方案**:
```python
def refresh_token_if_needed(client):
    """检查并刷新令牌"""
    try:
        # 测试当前令牌
        response = client.session.get(f"{client.base_url}/system/status")
        if response.status_code == 401:
            print("令牌已过期，重新登录...")
            return client.login("admin", "password")
        return True
    except Exception as e:
        print(f"令牌检查失败: {e}")
        return False
```## 数
据获取问题

### 1. 数据源不可用

**症状**: 503 Service Unavailable 或数据返回为空

**诊断步骤**:

```python
def diagnose_data_source():
    """诊断数据源问题"""
    import requests
    
    # 检查系统状态
    response = requests.get('http://localhost:8000/api/v1/system/data-sources')
    data_sources = response.json()['data_sources']
    
    for source in data_sources:
        print(f"数据源: {source['name']}")
        print(f"  状态: {source['status']}")
        print(f"  健康: {source['health']['status']}")
        print(f"  响应时间: {source['health']['response_time']:.3f}s")
        
        if source['status'] != 'active':
            print(f"  ⚠️ 数据源 {source['name']} 不可用")
        
        if source['health']['response_time'] > 2.0:
            print(f"  ⚠️ 数据源 {source['name']} 响应缓慢")

diagnose_data_source()
```

**解决方案**:
1. 检查网络连接
2. 验证API密钥和配额
3. 切换到备用数据源
4. 联系数据源提供商

### 2. 数据格式错误

**症状**: 数据解析失败或字段缺失

**调试代码**:
```python
def debug_data_format(symbol):
    """调试数据格式问题"""
    import requests
    import json
    
    url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
    headers = {'Authorization': 'Bearer your_token'}
    
    response = requests.get(url, headers=headers)
    
    print(f"状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    
    try:
        data = response.json()
        print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # 检查必需字段
        required_fields = ['symbol', 'price', 'change', 'volume']
        if 'data' in data:
            missing_fields = [f for f in required_fields if f not in data['data']]
            if missing_fields:
                print(f"⚠️ 缺失字段: {missing_fields}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        print(f"原始响应: {response.text}")

debug_data_format('000001.SZ')
```#
## 3. 缓存问题

**症状**: 数据不更新或返回过期数据

**诊断和解决**:

```python
def diagnose_cache_issues():
    """诊断缓存问题"""
    import requests
    
    # 获取缓存统计
    response = requests.get('http://localhost:8000/api/v1/system/cache/stats')
    cache_stats = response.json()['cache_stats']
    
    print("缓存统计:")
    print(f"  命中率: {cache_stats['hit_rate']:.2%}")
    print(f"  内存使用: {cache_stats['memory_usage']}")
    print(f"  总键数: {cache_stats['total_keys']}")
    
    # 检查缓存问题
    if cache_stats['hit_rate'] < 0.5:
        print("⚠️ 缓存命中率过低，可能存在配置问题")
    
    if cache_stats['memory_usage_pct'] > 0.9:
        print("⚠️ 缓存内存使用率过高")
        
        # 清理缓存
        clear_response = requests.delete(
            'http://localhost:8000/api/v1/system/cache',
            json={'cache_type': 'realtime_data', 'pattern': 'old:*'}
        )
        print(f"清理结果: {clear_response.json()}")

def force_refresh_data(symbol):
    """强制刷新数据（绕过缓存）"""
    import requests
    import time
    
    url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
    headers = {'Authorization': 'Bearer your_token'}
    
    # 添加时间戳参数绕过缓存
    params = {'_t': int(time.time())}
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()
```

## 性能问题

### 1. 响应时间过长

**症状**: API请求响应时间超过5秒

**性能分析工具**:

```python
import time
import requests
from contextlib import contextmanager

@contextmanager
def timer():
    """计时器上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    print(f"执行时间: {end - start:.3f}秒")

def performance_test():
    """性能测试"""
    symbols = ['000001.SZ', '000002.SZ', '600000.SH']
    
    # 单个请求测试
    print("单个请求性能测试:")
    for symbol in symbols:
        with timer():
            response = requests.get(f'http://localhost:8000/api/v1/data/realtime/{symbol}')
            print(f"  {symbol}: {response.status_code}")
    
    # 批量请求测试
    print("\n批量请求性能测试:")
    with timer():
        batch_data = {
            'requests': [{'data_type': 'stock_realtime', 'symbol': s} for s in symbols]
        }
        response = requests.post('http://localhost:8000/api/v1/data/batch', json=batch_data)
        print(f"  批量请求: {response.status_code}")

performance_test()
```#
## 2. 内存使用过高

**症状**: 系统内存使用率超过80%

**诊断步骤**:

```bash
# 检查系统内存使用
free -h
top -p $(pgrep -f "python.*start_server")

# 检查Python进程内存
ps aux --sort=-%mem | head -10

# 使用memory_profiler分析内存使用
pip install memory-profiler
python -m memory_profiler your_script.py
```

**内存优化**:

```python
import gc
import psutil
import os

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RSS内存: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS内存: {memory_info.vms / 1024 / 1024:.2f} MB")
    print(f"内存百分比: {process.memory_percent():.2f}%")
    
    # 强制垃圾回收
    collected = gc.collect()
    print(f"垃圾回收: {collected} 个对象")

def optimize_memory():
    """内存优化建议"""
    # 1. 限制数据缓存大小
    # 2. 定期清理过期数据
    # 3. 使用生成器而不是列表
    # 4. 及时关闭数据库连接
    pass
```

### 3. 数据库连接问题

**症状**: 数据库连接超时或连接池耗尽

**诊断工具**:

```python
def diagnose_database():
    """诊断数据库问题"""
    import psycopg2
    from sqlalchemy import create_engine, text
    
    try:
        # 测试数据库连接
        engine = create_engine('postgresql://user:pass@localhost/dbname')
        
        with engine.connect() as conn:
            # 检查连接数
            result = conn.execute(text("""
                SELECT count(*) as active_connections 
                FROM pg_stat_activity 
                WHERE state = 'active'
            """))
            
            active_connections = result.fetchone()[0]
            print(f"活跃连接数: {active_connections}")
            
            # 检查长时间运行的查询
            result = conn.execute(text("""
                SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
                FROM pg_stat_activity 
                WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
            """))
            
            long_queries = result.fetchall()
            if long_queries:
                print("长时间运行的查询:")
                for query in long_queries:
                    print(f"  PID: {query[0]}, 时长: {query[1]}")
            
    except Exception as e:
        print(f"数据库诊断失败: {e}")

diagnose_database()
```## 系统监控和诊断


### 1. 健康检查脚本

```python
#!/usr/bin/env python3
"""
系统健康检查脚本
"""

import requests
import json
import sys
from datetime import datetime

def health_check():
    """执行健康检查"""
    checks = {
        'api_server': check_api_server,
        'database': check_database,
        'cache': check_cache,
        'data_sources': check_data_sources
    }
    
    results = {}
    overall_healthy = True
    
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            results[check_name] = result
            if not result['healthy']:
                overall_healthy = False
        except Exception as e:
            results[check_name] = {'healthy': False, 'error': str(e)}
            overall_healthy = False
    
    # 输出结果
    print(f"健康检查报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    for check_name, result in results.items():
        status = "✅" if result['healthy'] else "❌"
        print(f"{status} {check_name}: {result.get('message', 'OK')}")
        
        if 'details' in result:
            for key, value in result['details'].items():
                print(f"    {key}: {value}")
    
    print(f"\n整体状态: {'健康' if overall_healthy else '异常'}")
    return overall_healthy

def check_api_server():
    """检查API服务器"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'healthy': data['status'] == 'healthy',
                'message': f"API服务器状态: {data['status']}",
                'details': {
                    '响应时间': f"{response.elapsed.total_seconds():.3f}s"
                }
            }
        else:
            return {
                'healthy': False,
                'message': f"API服务器返回错误: {response.status_code}"
            }
    except Exception as e:
        return {
            'healthy': False,
            'message': f"无法连接API服务器: {e}"
        }

def check_database():
    """检查数据库"""
    try:
        response = requests.get('http://localhost:8000/api/v1/system/status')
        # 这里应该包含数据库状态检查逻辑
        return {'healthy': True, 'message': '数据库连接正常'}
    except:
        return {'healthy': False, 'message': '数据库连接失败'}

def check_cache():
    """检查缓存"""
    try:
        response = requests.get('http://localhost:8000/api/v1/system/cache/stats')
        if response.status_code == 200:
            stats = response.json()['cache_stats']
            return {
                'healthy': True,
                'message': '缓存服务正常',
                'details': {
                    '命中率': f"{stats['hit_rate']:.2%}",
                    '内存使用': stats['memory_usage']
                }
            }
        else:
            return {'healthy': False, 'message': '缓存服务异常'}
    except:
        return {'healthy': False, 'message': '无法连接缓存服务'}

def check_data_sources():
    """检查数据源"""
    try:
        response = requests.get('http://localhost:8000/api/v1/system/data-sources')
        if response.status_code == 200:
            sources = response.json()['data_sources']
            healthy_sources = sum(1 for s in sources if s['status'] == 'active')
            total_sources = len(sources)
            
            return {
                'healthy': healthy_sources > 0,
                'message': f'数据源状态: {healthy_sources}/{total_sources} 可用',
                'details': {
                    source['name']: source['status'] for source in sources
                }
            }
        else:
            return {'healthy': False, 'message': '无法获取数据源状态'}
    except:
        return {'healthy': False, 'message': '数据源检查失败'}

if __name__ == "__main__":
    healthy = health_check()
    sys.exit(0 if healthy else 1)
```#
## 2. 日志分析工具

```python
#!/usr/bin/env python3
"""
日志分析工具
"""

import re
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter

class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.error_patterns = {
            'connection_error': r'Connection.*error|connection.*failed',
            'timeout_error': r'timeout|timed out',
            'auth_error': r'authentication.*failed|unauthorized',
            'data_error': r'data.*error|invalid.*data',
            'database_error': r'database.*error|sql.*error'
        }
    
    def analyze_errors(self, hours=24):
        """分析错误日志"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        error_counts = defaultdict(int)
        error_details = defaultdict(list)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if 'ERROR' in line or 'CRITICAL' in line:
                        # 解析时间戳
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            timestamp_str = timestamp_match.group(1)
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            
                            if timestamp > cutoff_time:
                                # 分类错误
                                error_type = 'unknown'
                                for error_name, pattern in self.error_patterns.items():
                                    if re.search(pattern, line, re.IGNORECASE):
                                        error_type = error_name
                                        break
                                
                                error_counts[error_type] += 1
                                error_details[error_type].append({
                                    'timestamp': timestamp_str,
                                    'message': line.strip()
                                })
        
        except FileNotFoundError:
            print(f"日志文件不存在: {self.log_file}")
            return None
        
        return {
            'error_counts': dict(error_counts),
            'error_details': dict(error_details),
            'total_errors': sum(error_counts.values())
        }
    
    def analyze_performance(self, hours=24):
        """分析性能日志"""
        response_times = []
        slow_requests = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    # 查找响应时间信息
                    time_match = re.search(r'response_time[:\s]+(\d+\.?\d*)ms', line)
                    if time_match:
                        response_time = float(time_match.group(1))
                        response_times.append(response_time)
                        
                        if response_time > 1000:  # 超过1秒的请求
                            slow_requests.append({
                                'response_time': response_time,
                                'line': line.strip()
                            })
        
        except FileNotFoundError:
            return None
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            return {
                'average_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'total_requests': len(response_times),
                'slow_requests': len(slow_requests),
                'slow_request_details': slow_requests[:10]  # 只显示前10个
            }
        
        return {'message': '未找到性能数据'}
    
    def generate_report(self):
        """生成分析报告"""
        print("日志分析报告")
        print("=" * 50)
        
        # 错误分析
        error_analysis = self.analyze_errors()
        if error_analysis:
            print(f"\n📊 错误统计 (最近24小时):")
            print(f"总错误数: {error_analysis['total_errors']}")
            
            for error_type, count in error_analysis['error_counts'].items():
                print(f"  {error_type}: {count}")
            
            # 显示最近的错误
            print(f"\n🔍 最近错误详情:")
            for error_type, details in error_analysis['error_details'].items():
                if details:
                    latest = details[-1]
                    print(f"  {error_type}: {latest['timestamp']}")
                    print(f"    {latest['message'][:100]}...")
        
        # 性能分析
        perf_analysis = self.analyze_performance()
        if perf_analysis and 'average_response_time' in perf_analysis:
            print(f"\n⚡ 性能统计:")
            print(f"平均响应时间: {perf_analysis['average_response_time']:.2f}ms")
            print(f"最大响应时间: {perf_analysis['max_response_time']:.2f}ms")
            print(f"总请求数: {perf_analysis['total_requests']}")
            print(f"慢请求数: {perf_analysis['slow_requests']}")

# 使用示例
if __name__ == "__main__":
    analyzer = LogAnalyzer('/var/log/stock_analysis.log')
    analyzer.generate_report()
```## 错误
代码参考

### HTTP状态码

| 状态码 | 含义 | 常见原因 | 解决方案 |
|--------|------|----------|----------|
| 400 | 请求参数错误 | 参数格式不正确、缺少必需参数 | 检查请求参数格式和完整性 |
| 401 | 认证失败 | 令牌无效或过期 | 重新登录获取新令牌 |
| 403 | 权限不足 | 用户权限不够 | 检查用户权限设置 |
| 404 | 资源不存在 | 股票代码不存在、接口路径错误 | 验证股票代码和API路径 |
| 429 | 请求频率超限 | 超过API调用限制 | 降低请求频率或使用批量接口 |
| 500 | 服务器内部错误 | 服务器异常 | 查看服务器日志，联系技术支持 |
| 503 | 服务不可用 | 数据源不可用、系统维护 | 等待服务恢复或切换数据源 |

### 业务错误代码

| 错误代码 | 描述 | 解决方案 |
|----------|------|----------|
| DATA_NOT_FOUND | 请求的数据不存在 | 检查股票代码和日期范围 |
| DATA_SOURCE_UNAVAILABLE | 数据源不可用 | 等待数据源恢复或使用其他数据源 |
| INVALID_SYMBOL | 无效的股票代码 | 使用正确的股票代码格式 |
| INVALID_DATE_RANGE | 无效的日期范围 | 检查日期格式和范围 |
| QUOTA_EXCEEDED | 配额超限 | 等待配额重置或升级账户 |
| CACHE_ERROR | 缓存错误 | 清理缓存或重启缓存服务 |

## 调试工具

### 1. API测试工具

```python
#!/usr/bin/env python3
"""
API测试工具
"""

import requests
import json
import time
from datetime import datetime

class APITester:
    """API测试器"""
    
    def __init__(self, base_url, token=None):
        self.base_url = base_url
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def test_endpoint(self, method, endpoint, **kwargs):
        """测试单个端点"""
        url = f"{self.base_url}{endpoint}"
        
        print(f"\n🧪 测试: {method.upper()} {endpoint}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            elapsed = time.time() - start_time
            
            print(f"状态码: {response.status_code}")
            print(f"响应时间: {elapsed:.3f}s")
            print(f"响应大小: {len(response.content)} bytes")
            
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    data = response.json()
                    print(f"响应数据: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}...")
                except:
                    print("响应不是有效的JSON")
            
            return response
            
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def run_basic_tests(self):
        """运行基础测试"""
        tests = [
            ('GET', '/health'),
            ('POST', '/auth/login', {'json': {'username': 'admin', 'password': 'password'}}),
            ('GET', '/data/realtime/000001.SZ'),
            ('GET', '/system/status'),
        ]
        
        results = []
        for method, endpoint, *args in tests:
            kwargs = args[0] if args else {}
            response = self.test_endpoint(method, endpoint, **kwargs)
            results.append({
                'endpoint': endpoint,
                'success': response is not None and response.status_code < 400
            })
        
        # 汇总结果
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"\n📊 测试结果: {successful}/{total} 通过")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['endpoint']}")

# 使用示例
if __name__ == "__main__":
    tester = APITester('http://localhost:8000/api/v1')
    tester.run_basic_tests()
```### 2
. 网络诊断工具

```bash
#!/bin/bash
# 网络诊断脚本

echo "🌐 网络诊断工具"
echo "================"

# 检查基本连接
echo "1. 检查本地服务连接:"
curl -s -o /dev/null -w "HTTP状态: %{http_code}, 响应时间: %{time_total}s\n" http://localhost:8000/health

# 检查DNS解析
echo -e "\n2. 检查DNS解析:"
nslookup api.tushare.pro
nslookup akshare.akfamily.xyz

# 检查外部API连接
echo -e "\n3. 检查外部API连接:"
curl -s -o /dev/null -w "Tushare API - HTTP状态: %{http_code}, 响应时间: %{time_total}s\n" https://api.tushare.pro
curl -s -o /dev/null -w "AKShare API - HTTP状态: %{http_code}, 响应时间: %{time_total}s\n" https://akshare.akfamily.xyz

# 检查端口占用
echo -e "\n4. 检查端口占用:"
netstat -tlnp | grep :8000
netstat -tlnp | grep :5432  # PostgreSQL
netstat -tlnp | grep :6379  # Redis

# 检查防火墙
echo -e "\n5. 检查防火墙状态:"
if command -v ufw &> /dev/null; then
    sudo ufw status
elif command -v iptables &> /dev/null; then
    sudo iptables -L INPUT | grep -E "(8000|5432|6379)"
fi

echo -e "\n✅ 网络诊断完成"
```

### 3. 系统资源监控

```python
#!/usr/bin/env python3
"""
系统资源监控工具
"""

import psutil
import time
import json
from datetime import datetime

class ResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.history = []
    
    def collect_metrics(self):
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 网络统计
        net_io = psutil.net_io_counters()
        
        # 进程信息
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'processes': processes
        }
        
        self.history.append(metrics)
        return metrics
    
    def print_current_status(self):
        """打印当前状态"""
        metrics = self.collect_metrics()
        
        print(f"📊 系统资源状态 - {metrics['timestamp']}")
        print("=" * 50)
        
        # CPU
        cpu_status = "🔴" if metrics['cpu']['percent'] > 80 else "🟡" if metrics['cpu']['percent'] > 60 else "🟢"
        print(f"{cpu_status} CPU使用率: {metrics['cpu']['percent']:.1f}% ({metrics['cpu']['count']} 核)")
        
        # 内存
        mem_percent = metrics['memory']['percent']
        mem_status = "🔴" if mem_percent > 80 else "🟡" if mem_percent > 60 else "🟢"
        print(f"{mem_status} 内存使用率: {mem_percent:.1f}% ({metrics['memory']['used']/1024/1024/1024:.1f}GB / {metrics['memory']['total']/1024/1024/1024:.1f}GB)")
        
        # 磁盘
        disk_percent = metrics['disk']['percent']
        disk_status = "🔴" if disk_percent > 80 else "🟡" if disk_percent > 60 else "🟢"
        print(f"{disk_status} 磁盘使用率: {disk_percent:.1f}% ({metrics['disk']['used']/1024/1024/1024:.1f}GB / {metrics['disk']['total']/1024/1024/1024:.1f}GB)")
        
        # Python进程
        if metrics['processes']:
            print(f"\n🐍 Python进程:")
            for proc in metrics['processes'][:5]:  # 显示前5个
                print(f"  PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu_percent']:.1f}%, 内存: {proc['memory_percent']:.1f}%")
    
    def monitor_continuously(self, duration=300, interval=10):
        """持续监控"""
        print(f"🔄 开始持续监控 {duration}秒，每{interval}秒更新一次")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.print_current_status()
            print("\n" + "-" * 50)
            time.sleep(interval)
        
        print("✅ 监控完成")
    
    def export_history(self, filename):
        """导出历史数据"""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📄 历史数据已导出到: {filename}")

# 使用示例
if __name__ == "__main__":
    monitor = ResourceMonitor()
    
    # 显示当前状态
    monitor.print_current_status()
    
    # 可选：持续监控
    # monitor.monitor_continuously(duration=60, interval=5)
```

## 联系支持

如果以上故障排除步骤无法解决问题，请联系技术支持：

- **技术支持邮箱**: support@stockanalysis.com
- **问题反馈**: https://github.com/stockanalysis/issues
- **文档中心**: https://docs.stockanalysis.com
- **社区论坛**: https://community.stockanalysis.com

### 提交问题时请包含以下信息：

1. 问题详细描述
2. 错误信息和日志
3. 系统环境信息
4. 重现步骤
5. 期望的结果

### 获取系统信息脚本：

```bash
#!/bin/bash
echo "系统信息收集脚本"
echo "=================="
echo "操作系统: $(uname -a)"
echo "Python版本: $(python --version)"
echo "内存信息: $(free -h | grep Mem)"
echo "磁盘空间: $(df -h / | tail -1)"
echo "网络连接: $(netstat -tlnp | grep :8000)"
echo "进程信息: $(ps aux | grep python | head -5)"
```