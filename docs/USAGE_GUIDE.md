# 爬虫接口集成系统使用指南

## 目录

1. [快速开始](#快速开始)
2. [基础概念](#基础概念)
3. [数据获取指南](#数据获取指南)
4. [系统监控指南](#系统监控指南)
5. [最佳实践](#最佳实践)
6. [常见问题](#常见问题)
7. [示例代码](#示例代码)

## 快速开始

### 环境要求

- Python 3.8+
- Redis 6.0+
- PostgreSQL 12+
- 网络连接（访问外部数据源）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等信息
```

4. **初始化数据库**
```bash
alembic upgrade head
```

5. **启动服务**
```bash
python start_server.py
```

### 第一个API调用

```python
import requests

# 获取认证令牌
auth_response = requests.post('http://localhost:8000/api/v1/auth/login', json={
    'username': 'admin',
    'password': 'password'
})
token = auth_response.json()['access_token']

# 获取股票实时数据
headers = {'Authorization': f'Bearer {token}'}
response = requests.get(
    'http://localhost:8000/api/v1/data/realtime/000001.SZ',
    headers=headers
)
data = response.json()
print(data)
```

## 基础概念

### 数据源

系统支持多个数据源，包括：

- **Tushare**: 专业的金融数据接口，数据质量高
- **AKShare**: 开源的金融数据接口，免费使用
- **Wind**: 万得金融终端接口，机构级数据

### 数据类型

- **stock_realtime**: 股票实时行情
- **stock_history**: 股票历史行情
- **fund_flow**: 资金流向数据
- **dragon_tiger**: 龙虎榜数据
- **limitup_reason**: 涨停原因
- **etf_data**: ETF基金数据

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   客户端应用    │    │   API网关       │    │   数据处理层    │
│                 │───▶│                 │───▶│                 │
│ Web/Mobile/SDK  │    │ 认证/限流/路由  │    │ 数据获取/清洗   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   缓存层        │    │   数据源层      │
                       │                 │◀───│                 │
                       │ Redis/Memory    │    │ Tushare/AKShare │
                       └─────────────────┘    └─────────────────┘
```

## 数据获取指南

### 1. 实时行情数据

实时行情数据提供股票的最新价格、成交量等信息。

**基础用法**:
```python
import requests

def get_realtime_data(symbol, token):
    """获取股票实时行情"""
    url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['data']
    else:
        print(f"Error: {response.status_code}")
        return None

# 使用示例
data = get_realtime_data('000001.SZ', your_token)
print(f"股票价格: {data['price']}")
print(f"涨跌幅: {data['change_pct']}%")
```

**高级用法 - 指定字段**:
```python
def get_realtime_data_fields(symbol, fields, token):
    """获取指定字段的实时数据"""
    url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
    params = {'fields': ','.join(fields)}
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, params=params, headers=headers)
    return response.json()['data']

# 只获取价格和成交量
data = get_realtime_data_fields('000001.SZ', ['price', 'volume'], your_token)
```

### 2. 历史行情数据

历史行情数据用于技术分析和回测。

**基础用法**:
```python
def get_history_data(symbol, start_date, end_date, token):
    """获取历史行情数据"""
    url = f'http://localhost:8000/api/v1/data/history/{symbol}'
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'frequency': 'daily'
    }
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, params=params, headers=headers)
    return response.json()['data']

# 获取2024年全年数据
history = get_history_data('000001.SZ', '2024-01-01', '2024-12-31', your_token)
```

**分页获取大量数据**:
```python
def get_all_history_data(symbol, start_date, end_date, token):
    """分页获取所有历史数据"""
    all_data = []
    page = 1
    page_size = 1000
    
    while True:
        url = f'http://localhost:8000/api/v1/data/history/{symbol}'
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'page': page,
            'page_size': page_size
        }
        headers = {'Authorization': f'Bearer {token}'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        all_data.extend(data['data'])
        
        # 检查是否还有更多数据
        if len(data['data']) < page_size:
            break
        page += 1
    
    return all_data
```

### 3. 龙虎榜数据

龙虎榜数据显示大额交易信息。

```python
def get_dragon_tiger_data(date, token, symbol=None):
    """获取龙虎榜数据"""
    url = 'http://localhost:8000/api/v1/data/dragon-tiger'
    params = {'date': date}
    if symbol:
        params['symbol'] = symbol
    
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, params=params, headers=headers)
    return response.json()['data']

# 获取某日所有龙虎榜数据
dragon_tiger = get_dragon_tiger_data('2024-01-01', your_token)

# 分析机构买入情况
for item in dragon_tiger:
    institutions = item.get('institutions', [])
    institution_buy = sum(inst['buy_amount'] for inst in institutions 
                         if inst['type'] == 'institution')
    if institution_buy > 0:
        print(f"{item['name']}: 机构买入 {institution_buy/10000:.2f}万元")
```

### 4. 资金流向数据

资金流向数据帮助分析主力资金动向。

```python
def get_fund_flow_data(symbol, period, token):
    """获取资金流向数据"""
    url = f'http://localhost:8000/api/v1/data/fund-flow/{symbol}'
    params = {'period': period}
    headers = {'Authorization': f'Bearer {token}'}
    
    response = requests.get(url, params=params, headers=headers)
    return response.json()['data']

def analyze_fund_flow(symbol, token):
    """分析资金流向趋势"""
    periods = ['1d', '3d', '5d', '10d', '20d']
    flow_data = {}
    
    for period in periods:
        data = get_fund_flow_data(symbol, period, token)
        flow_data[period] = data['main_net_inflow']
    
    # 分析趋势
    print(f"{symbol} 主力资金流向:")
    for period, amount in flow_data.items():
        direction = "流入" if amount > 0 else "流出"
        print(f"{period}: {direction} {abs(amount)/10000:.2f}万元")
    
    return flow_data

# 分析平安银行资金流向
analyze_fund_flow('000001.SZ', your_token)
```

### 5. 批量数据获取

批量获取多只股票数据，提高效率。

```python
def batch_get_realtime_data(symbols, token):
    """批量获取实时数据"""
    url = 'http://localhost:8000/api/v1/data/batch'
    requests_data = [
        {'data_type': 'stock_realtime', 'symbol': symbol}
        for symbol in symbols
    ]
    
    payload = {
        'requests': requests_data,
        'options': {'parallel': True, 'timeout': 30}
    }
    
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# 批量获取多只股票数据
symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
batch_result = batch_get_realtime_data(symbols, your_token)

# 处理结果
for result in batch_result['results']:
    if result['success']:
        data = result['data']
        print(f"{data['symbol']}: {data['price']} ({data['change_pct']:+.2f}%)")
    else:
        print(f"{result['symbol']}: 获取失败 - {result['error']}")
```

## 系统监控指南

### 1. 健康检查

定期检查系统健康状态。

```python
def check_system_health():
    """检查系统健康状态"""
    url = 'http://localhost:8000/api/v1/health'
    response = requests.get(url)
    health = response.json()
    
    print(f"系统状态: {health['status']}")
    print("组件状态:")
    
    for component, status in health['components'].items():
        status_icon = "✅" if status['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {component}: {status['status']} "
              f"({status['response_time']:.3f}s)")
    
    return health['status'] == 'healthy'

# 定期健康检查
import time
while True:
    if not check_system_health():
        print("⚠️ 系统异常，请检查！")
    time.sleep(60)  # 每分钟检查一次
```

### 2. 性能监控

监控系统性能指标。

```python
def get_system_status(token):
    """获取系统状态"""
    url = 'http://localhost:8000/api/v1/system/status'
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    return response.json()

def monitor_performance(token):
    """监控系统性能"""
    status = get_system_status(token)
    metrics = status['performance_metrics']
    
    print("性能指标:")
    print(f"  请求/秒: {metrics['requests_per_second']}")
    print(f"  平均响应时间: {metrics['average_response_time']:.3f}s")
    print(f"  错误率: {metrics['error_rate']:.2%}")
    print(f"  缓存命中率: {metrics['cache_hit_rate']:.2%}")
    
    # 性能告警
    if metrics['average_response_time'] > 1.0:
        print("⚠️ 响应时间过长！")
    if metrics['error_rate'] > 0.05:
        print("⚠️ 错误率过高！")
    
    return metrics

# 持续监控
while True:
    monitor_performance(your_token)
    time.sleep(30)
```

### 3. 缓存管理

管理和优化缓存使用。

```python
def get_cache_stats(token):
    """获取缓存统计"""
    url = 'http://localhost:8000/api/v1/system/cache/stats'
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    return response.json()

def optimize_cache(token):
    """缓存优化建议"""
    stats = get_cache_stats(token)
    cache_stats = stats['cache_stats']
    
    print("缓存统计:")
    print(f"  总键数: {cache_stats['total_keys']}")
    print(f"  内存使用: {cache_stats['memory_usage']}")
    print(f"  命中率: {cache_stats['hit_rate']:.2%}")
    
    # 优化建议
    if cache_stats['hit_rate'] < 0.8:
        print("💡 建议: 缓存命中率较低，考虑调整缓存策略")
    
    if cache_stats['memory_usage_pct'] > 0.9:
        print("💡 建议: 内存使用率过高，考虑清理缓存")
        # 清理旧缓存
        clear_cache(token, 'realtime_data', 'old:*')
    
    return stats

def clear_cache(token, cache_type, pattern):
    """清理缓存"""
    url = 'http://localhost:8000/api/v1/system/cache'
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'cache_type': cache_type, 'pattern': pattern}
    
    response = requests.delete(url, json=payload, headers=headers)
    result = response.json()
    print(f"清理了 {result['cleared_keys']} 个缓存键")
```

## 最佳实践

### 1. 错误处理

```python
import time
import random
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    # 指数退避
                    wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"请求失败，{wait_time:.2f}秒后重试...")
                    time.sleep(wait_time)
            
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def robust_api_call(url, headers, params=None):
    """健壮的API调用"""
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()
```

### 2. 数据缓存

```python
import pickle
import os
from datetime import datetime, timedelta

class DataCache:
    """本地数据缓存"""
    
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, key, max_age_hours=1):
        """获取缓存数据"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        # 检查缓存是否过期
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            os.remove(cache_path)
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def set(self, key, data):
        """设置缓存数据"""
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

# 使用缓存
cache = DataCache()

def get_cached_realtime_data(symbol, token):
    """带缓存的实时数据获取"""
    cache_key = f"realtime_{symbol}"
    
    # 尝试从缓存获取
    cached_data = cache.get(cache_key, max_age_hours=0.1)  # 6分钟缓存
    if cached_data:
        print(f"从缓存获取 {symbol} 数据")
        return cached_data
    
    # 从API获取
    data = get_realtime_data(symbol, token)
    if data:
        cache.set(cache_key, data)
    
    return data
```

### 3. 数据验证

```python
def validate_stock_data(data):
    """验证股票数据完整性"""
    required_fields = ['symbol', 'price', 'volume', 'timestamp']
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"缺少必需字段: {field}")
    
    # 数据合理性检查
    if data['price'] <= 0:
        raise ValueError("股价不能为负数或零")
    
    if data['volume'] < 0:
        raise ValueError("成交量不能为负数")
    
    # 时间戳检查
    try:
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        if timestamp > datetime.now():
            raise ValueError("时间戳不能是未来时间")
    except ValueError as e:
        raise ValueError(f"时间戳格式错误: {e}")
    
    return True

def safe_get_realtime_data(symbol, token):
    """安全的数据获取"""
    try:
        data = get_realtime_data(symbol, token)
        if data:
            validate_stock_data(data)
        return data
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None
```

### 4. 批量处理优化

```python
import asyncio
import aiohttp

async def async_get_realtime_data(session, symbol, token):
    """异步获取实时数据"""
    url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return {'symbol': symbol, 'success': True, 'data': data['data']}
            else:
                return {'symbol': symbol, 'success': False, 'error': f'HTTP {response.status}'}
    except Exception as e:
        return {'symbol': symbol, 'success': False, 'error': str(e)}

async def batch_get_realtime_async(symbols, token):
    """异步批量获取实时数据"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_get_realtime_data(session, symbol, token) for symbol in symbols]
        results = await asyncio.gather(*tasks)
    
    return results

# 使用异步批量获取
symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'] * 10  # 40只股票
results = asyncio.run(batch_get_realtime_async(symbols, your_token))

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"成功获取: {len(successful)}, 失败: {len(failed)}")
```

## 常见问题

### Q1: 如何处理API限流？

**A**: 系统有内置限流机制，建议：

1. 使用批量接口减少请求次数
2. 实现请求间隔控制
3. 监控响应头中的限流信息

```python
def rate_limited_request(url, headers, delay=0.1):
    """限流请求"""
    response = requests.get(url, headers=headers)
    
    # 检查限流头
    if 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers['X-RateLimit-Remaining'])
        if remaining < 10:  # 剩余请求数较少时增加延迟
            delay *= 2
    
    time.sleep(delay)
    return response
```

### Q2: 数据源不可用时怎么办？

**A**: 系统有自动故障转移机制，但也可以手动指定数据源：

```python
def get_data_with_fallback(symbol, token):
    """带故障转移的数据获取"""
    sources = ['tushare', 'akshare', 'wind']
    
    for source in sources:
        try:
            url = f'http://localhost:8000/api/v1/data/realtime/{symbol}'
            params = {'source': source}
            headers = {'Authorization': f'Bearer {token}'}
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                return response.json()['data']
        except Exception as e:
            print(f"数据源 {source} 失败: {e}")
            continue
    
    raise Exception("所有数据源都不可用")
```

### Q3: 如何优化大量历史数据的获取？

**A**: 使用分页和并行处理：

```python
import concurrent.futures
from datetime import datetime, timedelta

def get_history_chunk(symbol, start_date, end_date, token):
    """获取历史数据块"""
    return get_history_data(symbol, start_date, end_date, token)

def get_large_history_data(symbol, start_date, end_date, token, chunk_months=3):
    """并行获取大量历史数据"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 分割时间段
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_months*30), end)
        chunks.append((current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        current = chunk_end + timedelta(days=1)
    
    # 并行获取
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(get_history_chunk, symbol, chunk_start, chunk_end, token)
            for chunk_start, chunk_end in chunks
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result()
                all_data.extend(data)
            except Exception as e:
                print(f"获取数据块失败: {e}")
    
    # 按日期排序
    all_data.sort(key=lambda x: x['date'])
    return all_data
```

## 示例代码

### 完整的股票分析示例

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockAnalyzer:
    """股票分析器"""
    
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def get_stock_data(self, symbol, days=30):
        """获取股票数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        url = f'{self.base_url}/data/history/{symbol}'
        params = {'start_date': start_date, 'end_date': end_date}
        
        response = requests.get(url, params=params, headers=self.headers)
        return response.json()['data']
    
    def calculate_ma(self, data, window=5):
        """计算移动平均线"""
        df = pd.DataFrame(data)
        df['ma'] = df['close'].rolling(window=window).mean()
        return df
    
    def analyze_trend(self, symbol):
        """趋势分析"""
        # 获取数据
        data = self.get_stock_data(symbol, days=60)
        df = self.calculate_ma(data, window=5)
        
        # 计算技术指标
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # 生成信号
        latest = df.iloc[-1]
        signals = []
        
        if latest['close'] > latest['ma5']:
            signals.append("价格突破5日均线")
        
        if latest['rsi'] > 70:
            signals.append("RSI超买")
        elif latest['rsi'] < 30:
            signals.append("RSI超卖")
        
        return {
            'symbol': symbol,
            'latest_price': latest['close'],
            'ma5': latest['ma5'],
            'ma20': latest['ma20'],
            'rsi': latest['rsi'],
            'signals': signals
        }
    
    def calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def plot_analysis(self, symbol):
        """绘制分析图表"""
        data = self.get_stock_data(symbol, days=60)
        df = self.calculate_ma(data, window=5)
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 价格图
        ax1.plot(df.index, df['close'], label='收盘价', linewidth=2)
        ax1.plot(df.index, df['ma'], label='MA5', alpha=0.7)
        ax1.plot(df.index, df['ma20'], label='MA20', alpha=0.7)
        ax1.set_title(f'{symbol} 价格走势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI图
        rsi = self.calculate_rsi(df['close'])
        ax2.plot(df.index, rsi, label='RSI', color='orange')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='超买线')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='超卖线')
        ax2.set_title('RSI指标')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用示例
analyzer = StockAnalyzer('http://localhost:8000/api/v1', your_token)

# 分析单只股票
analysis = analyzer.analyze_trend('000001.SZ')
print(f"股票: {analysis['symbol']}")
print(f"最新价格: {analysis['latest_price']}")
print(f"技术信号: {', '.join(analysis['signals'])}")

# 绘制图表
analyzer.plot_analysis('000001.SZ')

# 批量分析
symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
for symbol in symbols:
    try:
        analysis = analyzer.analyze_trend(symbol)
        print(f"\n{symbol}: {analysis['latest_price']} - {', '.join(analysis['signals'])}")
    except Exception as e:
        print(f"{symbol}: 分析失败 - {e}")
```

### WebSocket实时数据示例

```python
import websocket
import json
import threading

class RealtimeDataClient:
    """实时数据客户端"""
    
    def __init__(self, ws_url, token):
        self.ws_url = ws_url
        self.token = token
        self.ws = None
        self.subscriptions = set()
    
    def on_message(self, ws, message):
        """处理接收到的消息"""
        data = json.loads(message)
        
        if data['type'] == 'data':
            self.handle_data(data)
        elif data['type'] == 'error':
            print(f"错误: {data['message']}")
    
    def on_error(self, ws, error):
        """处理错误"""
        print(f"WebSocket错误: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭"""
        print("WebSocket连接已关闭")
    
    def on_open(self, ws):
        """连接建立"""
        print("WebSocket连接已建立")
        
        # 发送认证
        auth_msg = {
            'type': 'auth',
            'token': self.token
        }
        ws.send(json.dumps(auth_msg))
    
    def handle_data(self, data):
        """处理实时数据"""
        channel = data['channel']
        stock_data = data['data']
        
        print(f"{channel}: {stock_data['symbol']} "
              f"价格={stock_data['price']} "
              f"涨跌={stock_data['change']:+.2f} "
              f"({stock_data['change_pct']:+.2f}%)")
    
    def connect(self):
        """建立连接"""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # 在新线程中运行
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
    
    def subscribe(self, symbols):
        """订阅股票数据"""
        channels = [f'stock.{symbol}' for symbol in symbols]
        self.subscriptions.update(channels)
        
        subscribe_msg = {
            'type': 'subscribe',
            'channels': channels
        }
        
        if self.ws:
            self.ws.send(json.dumps(subscribe_msg))
    
    def unsubscribe(self, symbols):
        """取消订阅"""
        channels = [f'stock.{symbol}' for symbol in symbols]
        
        unsubscribe_msg = {
            'type': 'unsubscribe',
            'channels': channels
        }
        
        if self.ws:
            self.ws.send(json.dumps(unsubscribe_msg))
        
        self.subscriptions -= set(channels)

# 使用示例
client = RealtimeDataClient('ws://localhost:8000/ws/realtime', your_token)
client.connect()

# 订阅股票
client.subscribe(['000001.SZ', '000002.SZ', '600000.SH'])

# 保持连接
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("退出程序")
```

这个使用指南提供了从基础到高级的完整示例，帮助用户快速上手并掌握系统的各种功能。每个示例都包含了错误处理、性能优化等最佳实践。