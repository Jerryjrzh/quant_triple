# 爬虫接口集成系统 API 文档

## 概述

本文档详细描述了爬虫接口集成系统的所有API接口，包括数据获取、健康检查、系统管理等功能。系统采用RESTful API设计，支持JSON格式的数据交换。

## 基础信息

- **基础URL**: `http://localhost:8000/api/v1`
- **认证方式**: Bearer Token
- **数据格式**: JSON
- **字符编码**: UTF-8
- **API版本**: v1.0

## 认证

所有API请求都需要在请求头中包含认证令牌：

```http
Authorization: Bearer <your_token_here>
```

### 获取认证令牌

```http
POST /auth/login
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

**响应示例**:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

## 数据获取接口

### 1. 统一数据请求接口

获取各种类型的市场数据，支持多数据源自动切换。

```http
POST /data/request
Content-Type: application/json
Authorization: Bearer <token>

{
    "data_type": "stock_realtime",
    "symbol": "000001.SZ",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "source": "tushare",
    "parameters": {
        "fields": ["open", "high", "low", "close", "volume"],
        "frequency": "daily"
    }
}
```

**请求参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| data_type | string | 是 | 数据类型：stock_realtime, stock_history, fund_flow, dragon_tiger, limitup_reason, etf_data |
| symbol | string | 是 | 股票代码或ETF代码 |
| start_date | string | 否 | 开始日期 (YYYY-MM-DD) |
| end_date | string | 否 | 结束日期 (YYYY-MM-DD) |
| source | string | 否 | 指定数据源，不指定则自动选择 |
| parameters | object | 否 | 额外参数 |

**响应示例**:
```json
{
    "success": true,
    "data": [
        {
            "date": "2024-01-01",
            "symbol": "000001.SZ",
            "open": 10.50,
            "high": 10.80,
            "low": 10.30,
            "close": 10.65,
            "volume": 1000000
        }
    ],
    "metadata": {
        "source": "tushare",
        "total_records": 1,
        "query_time": "2024-01-01T10:00:00Z",
        "cache_hit": false
    }
}
```

### 2. 实时行情数据

获取股票实时行情数据。

```http
GET /data/realtime/{symbol}
Authorization: Bearer <token>
```

**路径参数**:
- `symbol`: 股票代码 (例如: 000001.SZ)

**查询参数**:
- `fields`: 返回字段列表，逗号分隔 (可选)
- `source`: 指定数据源 (可选)

**响应示例**:
```json
{
    "success": true,
    "data": {
        "symbol": "000001.SZ",
        "name": "平安银行",
        "price": 10.65,
        "change": 0.15,
        "change_pct": 1.43,
        "volume": 1000000,
        "amount": 10650000,
        "timestamp": "2024-01-01T15:00:00Z"
    },
    "source": "tushare"
}
```

### 3. 历史行情数据

获取股票历史行情数据。

```http
GET /data/history/{symbol}
Authorization: Bearer <token>
```

**查询参数**:
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `frequency`: 数据频率 (daily, weekly, monthly)
- `adjust`: 复权类型 (none, qfq, hfq)

**响应示例**:
```json
{
    "success": true,
    "data": [
        {
            "date": "2024-01-01",
            "open": 10.50,
            "high": 10.80,
            "low": 10.30,
            "close": 10.65,
            "volume": 1000000,
            "amount": 10650000
        }
    ],
    "total": 250,
    "page": 1,
    "page_size": 100
}
```

### 4. 龙虎榜数据

获取龙虎榜数据。

```http
GET /data/dragon-tiger
Authorization: Bearer <token>
```

**查询参数**:
- `date`: 查询日期 (YYYY-MM-DD)
- `symbol`: 股票代码 (可选)
- `reason`: 上榜原因 (可选)

**响应示例**:
```json
{
    "success": true,
    "data": [
        {
            "date": "2024-01-01",
            "symbol": "000001.SZ",
            "name": "平安银行",
            "close_price": 10.65,
            "change_pct": 1.43,
            "reason": "日涨幅偏离值达7%的证券",
            "buy_amount": 50000000,
            "sell_amount": 30000000,
            "net_amount": 20000000,
            "institutions": [
                {
                    "name": "机构专用",
                    "type": "institution",
                    "buy_amount": 10000000,
                    "sell_amount": 0
                }
            ]
        }
    ]
}
```

### 5. 资金流向数据

获取资金流向数据。

```http
GET /data/fund-flow/{symbol}
Authorization: Bearer <token>
```

**查询参数**:
- `period`: 时间周期 (1d, 3d, 5d, 10d, 20d)
- `date`: 查询日期 (YYYY-MM-DD)

**响应示例**:
```json
{
    "success": true,
    "data": {
        "symbol": "000001.SZ",
        "date": "2024-01-01",
        "period": "1d",
        "main_net_inflow": 10000000,
        "main_net_inflow_pct": 2.5,
        "super_large_net_inflow": 5000000,
        "super_large_net_inflow_pct": 1.2,
        "large_net_inflow": 3000000,
        "large_net_inflow_pct": 0.8,
        "medium_net_inflow": 1000000,
        "medium_net_inflow_pct": 0.3,
        "small_net_inflow": 1000000,
        "small_net_inflow_pct": 0.2
    }
}
```

### 6. 涨停原因数据

获取涨停股票原因分析。

```http
GET /data/limitup-reason
Authorization: Bearer <token>
```

**查询参数**:
- `date`: 查询日期 (YYYY-MM-DD)
- `symbol`: 股票代码 (可选)

**响应示例**:
```json
{
    "success": true,
    "data": [
        {
            "date": "2024-01-01",
            "symbol": "000001.SZ",
            "name": "平安银行",
            "reason": "银行板块利好消息",
            "detail_reason": "央行降准释放流动性，银行股受益",
            "latest_price": 10.65,
            "change_pct": 10.0,
            "turnover_rate": 5.2,
            "volume": 2000000,
            "amount": 21300000
        }
    ]
}
```

### 7. ETF数据

获取ETF基金数据。

```http
GET /data/etf/{symbol}
Authorization: Bearer <token>
```

**查询参数**:
- `data_type`: 数据类型 (realtime, history, holdings)
- `start_date`: 开始日期 (可选)
- `end_date`: 结束日期 (可选)

**响应示例**:
```json
{
    "success": true,
    "data": {
        "symbol": "510050.SH",
        "name": "50ETF",
        "price": 2.850,
        "change": 0.015,
        "change_pct": 0.53,
        "volume": 50000000,
        "amount": 142500000,
        "nav": 2.8523,
        "premium_rate": -0.08,
        "timestamp": "2024-01-01T15:00:00Z"
    }
}
```

## 系统管理接口

### 1. 健康检查

检查系统各组件健康状态。

```http
GET /health
```

**响应示例**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T10:00:00Z",
    "components": {
        "database": {
            "status": "healthy",
            "response_time": 0.05,
            "details": {
                "connection_pool": "10/20",
                "active_connections": 3
            }
        },
        "cache": {
            "status": "healthy",
            "response_time": 0.02,
            "details": {
                "memory_usage": "65%",
                "hit_rate": "85%"
            }
        },
        "data_sources": {
            "tushare": {
                "status": "healthy",
                "response_time": 0.2,
                "quota_remaining": 1000
            },
            "akshare": {
                "status": "degraded",
                "response_time": 1.5,
                "error": "Rate limit approaching"
            }
        }
    }
}
```

### 2. 系统状态

获取系统整体运行状态。

```http
GET /system/status
Authorization: Bearer <token>
```

**响应示例**:
```json
{
    "system_status": "normal",
    "degradation_level": "normal",
    "active_degradations": [],
    "failover_status": {
        "active_resources": {
            "database": "primary_db",
            "data_source": "tushare_api",
            "cache": "redis_primary"
        },
        "total_failovers": 5,
        "success_rate": 100.0
    },
    "performance_metrics": {
        "requests_per_second": 150,
        "average_response_time": 0.25,
        "error_rate": 0.02,
        "cache_hit_rate": 0.85
    }
}
```

### 3. 数据源管理

获取数据源配置和状态。

```http
GET /system/data-sources
Authorization: Bearer <token>
```

**响应示例**:
```json
{
    "data_sources": [
        {
            "id": "tushare",
            "name": "Tushare API",
            "type": "api",
            "status": "active",
            "priority": 1,
            "quota": {
                "total": 10000,
                "used": 2500,
                "remaining": 7500,
                "reset_time": "2024-01-02T00:00:00Z"
            },
            "health": {
                "status": "healthy",
                "last_check": "2024-01-01T10:00:00Z",
                "response_time": 0.2
            }
        }
    ]
}
```

### 4. 缓存管理

管理系统缓存。

```http
GET /system/cache/stats
Authorization: Bearer <token>
```

**响应示例**:
```json
{
    "cache_stats": {
        "total_keys": 15000,
        "memory_usage": "512MB",
        "memory_usage_pct": 65,
        "hit_rate": 0.85,
        "miss_rate": 0.15,
        "evictions": 100,
        "connections": 25
    },
    "cache_types": {
        "realtime_data": {
            "keys": 5000,
            "ttl": 60,
            "hit_rate": 0.90
        },
        "historical_data": {
            "keys": 8000,
            "ttl": 3600,
            "hit_rate": 0.82
        },
        "metadata": {
            "keys": 2000,
            "ttl": 86400,
            "hit_rate": 0.95
        }
    }
}
```

清除缓存：

```http
DELETE /system/cache
Authorization: Bearer <token>
Content-Type: application/json

{
    "cache_type": "realtime_data",
    "pattern": "stock:*"
}
```

### 5. 错误处理和降级

获取错误统计信息。

```http
GET /system/errors
Authorization: Bearer <token>
```

**查询参数**:
- `hours`: 统计时间范围（小时）
- `category`: 错误类别过滤

**响应示例**:
```json
{
    "error_statistics": {
        "total_errors": 150,
        "error_rate": 0.02,
        "categories": {
            "network": 80,
            "data_format": 30,
            "database": 20,
            "external_api": 20
        },
        "recent_errors": [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "category": "network",
                "message": "Connection timeout to data source",
                "count": 5
            }
        ]
    },
    "degradation_status": {
        "current_level": "normal",
        "active_degradations": [],
        "degradation_history": []
    }
}
```

手动触发降级：

```http
POST /system/degradation
Authorization: Bearer <token>
Content-Type: application/json

{
    "level": "moderate",
    "services": ["data_collection", "analysis"],
    "reason": "Manual maintenance"
}
```

### 6. 故障转移管理

获取故障转移状态。

```http
GET /system/failover
Authorization: Bearer <token>
```

**响应示例**:
```json
{
    "failover_status": {
        "total_resources": 10,
        "healthy_resources": 8,
        "failed_resources": 2,
        "active_resources": {
            "database": "backup_db",
            "data_source": "tushare_api",
            "cache": "redis_primary"
        },
        "recent_failovers": [
            {
                "timestamp": "2024-01-01T09:30:00Z",
                "resource_type": "database",
                "from": "primary_db",
                "to": "backup_db",
                "reason": "Connection timeout",
                "success": true
            }
        ]
    }
}
```

手动触发故障转移：

```http
POST /system/failover
Authorization: Bearer <token>
Content-Type: application/json

{
    "resource_type": "database",
    "failed_resource": "primary_db",
    "reason": "Manual failover for maintenance"
}
```

## 批量操作接口

### 1. 批量数据请求

一次请求获取多个股票的数据。

```http
POST /data/batch
Authorization: Bearer <token>
Content-Type: application/json

{
    "requests": [
        {
            "data_type": "stock_realtime",
            "symbol": "000001.SZ"
        },
        {
            "data_type": "stock_realtime",
            "symbol": "000002.SZ"
        }
    ],
    "options": {
        "parallel": true,
        "timeout": 30
    }
}
```

**响应示例**:
```json
{
    "success": true,
    "results": [
        {
            "symbol": "000001.SZ",
            "success": true,
            "data": { /* 股票数据 */ }
        },
        {
            "symbol": "000002.SZ",
            "success": false,
            "error": "Data not available"
        }
    ],
    "summary": {
        "total": 2,
        "successful": 1,
        "failed": 1,
        "execution_time": 1.5
    }
}
```

## WebSocket 实时数据接口

### 连接WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

// 认证
ws.send(JSON.stringify({
    type: 'auth',
    token: 'your_token_here'
}));

// 订阅股票数据
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['stock.000001.SZ', 'stock.000002.SZ']
}));
```

### 消息格式

**订阅消息**:
```json
{
    "type": "subscribe",
    "channels": ["stock.000001.SZ", "fund_flow.000001.SZ"]
}
```

**实时数据推送**:
```json
{
    "type": "data",
    "channel": "stock.000001.SZ",
    "timestamp": "2024-01-01T10:00:00Z",
    "data": {
        "symbol": "000001.SZ",
        "price": 10.65,
        "change": 0.15,
        "volume": 1000000
    }
}
```

## 错误处理

### 错误响应格式

所有错误响应都遵循统一格式：

```json
{
    "success": false,
    "error": {
        "code": "DATA_NOT_FOUND",
        "message": "Requested data not found",
        "details": {
            "symbol": "000001.SZ",
            "date": "2024-01-01"
        },
        "timestamp": "2024-01-01T10:00:00Z",
        "request_id": "req_123456789"
    }
}
```

### 常见错误代码

| 错误代码 | HTTP状态码 | 描述 |
|----------|------------|------|
| INVALID_TOKEN | 401 | 认证令牌无效或过期 |
| INSUFFICIENT_PERMISSIONS | 403 | 权限不足 |
| DATA_NOT_FOUND | 404 | 请求的数据不存在 |
| INVALID_PARAMETERS | 400 | 请求参数无效 |
| RATE_LIMIT_EXCEEDED | 429 | 请求频率超限 |
| DATA_SOURCE_UNAVAILABLE | 503 | 数据源不可用 |
| INTERNAL_ERROR | 500 | 内部服务器错误 |
| TIMEOUT | 504 | 请求超时 |

## 限流和配额

### 请求限制

- **默认限制**: 每分钟1000次请求
- **批量请求**: 每次最多100个项目
- **WebSocket连接**: 每个用户最多10个连接

### 响应头

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-Request-ID: req_123456789
```

## SDK和示例代码

### Python SDK示例

```python
from stock_analysis_client import StockAnalysisClient

# 初始化客户端
client = StockAnalysisClient(
    base_url="http://localhost:8000/api/v1",
    token="your_token_here"
)

# 获取实时数据
realtime_data = client.get_realtime_data("000001.SZ")
print(realtime_data)

# 获取历史数据
history_data = client.get_history_data(
    symbol="000001.SZ",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# 批量请求
batch_result = client.batch_request([
    {"data_type": "stock_realtime", "symbol": "000001.SZ"},
    {"data_type": "stock_realtime", "symbol": "000002.SZ"}
])
```

### JavaScript SDK示例

```javascript
import { StockAnalysisClient } from 'stock-analysis-js-sdk';

const client = new StockAnalysisClient({
    baseURL: 'http://localhost:8000/api/v1',
    token: 'your_token_here'
});

// 获取实时数据
const realtimeData = await client.getRealtimeData('000001.SZ');
console.log(realtimeData);

// WebSocket连接
const ws = client.createWebSocketConnection();
ws.subscribe(['stock.000001.SZ']);
ws.on('data', (data) => {
    console.log('Received:', data);
});
```

### cURL示例

```bash
# 获取认证令牌
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'

# 获取实时数据
curl -X GET http://localhost:8000/api/v1/data/realtime/000001.SZ \
  -H "Authorization: Bearer your_token_here"

# 批量请求
curl -X POST http://localhost:8000/api/v1/data/batch \
  -H "Authorization: Bearer your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"data_type": "stock_realtime", "symbol": "000001.SZ"},
      {"data_type": "stock_realtime", "symbol": "000002.SZ"}
    ]
  }'
```

## 最佳实践

### 1. 性能优化

- **使用缓存**: 对于不经常变化的数据，利用缓存机制
- **批量请求**: 尽量使用批量接口减少请求次数
- **字段选择**: 只请求需要的字段，减少数据传输量
- **分页查询**: 对于大量数据使用分页查询

### 2. 错误处理

- **重试机制**: 对于临时性错误实现指数退避重试
- **降级处理**: 当主要数据源不可用时，使用备用数据源
- **超时设置**: 设置合理的请求超时时间

### 3. 安全考虑

- **令牌管理**: 定期刷新认证令牌
- **HTTPS**: 生产环境使用HTTPS协议
- **参数验证**: 客户端也要进行参数验证

### 4. 监控和日志

- **请求日志**: 记录所有API请求用于调试
- **性能监控**: 监控API响应时间和成功率
- **错误追踪**: 使用request_id追踪错误

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持基础数据获取接口
- 实现认证和权限管理
- 添加WebSocket实时数据推送

### v1.1.0 (2024-02-01)
- 新增批量操作接口
- 增强错误处理机制
- 添加系统管理接口
- 优化性能和缓存策略

## 联系支持

如有问题或建议，请联系：

- **技术支持**: support@stockanalysis.com
- **API文档**: https://docs.stockanalysis.com
- **GitHub**: https://github.com/stockanalysis/api
- **问题反馈**: https://github.com/stockanalysis/api/issues