# Stock Analysis System API 端点文档

## 概述

股票分析系统提供了一套完整的 RESTful API，支持股票数据查询、春节分析、股票筛选和用户警报等功能。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1
- **认证方式**: JWT Bearer Token
- **限流**: 每分钟100次请求（可配置）

## 端点列表

### 1. 系统端点

#### GET `/`
根端点，返回系统基本信息。

**响应示例**:
```json
{
  "message": "Welcome to Stock Analysis System",
  "version": "0.1.0",
  "environment": "development",
  "status": "running"
}
```

#### GET `/health`
健康检查端点，检查系统和数据库状态。

**响应示例**:
```json
{
  "status": "ok",
  "database": "healthy",
  "version": "0.1.0",
  "environment": "development"
}
```

#### GET `/api/v1/info`
API信息端点，返回API版本和功能列表。

**响应示例**:
```json
{
  "api_version": "v1",
  "app_name": "Stock Analysis System",
  "app_version": "0.1.0",
  "environment": "development",
  "features": [
    "Spring Festival Analysis",
    "Institutional Fund Tracking",
    "Risk Management",
    "Stock Screening",
    "Real-time Alerts"
  ]
}
```

### 2. 股票数据端点

#### GET `/api/v1/stocks`
获取股票列表，支持搜索和分页。

**查询参数**:
- `limit` (int, 可选): 返回结果数量限制，默认50
- `offset` (int, 可选): 分页偏移量，默认0
- `search` (string, 可选): 搜索关键词

**响应示例**:
```json
{
  "stocks": [
    {
      "symbol": "000001.SZ",
      "name": "平安银行",
      "market": "深圳",
      "sector": "金融",
      "last_price": 12.5,
      "change_percent": 2.1
    }
  ],
  "total": 3,
  "limit": 50,
  "offset": 0
}
```

#### GET `/api/v1/stocks/{symbol}`
获取特定股票的详细信息。

**路径参数**:
- `symbol` (string): 股票代码，如 "000001.SZ"

**响应示例**:
```json
{
  "symbol": "000001.SZ",
  "name": "平安银行",
  "market": "深圳",
  "sector": "金融",
  "industry": "银行",
  "current_price": 12.5,
  "change": 0.26,
  "change_percent": 2.1,
  "volume": 125000000,
  "market_cap": 241500000000,
  "pe_ratio": 5.8,
  "pb_ratio": 0.65,
  "dividend_yield": 3.2,
  "last_updated": "2025-08-01T18:24:38.707307"
}
```

### 3. 春节分析端点

#### GET `/api/v1/stocks/{symbol}/spring-festival`
获取特定股票的春节分析数据。

**路径参数**:
- `symbol` (string): 股票代码

**查询参数**:
- `years` (int, 可选): 分析年数，默认5年

**响应示例**:
```json
{
  "symbol": "000001.SZ",
  "analysis_period": "2020-2025",
  "spring_festival_pattern": {
    "average_return_before": 2.3,
    "average_return_after": -1.1,
    "volatility_increase": 15.2,
    "pattern_strength": 0.75,
    "confidence_score": 0.82
  },
  "yearly_data": [
    {
      "year": 2024,
      "spring_festival_date": "2024-02-10",
      "return_before": 3.1,
      "return_after": -0.8,
      "volatility": 18.5
    }
  ],
  "recommendations": [
    "历史数据显示该股票在春节前通常有正收益",
    "春节后收益相对较弱，建议谨慎持有",
    "春节期间波动性增加，注意风险控制"
  ]
}
```

### 4. 股票筛选端点

#### GET `/api/v1/screening`
根据各种条件筛选股票。

**查询参数**:
- `min_market_cap` (float, 可选): 最小市值
- `max_pe_ratio` (float, 可选): 最大市盈率
- `min_dividend_yield` (float, 可选): 最小股息率
- `sector` (string, 可选): 行业筛选
- `limit` (int, 可选): 返回结果数量限制，默认50

**响应示例**:
```json
{
  "screened_stocks": [
    {
      "symbol": "000001.SZ",
      "name": "平安银行",
      "market_cap": 241500000000,
      "pe_ratio": 5.8,
      "dividend_yield": 3.2,
      "sector": "金融",
      "score": 85.2
    }
  ],
  "total_matches": 2,
  "criteria": {
    "min_market_cap": null,
    "max_pe_ratio": 10.0,
    "min_dividend_yield": null,
    "sector": "金融"
  }
}
```

### 5. 用户警报端点（需要认证）

#### GET `/api/v1/alerts`
获取用户的警报列表。

**认证**: 需要 JWT Bearer Token

**查询参数**:
- `active_only` (bool, 可选): 只返回活跃警报，默认true
- `limit` (int, 可选): 返回结果数量限制，默认20

**响应示例**:
```json
{
  "alerts": [
    {
      "id": 1,
      "symbol": "000001.SZ",
      "type": "price_target",
      "condition": "price >= 13.00",
      "status": "active",
      "created_at": "2024-01-15T10:30:00",
      "triggered_at": null
    }
  ],
  "total": 1,
  "user": "username"
}
```

## 错误处理

API 使用标准的 HTTP 状态码：

- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误
- `401 Unauthorized`: 未认证或认证失败
- `403 Forbidden`: 权限不足
- `404 Not Found`: 资源不存在
- `429 Too Many Requests`: 请求频率超限
- `500 Internal Server Error`: 服务器内部错误

错误响应格式：
```json
{
  "detail": "错误描述信息"
}
```

## 认证

对于需要认证的端点，需要在请求头中包含 JWT token：

```
Authorization: Bearer <your-jwt-token>
```

## 限流

API 实现了基于 IP 的限流机制，默认每分钟100次请求。超出限制时会返回 429 状态码。

## 开发状态

当前实现的是 MVP 版本，包含基础的 API 端点和模拟数据。后续版本将集成真实的数据源和完整的分析引擎。

## 测试

使用提供的 `test_api.py` 脚本可以测试所有端点：

```bash
python test_api.py
```