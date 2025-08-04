# API Reference Documentation

## 📋 Overview

The Stock Analysis System provides a comprehensive RESTful API built with FastAPI. This document details all available endpoints, request/response formats, authentication requirements, and usage examples.

**Base URL**: `http://localhost:8000`  
**API Version**: v1  
**Documentation**: `http://localhost:8000/docs` (Swagger UI)  
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

## 🔐 Authentication

### JWT Token Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```http
POST /auth/login
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

## 🏥 Health & System Endpoints

### Health Check

Check system health and component status.

```http
GET /health
```

**Response:**
```json
{
    "status": "ok",
    "timestamp": "2025-01-01T12:00:00Z",
    "version": "0.1.0",
    "environment": "development",
    "database": "healthy",
    "redis": "healthy",
    "data_sources": {
        "tushare": {
            "status": "healthy",
            "reliability_score": 0.95,
            "last_check": "2025-01-01T11:59:00Z"
        },
        "akshare": {
            "status": "healthy", 
            "reliability_score": 0.92,
            "last_check": "2025-01-01T11:59:00Z"
        }
    }
}
```

### System Information

Get detailed system information and configuration.

```http
GET /api/v1/info
```

**Response:**
```json
{
    "system": {
        "name": "Stock Analysis System",
        "version": "0.1.0",
        "environment": "development",
        "uptime": "2 days, 3 hours, 45 minutes"
    },
    "features": {
        "spring_festival_analysis": true,
        "parallel_processing": true,
        "data_quality_validation": true,
        "visualization": true
    },
    "limits": {
        "max_symbols_per_request": 100,
        "max_years_analysis": 10,
        "rate_limit_per_minute": 200
    }
}
```

## 📊 Stock Data Endpoints

### Search Stocks

Search for stocks by symbol or name with auto-complete functionality.

```http
GET /api/v1/stocks?q={query}&limit={limit}
```

**Parameters:**
- `q` (string, required): Search query (stock code or company name)
- `limit` (integer, optional): Maximum results to return (default: 10, max: 50)

**Example:**
```http
GET /api/v1/stocks?q=平安&limit=5
```

**Response:**
```json
{
    "stocks": [
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "market": "深圳",
            "sector": "银行",
            "industry": "银行",
            "list_date": "1991-04-03",
            "is_active": true
        },
        {
            "symbol": "601318.SH", 
            "name": "中国平安",
            "market": "上海",
            "sector": "保险",
            "industry": "保险",
            "list_date": "2007-03-01",
            "is_active": true
        }
    ],
    "total": 2,
    "query": "平安",
    "execution_time": 0.045
}
```

### Get Stock Details

Get detailed information about a specific stock.

```http
GET /api/v1/stocks/{symbol}
```

**Parameters:**
- `symbol` (string, required): Stock symbol (e.g., "000001.SZ")

**Example:**
```http
GET /api/v1/stocks/000001.SZ
```

**Response:**
```json
{
    "symbol": "000001.SZ",
    "name": "平安银行",
    "market": "深圳",
    "sector": "银行",
    "industry": "银行",
    "list_date": "1991-04-03",
    "total_shares": 19405918198,
    "float_shares": 19405918198,
    "market_cap": 387318363960,
    "pe_ratio": 4.85,
    "pb_ratio": 0.67,
    "dividend_yield": 0.0312,
    "last_update": "2025-01-01T15:00:00Z"
}
```

### Get Stock Historical Data

Retrieve historical OHLCV data for a stock.

```http
GET /api/v1/stocks/{symbol}/data?start_date={start}&end_date={end}&timeframe={timeframe}
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `timeframe` (string, optional): Data frequency ("daily", "weekly", "monthly", default: "daily")

**Example:**
```http
GET /api/v1/stocks/000001.SZ/data?start_date=2024-01-01&end_date=2024-12-31
```

**Response:**
```json
{
    "symbol": "000001.SZ",
    "timeframe": "daily",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "data": [
        {
            "trade_date": "2024-01-02",
            "open_price": 12.50,
            "high_price": 12.80,
            "low_price": 12.45,
            "close_price": 12.75,
            "volume": 45678900,
            "amount": 582345678.90,
            "adj_factor": 1.0
        }
    ],
    "total_records": 244,
    "data_quality_score": 0.98
}
```

## 🎯 Spring Festival Analysis Endpoints

### Perform Spring Festival Analysis

Analyze stock performance patterns relative to Spring Festival dates.

```http
POST /api/v1/analysis/spring-festival
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbol": "000001.SZ",
    "years": [2020, 2021, 2022, 2023, 2024],
    "window_days": 60,
    "include_pattern_analysis": true,
    "include_trading_signals": true
}
```

**Parameters:**
- `symbol` (string, required): Stock symbol to analyze
- `years` (array, required): Years to include in analysis
- `window_days` (integer, optional): Days before/after Spring Festival (default: 60)
- `include_pattern_analysis` (boolean, optional): Include ML pattern recognition (default: true)
- `include_trading_signals` (boolean, optional): Include trading signals (default: true)

**Response:**
```json
{
    "symbol": "000001.SZ",
    "analysis_date": "2025-01-01T12:00:00Z",
    "years_analyzed": [2020, 2021, 2022, 2023, 2024],
    "window_days": 60,
    "aligned_data": {
        "baseline_price": 12.50,
        "data_points": [
            {
                "relative_day": -60,
                "year": 2024,
                "date": "2023-12-21",
                "price": 12.30,
                "normalized_price": -1.60,
                "spring_festival_date": "2024-02-10"
            }
        ]
    },
    "seasonal_pattern": {
        "pattern_strength": 0.75,
        "confidence_level": 0.82,
        "consistency_score": 0.68,
        "average_return_before": 2.34,
        "average_return_after": -1.12,
        "volatility_before": 0.025,
        "volatility_after": 0.032,
        "peak_day": -15,
        "trough_day": 8
    },
    "trading_signals": {
        "current_position": "pre_festival",
        "days_to_spring_festival": 25,
        "signal": "buy",
        "signal_strength": 0.72,
        "recommended_action": "建议买入",
        "reason": "历史数据显示春节前15-30天通常有上涨趋势",
        "risk_warning": "注意市场整体波动风险"
    },
    "ml_analysis": {
        "cluster_id": 2,
        "cluster_confidence": 0.85,
        "anomaly_score": 0.12,
        "feature_importance": {
            "volatility": 0.35,
            "trend_strength": 0.28,
            "volume_pattern": 0.22,
            "price_momentum": 0.15
        }
    },
    "execution_time": 2.34
}
```

### Batch Spring Festival Analysis

Analyze multiple stocks simultaneously.

```http
POST /api/v1/analysis/spring-festival/batch
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbols": ["000001.SZ", "600000.SH", "000858.SZ"],
    "years": [2022, 2023, 2024],
    "window_days": 60,
    "parallel_processing": true
}
```

**Response:**
```json
{
    "batch_id": "batch_20250101_120000",
    "total_symbols": 3,
    "successful_analyses": 3,
    "failed_analyses": 0,
    "execution_time": 5.67,
    "results": [
        {
            "symbol": "000001.SZ",
            "status": "success",
            "analysis": { /* Same structure as single analysis */ }
        },
        {
            "symbol": "600000.SH", 
            "status": "success",
            "analysis": { /* Analysis data */ }
        },
        {
            "symbol": "000858.SZ",
            "status": "success", 
            "analysis": { /* Analysis data */ }
        }
    ],
    "summary": {
        "average_pattern_strength": 0.68,
        "strongest_pattern": "000001.SZ",
        "weakest_pattern": "000858.SZ",
        "bullish_signals": 2,
        "bearish_signals": 1
    }
}
```

### Get Current Spring Festival Position

Get current position relative to Spring Festival cycle.

```http
GET /api/v1/analysis/spring-festival/position?symbol={symbol}
```

**Example:**
```http
GET /api/v1/analysis/spring-festival/position?symbol=000001.SZ
```

**Response:**
```json
{
    "symbol": "000001.SZ",
    "current_date": "2025-01-01",
    "next_spring_festival": "2025-01-29",
    "days_to_spring_festival": 28,
    "position": "approaching",
    "in_analysis_window": true,
    "historical_performance": {
        "same_period_average_return": 1.85,
        "same_period_volatility": 0.028,
        "probability_of_positive_return": 0.72
    }
}
```

## 📈 Visualization Endpoints

### Generate Spring Festival Chart

Create interactive charts for Spring Festival analysis.

```http
POST /api/v1/visualization/spring-festival-chart
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbol": "000001.SZ",
    "years": [2020, 2021, 2022, 2023, 2024],
    "chart_type": "overlay",
    "title": "平安银行春节对齐分析",
    "show_pattern_info": true,
    "show_trading_signals": true,
    "export_format": "html"
}
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `years` (array, required): Years to display
- `chart_type` (string, optional): Chart type ("overlay", "comparison", "cluster", default: "overlay")
- `title` (string, optional): Chart title
- `show_pattern_info` (boolean, optional): Show pattern analysis info (default: true)
- `show_trading_signals` (boolean, optional): Show trading signals (default: false)
- `export_format` (string, optional): Export format ("html", "json", default: "html")

**Response:**
```json
{
    "chart_id": "chart_20250101_120000",
    "symbol": "000001.SZ",
    "chart_type": "overlay",
    "format": "html",
    "chart_data": "<div id='plotly-div'>...</div>",
    "metadata": {
        "width": 1200,
        "height": 800,
        "interactive": true,
        "export_enabled": true
    },
    "generation_time": 0.156
}
```

### Multi-Stock Comparison Chart

Create comparison charts for multiple stocks.

```http
POST /api/v1/visualization/multi-stock-chart
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbols": ["000001.SZ", "600000.SH", "000858.SZ"],
    "years": [2022, 2023, 2024],
    "chart_type": "comparison",
    "title": "多股票春节模式对比",
    "include_clustering": true
}
```

**Response:**
```json
{
    "chart_id": "multi_chart_20250101_120000",
    "symbols": ["000001.SZ", "600000.SH", "000858.SZ"],
    "chart_type": "comparison",
    "chart_data": "<div id='plotly-div'>...</div>",
    "clustering_results": {
        "n_clusters": 2,
        "cluster_assignments": {
            "000001.SZ": 0,
            "600000.SH": 0, 
            "000858.SZ": 1
        },
        "cluster_centers": [
            {"pattern_strength": 0.75, "volatility": 0.025},
            {"pattern_strength": 0.45, "volatility": 0.035}
        ]
    },
    "generation_time": 0.234
}
```

### Export Chart

Export charts in various formats.

```http
POST /api/v1/visualization/export
Content-Type: application/json
```

**Request Body:**
```json
{
    "chart_id": "chart_20250101_120000",
    "format": "png",
    "width": 1600,
    "height": 1000,
    "scale": 2,
    "filename": "spring_festival_analysis.png"
}
```

**Response:**
```json
{
    "export_id": "export_20250101_120000",
    "format": "png",
    "filename": "spring_festival_analysis.png",
    "file_size": 245760,
    "download_url": "/api/v1/visualization/download/export_20250101_120000",
    "expires_at": "2025-01-01T18:00:00Z"
}
```

### Download Exported Chart

Download exported chart files.

```http
GET /api/v1/visualization/download/{export_id}
```

**Response:** Binary file content with appropriate headers.

## 📊 Data Quality Endpoints

### Validate Data Quality

Check data quality for a specific stock.

```http
POST /api/v1/data-quality/validate
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbol": "000001.SZ",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "include_ml_analysis": true
}
```

**Response:**
```json
{
    "symbol": "000001.SZ",
    "validation_date": "2025-01-01T12:00:00Z",
    "data_period": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "total_records": 244
    },
    "quality_scores": {
        "overall_score": 0.94,
        "completeness_score": 0.98,
        "consistency_score": 0.92,
        "timeliness_score": 0.95,
        "accuracy_score": 0.91
    },
    "issues_found": [
        {
            "type": "missing_data",
            "severity": "medium",
            "count": 3,
            "description": "3个交易日数据缺失",
            "affected_dates": ["2024-05-15", "2024-08-22", "2024-11-08"]
        },
        {
            "type": "outlier_data",
            "severity": "low", 
            "count": 2,
            "description": "检测到2个异常数据点",
            "details": "价格波动超过3个标准差"
        }
    ],
    "recommendations": [
        "建议补充缺失的交易日数据",
        "检查异常数据点的准确性",
        "考虑使用数据清洗功能"
    ],
    "ml_analysis": {
        "anomaly_detection": {
            "total_anomalies": 5,
            "anomaly_rate": 0.02,
            "confidence_threshold": 0.1
        },
        "pattern_consistency": 0.87
    }
}
```

### Get Data Quality Report

Get comprehensive data quality report for multiple stocks.

```http
GET /api/v1/data-quality/report?symbols={symbols}&days={days}
```

**Example:**
```http
GET /api/v1/data-quality/report?symbols=000001.SZ,600000.SH&days=30
```

**Response:**
```json
{
    "report_id": "quality_report_20250101",
    "generation_date": "2025-01-01T12:00:00Z",
    "symbols_analyzed": ["000001.SZ", "600000.SH"],
    "analysis_period": "30 days",
    "summary": {
        "average_quality_score": 0.92,
        "total_issues": 8,
        "critical_issues": 0,
        "high_issues": 2,
        "medium_issues": 4,
        "low_issues": 2
    },
    "stock_reports": [
        {
            "symbol": "000001.SZ",
            "quality_score": 0.94,
            "issues": 3,
            "status": "good"
        },
        {
            "symbol": "600000.SH",
            "quality_score": 0.90,
            "issues": 5,
            "status": "acceptable"
        }
    ],
    "recommendations": [
        "定期检查数据源连接状态",
        "建立数据质量监控告警",
        "考虑增加数据验证规则"
    ]
}
```

## ⚙️ ETL & Background Tasks

### Trigger Data Update

Manually trigger data update for specific stocks.

```http
POST /api/v1/etl/update
Content-Type: application/json
```

**Request Body:**
```json
{
    "symbols": ["000001.SZ", "600000.SH"],
    "start_date": "2024-12-01",
    "end_date": "2024-12-31",
    "priority": "high",
    "validate_quality": true
}
```

**Response:**
```json
{
    "job_id": "etl_job_20250101_120000",
    "status": "queued",
    "symbols": ["000001.SZ", "600000.SH"],
    "estimated_completion": "2025-01-01T12:05:00Z",
    "priority": "high",
    "queue_position": 2
}
```

### Get Job Status

Check the status of background jobs.

```http
GET /api/v1/etl/jobs/{job_id}
```

**Response:**
```json
{
    "job_id": "etl_job_20250101_120000",
    "status": "completed",
    "created_at": "2025-01-01T12:00:00Z",
    "started_at": "2025-01-01T12:01:00Z",
    "completed_at": "2025-01-01T12:04:30Z",
    "progress": {
        "total_symbols": 2,
        "completed_symbols": 2,
        "failed_symbols": 0,
        "percentage": 100
    },
    "results": {
        "records_processed": 488,
        "records_inserted": 485,
        "records_updated": 3,
        "quality_score": 0.96
    },
    "errors": []
}
```

### List Active Jobs

Get list of active background jobs.

```http
GET /api/v1/etl/jobs?status={status}&limit={limit}
```

**Response:**
```json
{
    "jobs": [
        {
            "job_id": "etl_job_20250101_120000",
            "status": "running",
            "created_at": "2025-01-01T12:00:00Z",
            "progress": 75,
            "symbols_count": 5
        }
    ],
    "total": 1,
    "active_workers": 4,
    "queue_length": 3
}
```

## 🚨 Error Handling

### Standard Error Response Format

All API errors follow a consistent format:

```json
{
    "error": {
        "code": "INVALID_SYMBOL",
        "message": "股票代码格式不正确",
        "details": "Symbol must be in format XXXXXX.XX (e.g., 000001.SZ)",
        "timestamp": "2025-01-01T12:00:00Z",
        "request_id": "req_20250101_120000"
    }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_REQUEST` | 请求参数无效 |
| 400 | `INVALID_SYMBOL` | 股票代码格式错误 |
| 400 | `INVALID_DATE_RANGE` | 日期范围无效 |
| 401 | `UNAUTHORIZED` | 未授权访问 |
| 403 | `FORBIDDEN` | 权限不足 |
| 404 | `SYMBOL_NOT_FOUND` | 股票代码不存在 |
| 404 | `DATA_NOT_FOUND` | 数据不存在 |
| 429 | `RATE_LIMIT_EXCEEDED` | 请求频率超限 |
| 500 | `INTERNAL_ERROR` | 服务器内部错误 |
| 503 | `SERVICE_UNAVAILABLE` | 服务暂时不可用 |

## 📊 Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 195
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Rate Limits by Endpoint

| Endpoint Category | Requests per Minute | Burst Limit |
|------------------|-------------------|-------------|
| Health & Info | 1000 | 100 |
| Stock Search | 200 | 50 |
| Stock Data | 100 | 20 |
| Analysis | 50 | 10 |
| Visualization | 30 | 5 |
| ETL Operations | 10 | 2 |

## 📝 Request/Response Examples

### Complete Analysis Workflow

```bash
# 1. Search for stocks
curl -X GET "http://localhost:8000/api/v1/stocks?q=平安银行" \
  -H "Authorization: Bearer $TOKEN"

# 2. Perform Spring Festival analysis
curl -X POST "http://localhost:8000/api/v1/analysis/spring-festival" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "000001.SZ",
    "years": [2022, 2023, 2024],
    "window_days": 60
  }'

# 3. Generate visualization
curl -X POST "http://localhost:8000/api/v1/visualization/spring-festival-chart" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "000001.SZ", 
    "years": [2022, 2023, 2024],
    "chart_type": "overlay"
  }'

# 4. Export chart
curl -X POST "http://localhost:8000/api/v1/visualization/export" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chart_id": "chart_20250101_120000",
    "format": "png"
  }'
```

## 🔧 SDK & Client Libraries

### Python Client Example

```python
import requests
from typing import List, Dict, Any

class StockAnalysisClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        response = requests.get(
            f"{self.base_url}/api/v1/stocks",
            params={"q": query, "limit": limit},
            headers=self.headers
        )
        return response.json()["stocks"]
    
    def analyze_spring_festival(self, symbol: str, years: List[int]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/analysis/spring-festival",
            json={"symbol": symbol, "years": years},
            headers=self.headers
        )
        return response.json()

# Usage
client = StockAnalysisClient("http://localhost:8000", "your-token")
stocks = client.search_stocks("平安银行")
analysis = client.analyze_spring_festival("000001.SZ", [2022, 2023, 2024])
```

### JavaScript Client Example

```javascript
class StockAnalysisAPI {
    constructor(baseURL, token) {
        this.baseURL = baseURL;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async searchStocks(query, limit = 10) {
        const response = await fetch(
            `${this.baseURL}/api/v1/stocks?q=${query}&limit=${limit}`,
            { headers: this.headers }
        );
        const data = await response.json();
        return data.stocks;
    }
    
    async analyzeSpringFestival(symbol, years) {
        const response = await fetch(
            `${this.baseURL}/api/v1/analysis/spring-festival`,
            {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({ symbol, years })
            }
        );
        return await response.json();
    }
}

// Usage
const api = new StockAnalysisAPI('http://localhost:8000', 'your-token');
const stocks = await api.searchStocks('平安银行');
const analysis = await api.analyzeSpringFestival('000001.SZ', [2022, 2023, 2024]);
```

## 📋 Summary

The Stock Analysis System API provides comprehensive functionality for:

- ✅ **Stock Data Access**: Search, details, historical data
- ✅ **Spring Festival Analysis**: Core temporal analysis with ML
- ✅ **Visualization**: Interactive charts and export capabilities  
- ✅ **Data Quality**: Validation and monitoring
- ✅ **Background Processing**: ETL operations and job management
- ✅ **Authentication**: JWT-based security
- ✅ **Rate Limiting**: Fair usage policies
- ✅ **Error Handling**: Consistent error responses

The API is designed for both programmatic access and interactive use, with comprehensive documentation and client library examples.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**API Version**: v1  
**Maintained By**: Development Team