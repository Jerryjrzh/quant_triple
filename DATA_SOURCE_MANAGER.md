# Data Source Manager Implementation

## Overview

The Data Source Manager is a robust, fault-tolerant system for fetching stock market data from multiple sources with automatic failover capabilities. It implements circuit breaker patterns, rate limiting, and health monitoring to ensure reliable data access.

## Features Implemented

### ✅ Core Features

1. **Multi-Source Support**
   - Tushare Pro API integration
   - AkShare library integration
   - Extensible architecture for additional sources

2. **Fault Tolerance**
   - Circuit breaker pattern for failed sources
   - Automatic failover to backup sources
   - Health monitoring and reliability scoring

3. **Rate Limiting**
   - Configurable requests per minute limits
   - Burst protection with cooldown periods
   - Prevents API quota exhaustion

4. **Data Standardization**
   - Consistent column naming across sources
   - Unified data format for downstream processing
   - Proper date/time handling

### 🔧 Technical Implementation

#### Circuit Breaker Pattern
```python
class CircuitBreaker:
    - failure_threshold: Number of failures before opening circuit
    - recovery_timeout: Time before attempting to close circuit
    - States: closed, open, half_open
```

#### Rate Limiter
```python
class RateLimiter:
    - requests_per_minute: Maximum requests per minute
    - burst_limit: Maximum burst requests
    - cooldown_period: Time between burst requests
```

#### Health Monitoring
```python
class DataSourceHealth:
    - status: healthy, degraded, failed, circuit_open
    - reliability_score: 0.0 to 1.0 based on recent performance
    - success_rate: Percentage of successful requests
    - avg_response_time: Average response time
```

## API Integration

### Endpoints Enhanced

1. **GET /health**
   - Now includes data source health status
   - Shows reliability scores for each source
   - Indicates primary vs fallback source status

2. **GET /api/v1/stocks**
   - Uses real data from configured sources
   - Falls back to mock data if sources unavailable
   - Includes search functionality

3. **GET /api/v1/stocks/{symbol}/data**
   - Fetches historical stock data
   - Supports date range queries
   - Returns standardized OHLCV data

## Configuration

### Environment Variables

```bash
# Tushare Configuration
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_TIMEOUT=30

# AkShare Configuration  
AKSHARE_TIMEOUT=30

# Rate Limiting
DATA_REQUESTS_PER_MINUTE=200
DATA_RETRY_ATTEMPTS=3
DATA_RETRY_DELAY=1
```

### Settings Structure

```python
class DataSourceSettings:
    tushare_token: Optional[str] = None
    tushare_timeout: int = 30
    akshare_timeout: int = 30
    requests_per_minute: int = 200
    retry_attempts: int = 3
    retry_delay: int = 1
```

## Usage Examples

### Basic Usage

```python
from stock_analysis_system.data.data_source_manager import get_data_source_manager
from datetime import date

# Get manager instance
manager = await get_data_source_manager()

# Fetch stock data with automatic failover
data = await manager.get_stock_data(
    symbol="000001.SZ",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31)
)

# Get stock list
stocks = await manager.get_stock_list()

# Check health status
health = await manager.health_check()
```

### Health Monitoring

```python
# Get health summary
summary = manager.get_health_summary()
print(f"Total sources: {summary['total_sources']}")
print(f"Healthy sources: {summary['healthy_sources']}")

# Check individual source health
for source_type, health in health_status.items():
    print(f"{source_type.value}: {health.status.value}")
    print(f"  Reliability: {health.reliability_score:.2f}")
    print(f"  Success rate: {health.success_rate:.2f}")
```

## Testing

### Unit Tests
- Circuit breaker functionality
- Rate limiter behavior
- Data source implementations
- Failover scenarios
- Health monitoring

### Integration Tests
- API endpoint integration
- Real data source connections
- Error handling scenarios

### Running Tests

```bash
# Run all data source manager tests
python -m pytest tests/test_data_source_manager.py -v

# Run specific test class
python -m pytest tests/test_data_source_manager.py::TestDataSourceManager -v

# Test API integration
python test_data_source_api.py
```

## Error Handling

### Graceful Degradation
1. **Primary source fails** → Automatically try secondary source
2. **All sources fail** → Return cached data if available
3. **No data available** → Return empty DataFrame with warning
4. **Rate limit exceeded** → Wait and retry with exponential backoff

### Circuit Breaker States
- **Closed**: Normal operation, requests allowed
- **Open**: Source marked as failed, requests blocked
- **Half-Open**: Testing if source has recovered

### Reliability Scoring
- Based on success rate and recency of failures
- Scores above 0.9: Healthy
- Scores 0.5-0.9: Degraded  
- Scores below 0.5: Failed

## Performance Considerations

### Caching Strategy
- Circuit breaker state cached to avoid repeated failures
- Rate limiter state maintained in memory
- Health metrics updated in real-time

### Async Operations
- All data fetching operations are async
- Non-blocking failover between sources
- Concurrent health checks supported

### Memory Management
- Request history cleaned automatically
- Health metrics use rolling averages
- No persistent state storage required

## Future Enhancements

### Planned Features
1. **Redis-based caching** for fetched data
2. **Webhook notifications** for source failures
3. **Metrics export** to Prometheus
4. **Dynamic source prioritization** based on performance
5. **Data quality validation** before returning results

### Additional Data Sources
- Wind API integration
- Yahoo Finance fallback
- Custom data provider support
- Real-time data streaming

## Monitoring and Alerting

### Key Metrics to Monitor
- Source availability percentage
- Average response times
- Circuit breaker state changes
- Rate limit violations
- Data quality issues

### Recommended Alerts
- Source down for > 5 minutes
- Reliability score drops below 0.5
- All sources failing simultaneously
- Unusual spike in response times

## Conclusion

The Data Source Manager provides a robust foundation for reliable stock data access with automatic failover, rate limiting, and comprehensive health monitoring. It successfully implements the requirements from task 2.1 and provides a solid base for the remaining system components.

### Task 2.1 Completion Status: ✅ COMPLETED

**Implemented Features:**
- ✅ DataSourceManager class with circuit breaker pattern
- ✅ Failover logic across multiple data sources (Tushare, AkShare)
- ✅ Rate limiting and request throttling mechanisms  
- ✅ Data source health monitoring and reliability scoring
- ✅ Comprehensive test coverage
- ✅ API integration with health endpoints
- ✅ Error handling and graceful degradation