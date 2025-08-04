# Stock Analysis System

[![Docker Setup](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](DOCKER_SETUP_SUMMARY.md)
[![API Status](https://img.shields.io/badge/API-Working-green?logo=fastapi)](http://localhost:8000/docs)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-blue?logo=postgresql)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](requirements.txt)

A comprehensive stock analysis system that leverages calendar-based temporal analysis as its core foundation, integrating opportunity identification, risk assessment, stock screening strategies, and review marking capabilities.

> **🚀 快速开始**: 使用 `sudo docker-compose up -d postgres redis && python start_server.py` 一键启动系统

## 📊 Current System Status

**Phase 1 (Foundation & Core Infrastructure)**: 🟢 **75% Complete**

### ✅ Fully Implemented Components
- **Spring Festival Analysis Engine**: Complete temporal alignment and pattern recognition
- **Multi-Source Data Manager**: Tushare, AkShare, and local TDX data integration with failover
- **Data Quality Engine**: ML-based validation with Isolation Forest anomaly detection
- **ETL Pipeline**: Celery-powered background processing with comprehensive error handling
- **Database Infrastructure**: PostgreSQL with Alembic migrations and connection pooling
- **API Foundation**: FastAPI with JWT authentication, rate limiting, and async support
- **Configuration System**: Environment-based configuration with validation

### 🔄 In Progress
- **Parallel Processing**: Dask integration for large-scale analysis
- **Web Visualization**: Interactive Spring Festival charts and dashboards
- **Advanced Analytics**: Risk management and institutional tracking engines

## 🌟 Key Features

### ✅ Spring Festival Alignment Analysis (Implemented)
- **Temporal Anchor Analysis**: Uses Chinese New Year as temporal anchor points to normalize and compare historical stock performance patterns
- **Seasonal Pattern Recognition**: ML-powered pattern identification with confidence scoring and consistency analysis
- **Multi-Year Data Alignment**: Sophisticated algorithms for aligning stock data across multiple years relative to Spring Festival dates
- **Trading Signal Generation**: Automated signal generation based on historical seasonal patterns

### ✅ Advanced Data Management (Implemented)
- **Multi-Source Data Integration**: Seamless integration with Tushare, AkShare, and local TDX data sources
- **Intelligent Failover**: Circuit breaker pattern with automatic source switching for maximum reliability
- **ML-Based Data Quality**: Isolation Forest anomaly detection with comprehensive quality scoring
- **ETL Pipeline**: Celery-powered background processing with error handling and retry mechanisms

### ✅ Production-Ready Infrastructure (Implemented)
- **FastAPI Backend**: High-performance async API with JWT authentication and rate limiting
- **PostgreSQL Database**: Robust data storage with Alembic migrations and connection pooling
- **Redis Caching**: Performance optimization with intelligent caching strategies
- **Docker Support**: Complete containerization for easy deployment

### 🚧 Planned Features (In Development)
- **Interactive Visualizations**: WebGL-accelerated charts with Spring Festival overlays
- **Institutional Fund Tracking**: Dragon-tiger list analysis and attention scoring
- **Advanced Risk Management**: Dynamic VaR calculation and seasonal risk assessment
- **Multi-Dimensional Screening**: Technical, seasonal, and institutional factor screening

## 🏗️ Architecture

The system follows a four-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                 Presentation Layer                          │
│  React UI • Interactive Charts • Real-time Dashboards      │
├─────────────────────────────────────────────────────────────┤
│                 Application Layer                           │
│  FastAPI • Stock Pool Manager • Alert Engine • API Gateway │
├─────────────────────────────────────────────────────────────┤
│                 Analysis Layer                              │
│  Spring Festival Engine • Risk Engine • ML Models • Plugins│
├─────────────────────────────────────────────────────────────┤
│                 Data Layer                                  │
│  PostgreSQL • Redis Cache • ETL Pipeline • Data Sources    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 系统要求

- Python 3.9+
- Docker & Docker Compose (推荐)
- 或者 PostgreSQL 12+ 和 Redis 6+ (本地安装)

### 安装方式

#### 方式一：Docker 安装 (推荐)

这是最简单、最可靠的安装方式，自动处理所有依赖。

1. **克隆项目**
   ```bash
   git clone https://github.com/your-org/stock-analysis-system.git
   cd stock-analysis-system
   ```

2. **启动 Docker 服务**
   ```bash
   # 启动 PostgreSQL 和 Redis 容器
   sudo docker-compose up -d postgres redis
   
   # 验证服务状态 (应该显示 healthy)
   sudo docker-compose ps
   ```

3. **设置 Python 环境**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # 安装依赖
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   ```bash
   # 复制配置文件
   cp .env.example .env
   
   # 确保数据库密码正确 (默认: password)
   # 可以直接使用默认配置，或根据需要修改
   ```

5. **初始化数据库**
   ```bash
   # 运行数据库迁移
   make db-upgrade
   
   # 或者使用完整命令
   DB_HOST=localhost DB_PORT=5432 DB_NAME=stock_analysis DB_USER=postgres DB_PASSWORD=password alembic upgrade head
   ```

6. **启动应用程序**
   ```bash
   # 方式1: 使用智能启动脚本 (推荐)
   python start_server.py
   
   # 方式2: 使用 Make 命令
   make run-dev
   
   # 方式3: 使用 Docker 管理命令
   make docker-up    # 启动 Docker 服务
   make start-server # 启动 API 服务器
   ```

7. **验证安装**
   ```bash
   # 测试 API 端点
   python test_api.py
   
   # 或使用 Make 命令
   make test-api
   
   # 或手动测试
   curl http://localhost:8000/health
   ```

**🎉 成功！** 系统现在运行在 http://localhost:8000

### 可用的 Make 命令

```bash
# Docker 服务管理
make docker-up       # 启动 PostgreSQL 和 Redis
make docker-down     # 停止所有 Docker 服务
make docker-status   # 查看 Docker 服务状态

# 应用程序管理
make start-server    # 启动 API 服务器 (智能脚本)
make run-dev         # 启动开发服务器
make test-api        # 测试 API 端点

# 数据库管理
make db-upgrade      # 运行数据库迁移
make db-downgrade    # 回滚数据库迁移

# 开发工具
make test           # 运行测试套件
make lint           # 代码质量检查
make format         # 代码格式化
```

### 故障排除

#### Docker 权限问题
如果遇到权限错误：
```bash
# 方法1: 使用 sudo (临时解决)
sudo docker-compose up -d postgres redis

# 方法2: 添加用户到 docker 组 (永久解决)
sudo usermod -aG docker $USER
# 然后注销重新登录，或运行: newgrp docker
```

#### 数据库连接问题
```bash
# 检查 Docker 容器状态
sudo docker-compose ps

# 查看容器日志
sudo docker-compose logs postgres

# 确保 .env 文件中的密码匹配 docker-compose.yml
# 默认密码: password
```

#### 端口冲突
如果端口 5432 或 6379 被占用：
```bash
# 检查端口使用情况
sudo netstat -tlnp | grep :5432
sudo netstat -tlnp | grep :6379

# 修改 docker-compose.yml 中的端口映射
# 例如: "15432:5432" 和 "16379:6379"
```

#### API 服务器问题
```bash
# 使用智能启动脚本获得详细错误信息
python start_server.py

# 检查所有依赖是否安装
pip install -r requirements.txt

# 验证环境变量
python -c "from config.settings import get_settings; print(get_settings().database.url)"
```

### 系统架构验证

安装成功后，你应该看到：

1. **Docker 容器运行**:
   ```bash
   $ sudo docker-compose ps
   NAME                      STATUS
   quant_trigle-postgres-1   Up (healthy)
   quant_trigle-redis-1      Up (healthy)
   ```

2. **数据库表创建**:
   ```bash
   $ sudo docker-compose exec postgres psql -U postgres -d stock_analysis -c "\dt"
   # 应该显示 12 个表
   ```

3. **API 端点响应**:
   ```bash
   python start_server.py &
   $ curl http://localhost:8000/health
   {"status":"ok","database":"healthy","version":"0.1.0","environment":"development"}
   ```

### 快速验证系统功能

安装完成后，可以快速验证核心功能：

```bash
# 1. 检查系统状态
curl http://localhost:8000/health

# 2. 查看系统信息
curl http://localhost:8000/api/v1/info

# 3. 访问 API 文档
# 浏览器打开: http://localhost:8000/docs

# 4. 运行完整测试
python test_api.py
```

### 下一步

- 🔍 查看 [API 文档](http://localhost:8000/docs) (Swagger UI)
- 📖 阅读 [使用示例](#-usage-examples)
- 🛠️ 探索 [开发指南](#-development)
- 📋 查看 [Docker 设置详细记录](DOCKER_SETUP_SUMMARY.md)

#### 方式二：SQLite 安装 (测试/开发)

适用于快速测试或不想使用 Docker 的情况。

```bash
# 克隆项目
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system

# 设置 Python 环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 使用 SQLite 数据库测试
python test_migration.py

# 或直接运行迁移
DATABASE_URL="sqlite:///./stock_analysis.db" alembic upgrade head

# 启动应用 (功能受限，无 Redis 缓存)
DATABASE_URL="sqlite:///./stock_analysis.db" uvicorn stock_analysis_system.api.main:app --reload
```

#### 方式三：本地 PostgreSQL 安装

适用于生产环境或需要完全控制数据库的情况。

```bash
# 安装 PostgreSQL (Ubuntu/Debian)
sudo apt update && sudo apt install postgresql postgresql-contrib redis-server

# 创建数据库
sudo -u postgres psql -c "CREATE DATABASE stock_analysis;"
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE stock_analysis TO postgres;"

# 启动服务
sudo systemctl start postgresql redis-server
sudo systemctl enable postgresql redis-server

# 设置应用
cp .env.example .env
# 编辑 .env 文件配置数据库连接
pip install -r requirements.txt
alembic upgrade head

# 启动应用
uvicorn stock_analysis_system.api.main:app --reload
```

> **💡 提示**: 推荐使用 Docker 方式，它能自动处理所有依赖和配置，避免环境问题。

## 📊 Usage Examples

### Spring Festival Analysis (✅ Available Now)

```python
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalAlignmentEngine
from stock_analysis_system.data.data_source_manager import DataSourceManager
from datetime import date

# Initialize components
engine = SpringFestivalAlignmentEngine(window_days=60)
data_manager = DataSourceManager()

# Get stock data
stock_data = await data_manager.get_stock_data("000001.SZ", 
                                               date(2020, 1, 1), 
                                               date(2024, 12, 31))

# Perform Spring Festival alignment analysis
aligned_data = engine.align_to_spring_festival(stock_data, years=[2020, 2021, 2022, 2023, 2024])
seasonal_pattern = engine.identify_seasonal_patterns(aligned_data)

print(f"Pattern strength: {seasonal_pattern.pattern_strength:.2f}")
print(f"Average return before SF: {seasonal_pattern.average_return_before:.2f}%")
print(f"Average return after SF: {seasonal_pattern.average_return_after:.2f}%")
print(f"Confidence level: {seasonal_pattern.confidence_level:.2f}")

# Generate trading signals
current_position = engine.get_current_position("000001.SZ")
signals = engine.generate_trading_signals(seasonal_pattern, current_position)
print(f"Signal: {signals['signal']} (strength: {signals['strength']:.2f})")
```

### Data Quality Analysis (✅ Available Now)

```python
from stock_analysis_system.data.data_quality_engine import EnhancedDataQualityEngine

# Initialize quality engine
quality_engine = EnhancedDataQualityEngine()

# Train ML anomaly detector (optional)
quality_engine.train_ml_detector(stock_data, 
                                 feature_columns=['open_price', 'high_price', 'low_price', 'close_price', 'volume'])

# Validate data quality
quality_report = quality_engine.validate_data(stock_data, dataset_name="000001.SZ Daily Data")

print(f"Overall quality score: {quality_report.overall_score:.2f}")
print(f"Issues found: {len(quality_report.issues)}")
print(f"Recommendations: {quality_report.recommendations}")

# Clean data automatically
cleaned_data = quality_engine.clean_data(stock_data, quality_report)
```

### Multi-Source Data Access (✅ Available Now)

```python
from stock_analysis_system.data.data_source_manager import DataSourceManager

# Initialize with automatic failover
data_manager = DataSourceManager()

# Get data with automatic source selection
stock_data = await data_manager.get_stock_data("000001.SZ", 
                                               date(2024, 1, 1), 
                                               date(2024, 12, 31))

# Get intraday data (if available)
intraday_data = await data_manager.get_intraday_data("000001.SZ", 
                                                     date(2024, 12, 1), 
                                                     date(2024, 12, 31), 
                                                     timeframe='5min')

# Check source health
health_status = data_manager.get_health_status()
for source, health in health_status.items():
    print(f"{source}: {health.status.value} (reliability: {health.reliability_score:.2f})")
```

## 🧪 Testing

### Database Migration Testing

Test the database setup without requiring PostgreSQL:

```bash
# Test migration with SQLite (no database server required)
python test_migration.py

# Test with specific database URL
DATABASE_URL="sqlite:///./test.db" alembic upgrade head
DATABASE_URL="sqlite:///./test.db" alembic current
```

### Application Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=stock_analysis_system --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Test with SQLite database
DATABASE_URL="sqlite:///./test.db" pytest
```

## 🔧 Development

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks for quality checks

Run quality checks manually:

```bash
# Format code
black stock_analysis_system tests

# Sort imports
isort stock_analysis_system tests

# Lint code
flake8 stock_analysis_system tests

# Type checking
mypy stock_analysis_system

# Run all quality checks at once
make lint
```

### Development Setup

Use the automated setup script:

```bash
# Automated development environment setup
python scripts/setup_dev.py

# Or use make commands
make setup-dev    # Full development setup
make install-dev  # Install development dependencies
make test         # Run tests
make run-dev      # Start development server
```

### Project Structure

```
stock_analysis_system/
├── analysis/           # Analysis engines (Spring Festival, Risk, etc.)
├── api/               # FastAPI application and routes
├── core/              # Core utilities and base classes
├── data/              # Data access layer and ETL
├── utils/             # Utility functions and helpers
└── visualization/     # Chart generation and visualization

tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
└── fixtures/          # Test data and fixtures

config/                # Configuration files
docs/                  # Documentation
scripts/               # Utility scripts
```

## 📈 Performance

The system delivers high performance through:

- **Async Architecture**: Full async/await implementation with FastAPI and asyncio
- **Intelligent Caching**: Redis-powered caching with configurable TTL and cache warming
- **Circuit Breaker Pattern**: Automatic failover prevents cascade failures
- **Connection Pooling**: PostgreSQL connection pooling for optimal database performance
- **Batch Processing**: Celery-powered background tasks for large dataset processing
- **Rate Limiting**: Smart rate limiting prevents API throttling
- **Response Times**: 
  - API health checks: <100ms
  - Stock data queries: <500ms (cached), <2s (fresh)
  - Spring Festival analysis: <5s for 5-year dataset
  - Data quality validation: <3s for 1000 records

## 🔒 Security

Security features include:

- **JWT Authentication**: Secure API access
- **Input Validation**: Pydantic models for data validation
- **SQL Injection Protection**: SQLAlchemy ORM
- **Rate Limiting**: API rate limiting
- **Encryption**: Sensitive data encryption

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Chinese calendar calculations based on astronomical data
- Financial data provided by AkShare and Tushare
- Visualization powered by Plotly and D3.js
- Machine learning capabilities via scikit-learn and MLflow

## 📞 Support

For support and questions:

- 📧 Email: support@stockanalysis.com
- 💬 Discord: [Join our community](https://discord.gg/stockanalysis)
- 📖 Documentation: [Read the docs](https://stock-analysis-system.readthedocs.io/)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/stock-analysis-system/issues)