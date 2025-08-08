# 爬虫接口集成系统开发者指南

## 目录

1. [开发环境搭建](#开发环境搭建)
2. [系统架构](#系统架构)
3. [代码结构](#代码结构)
4. [开发规范](#开发规范)
5. [扩展开发](#扩展开发)
6. [测试指南](#测试指南)
7. [部署指南](#部署指南)
8. [贡献指南](#贡献指南)

## 开发环境搭建

### 系统要求

- Python 3.8+
- PostgreSQL 12+
- Redis 6.0+
- Node.js 16+ (前端开发)
- Git 2.20+

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/your-org/stock-analysis-system.git
cd stock-analysis-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置数据库等信息

# 5. 初始化数据库
alembic upgrade head

# 6. 运行测试
pytest

# 7. 启动开发服务器
python start_server.py --debug
```

### 开发工具配置

#### VS Code 配置

创建 `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### Pre-commit 钩子

```bash
# 安装 pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行检查
pre-commit run --all-files
```#
# 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层                                │
├─────────────────────────────────────────────────────────────┤
│  React Web App  │  Mobile App  │  Third-party Clients      │
└─────────────────┴──────────────┴───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        API网关层                             │
├─────────────────────────────────────────────────────────────┤
│  FastAPI  │  认证中间件  │  限流中间件  │  日志中间件        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        业务逻辑层                            │
├─────────────────────────────────────────────────────────────┤
│  数据管理器  │  分析引擎  │  监控系统  │  错误处理          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        数据访问层                            │
├─────────────────────────────────────────────────────────────┤
│  数据适配器  │  缓存管理  │  数据验证  │  故障转移          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        存储层                                │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL  │  Redis  │  外部数据源  │  文件存储          │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. 数据源管理器 (DataSourceManager)

负责管理多个数据源，提供统一的数据访问接口。

```python
class DataSourceManager:
    """数据源管理器"""
    
    def __init__(self):
        self.adapters = {}
        self.health_monitor = HealthMonitor()
        self.failover_manager = FailoverManager()
    
    async def get_data(self, request: DataRequest) -> DataResponse:
        """获取数据的统一入口"""
        pass
    
    def register_adapter(self, name: str, adapter: DataAdapter):
        """注册数据适配器"""
        pass
```

#### 2. 数据适配器 (DataAdapter)

为不同数据源提供统一接口。

```python
class DataAdapter(ABC):
    """数据适配器基类"""
    
    @abstractmethod
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        pass
    
    @abstractmethod
    async def get_history_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """获取历史数据"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
```

#### 3. 缓存管理器 (CacheManager)

管理多级缓存策略。

```python
class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis_client = redis.Redis()
        self.local_cache = {}
        self.cache_strategies = {}
    
    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """获取缓存数据"""
        pass
    
    async def set(self, key: str, value: Any, ttl: int, cache_type: str = 'default'):
        """设置缓存数据"""
        pass
```## 代码结构


### 项目目录结构

```
stock_analysis_system/
├── stock_analysis_system/          # 主要代码目录
│   ├── __init__.py
│   ├── api/                        # API层
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI应用入口
│   │   ├── auth.py                 # 认证相关
│   │   ├── data_endpoints.py       # 数据接口
│   │   └── system_endpoints.py     # 系统管理接口
│   ├── core/                       # 核心组件
│   │   ├── __init__.py
│   │   ├── database.py             # 数据库连接
│   │   ├── error_handler.py        # 错误处理
│   │   ├── degradation_strategy.py # 降级策略
│   │   └── failover_mechanism.py   # 故障转移
│   ├── data/                       # 数据层
│   │   ├── __init__.py
│   │   ├── data_source_manager.py  # 数据源管理
│   │   ├── cache_manager.py        # 缓存管理
│   │   ├── adapters/               # 数据适配器
│   │   │   ├── __init__.py
│   │   │   ├── tushare_adapter.py
│   │   │   ├── akshare_adapter.py
│   │   │   └── base_adapter.py
│   │   └── models.py               # 数据模型
│   ├── analysis/                   # 分析引擎
│   │   ├── __init__.py
│   │   ├── technical_analysis.py
│   │   └── risk_analysis.py
│   ├── monitoring/                 # 监控系统
│   │   ├── __init__.py
│   │   ├── health_monitor.py
│   │   └── metrics_collector.py
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── validators.py
│       └── helpers.py
├── tests/                          # 测试代码
│   ├── __init__.py
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   └── conftest.py                 # 测试配置
├── docs/                           # 文档
├── examples/                       # 示例代码
├── scripts/                        # 脚本文件
├── alembic/                        # 数据库迁移
├── requirements.txt                # 生产依赖
├── requirements-dev.txt            # 开发依赖
├── .env.example                    # 环境变量示例
├── .gitignore
├── README.md
└── setup.py
```

### 模块依赖关系

```mermaid
graph TD
    A[API层] --> B[业务逻辑层]
    B --> C[数据访问层]
    C --> D[存储层]
    
    B --> E[监控系统]
    B --> F[错误处理]
    B --> G[缓存管理]
    
    C --> H[数据适配器]
    C --> I[数据验证]
    
    E --> J[健康检查]
    E --> K[指标收集]
    
    F --> L[降级策略]
    F --> M[故障转移]
```

## 开发规范

### 代码风格

项目使用以下代码风格工具：

- **Black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查
- **mypy**: 类型检查

#### 配置文件

`pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```#
## 编码规范

#### 1. 命名规范

```python
# 类名使用 PascalCase
class DataSourceManager:
    pass

# 函数和变量使用 snake_case
def get_stock_data():
    stock_symbol = "000001.SZ"
    return stock_symbol

# 常量使用 UPPER_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# 私有方法使用下划线前缀
def _internal_method():
    pass
```

#### 2. 文档字符串

使用 Google 风格的文档字符串：

```python
def get_stock_data(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """获取股票历史数据.
    
    Args:
        symbol: 股票代码，如 '000001.SZ'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        包含股票数据的字典列表，每个字典包含日期、开盘价、收盘价等信息
    
    Raises:
        ValueError: 当股票代码格式不正确时
        DataSourceError: 当数据源不可用时
    
    Example:
        >>> data = get_stock_data('000001.SZ', '2024-01-01', '2024-01-31')
        >>> len(data)
        21
    """
    pass
```

#### 3. 类型注解

所有公共函数和方法都应该有类型注解：

```python
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

def process_market_data(
    data: List[Dict[str, Any]], 
    symbol: str,
    timestamp: Optional[datetime] = None
) -> Dict[str, Union[float, int, str]]:
    """处理市场数据"""
    pass
```

#### 4. 错误处理

使用自定义异常类：

```python
class StockAnalysisError(Exception):
    """股票分析系统基础异常"""
    pass

class DataSourceError(StockAnalysisError):
    """数据源相关异常"""
    pass

class ValidationError(StockAnalysisError):
    """数据验证异常"""
    pass

# 使用示例
def validate_symbol(symbol: str) -> None:
    if not symbol or len(symbol) < 6:
        raise ValidationError(f"Invalid symbol format: {symbol}")
```

#### 5. 日志记录

使用结构化日志：

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def log_api_request(endpoint: str, params: Dict[str, Any], response_time: float):
    """记录API请求日志"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'params': params,
        'response_time': response_time,
        'level': 'INFO'
    }
    logger.info(json.dumps(log_data, ensure_ascii=False))
```

### 配置管理

使用 Pydantic 进行配置管理：

```python
from pydantic import BaseSettings, Field
from typing import List, Optional

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    host: str = Field(..., env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    username: str = Field(..., env="DB_USERNAME")
    password: str = Field(..., env="DB_PASSWORD")
    database: str = Field(..., env="DB_DATABASE")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisSettings(BaseSettings):
    """Redis配置"""
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")

class AppSettings(BaseSettings):
    """应用配置"""
    debug: bool = Field(False, env="DEBUG")
    secret_key: str = Field(..., env="SECRET_KEY")
    allowed_hosts: List[str] = Field(["localhost"], env="ALLOWED_HOSTS")
    
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# 全局配置实例
settings = AppSettings()
```## 扩展开发


### 添加新的数据源

#### 1. 创建数据适配器

```python
# stock_analysis_system/data/adapters/new_source_adapter.py

from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from .base_adapter import BaseDataAdapter

class NewSourceAdapter(BaseDataAdapter):
    """新数据源适配器"""
    
    def __init__(self, api_key: str, base_url: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """初始化连接"""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        url = f"{self.base_url}/realtime/{symbol}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return self._normalize_realtime_data(data)
            else:
                raise DataSourceError(f"Failed to get data: {response.status}")
    
    async def get_history_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> List[Dict[str, Any]]:
        """获取历史数据"""
        url = f"{self.base_url}/history/{symbol}"
        params = {'start': start_date, 'end': end_date}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [self._normalize_history_data(item) for item in data]
            else:
                raise DataSourceError(f"Failed to get history data: {response.status}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            url = f"{self.base_url}/health"
            async with self.session.get(url, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    def _normalize_realtime_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化实时数据格式"""
        return {
            'symbol': raw_data['code'],
            'name': raw_data['name'],
            'price': float(raw_data['current_price']),
            'change': float(raw_data['price_change']),
            'change_pct': float(raw_data['change_percent']),
            'volume': int(raw_data['volume']),
            'amount': float(raw_data['turnover']),
            'timestamp': raw_data['update_time']
        }
    
    def _normalize_history_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化历史数据格式"""
        return {
            'date': raw_data['trade_date'],
            'open': float(raw_data['open_price']),
            'high': float(raw_data['high_price']),
            'low': float(raw_data['low_price']),
            'close': float(raw_data['close_price']),
            'volume': int(raw_data['volume']),
            'amount': float(raw_data['amount'])
        }
```

#### 2. 注册适配器

```python
# stock_analysis_system/data/data_source_manager.py

from .adapters.new_source_adapter import NewSourceAdapter

class DataSourceManager:
    def __init__(self):
        self.adapters = {}
        self._register_default_adapters()
    
    def _register_default_adapters(self):
        """注册默认适配器"""
        # 注册新数据源
        new_source = NewSourceAdapter(
            api_key=settings.new_source_api_key,
            base_url=settings.new_source_base_url
        )
        self.register_adapter('new_source', new_source, priority=3)
    
    def register_adapter(self, name: str, adapter: BaseDataAdapter, priority: int = 1):
        """注册数据适配器"""
        self.adapters[name] = {
            'adapter': adapter,
            'priority': priority,
            'enabled': True
        }
```

#### 3. 添加配置

```python
# config/settings.py

class AppSettings(BaseSettings):
    # 新数据源配置
    new_source_api_key: str = Field(..., env="NEW_SOURCE_API_KEY")
    new_source_base_url: str = Field(..., env="NEW_SOURCE_BASE_URL")
```

#### 4. 添加测试

```python
# tests/unit/test_new_source_adapter.py

import pytest
from unittest.mock import AsyncMock, patch
from stock_analysis_system.data.adapters.new_source_adapter import NewSourceAdapter

@pytest.fixture
async def adapter():
    adapter = NewSourceAdapter("test_key", "https://api.example.com")
    await adapter.initialize()
    yield adapter
    await adapter.cleanup()

@pytest.mark.asyncio
async def test_get_realtime_data(adapter):
    """测试获取实时数据"""
    mock_response = {
        'code': '000001.SZ',
        'name': '平安银行',
        'current_price': 10.50,
        'price_change': 0.15,
        'change_percent': 1.45,
        'volume': 1000000,
        'turnover': 10500000,
        'update_time': '2024-01-01T15:00:00Z'
    }
    
    with patch.object(adapter.session, 'get') as mock_get:
        mock_get.return_value.__aenter__.return_value.status = 200
        mock_get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        result = await adapter.get_realtime_data('000001.SZ')
        
        assert result['symbol'] == '000001.SZ'
        assert result['price'] == 10.50
        assert result['change_pct'] == 1.45
```###
 添加新的API端点

#### 1. 定义数据模型

```python
# stock_analysis_system/api/models.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class NewDataRequest(BaseModel):
    """新数据请求模型"""
    symbol: str = Field(..., description="股票代码")
    data_type: str = Field(..., description="数据类型")
    parameters: Optional[dict] = Field(None, description="额外参数")

class NewDataResponse(BaseModel):
    """新数据响应模型"""
    success: bool = Field(..., description="请求是否成功")
    data: List[dict] = Field(..., description="返回数据")
    metadata: dict = Field(..., description="元数据")
    timestamp: datetime = Field(..., description="响应时间")
```

#### 2. 实现业务逻辑

```python
# stock_analysis_system/services/new_data_service.py

from typing import Dict, List, Any
from ..data.data_source_manager import DataSourceManager
from ..core.error_handler import ErrorHandler

class NewDataService:
    """新数据服务"""
    
    def __init__(self, data_manager: DataSourceManager, error_handler: ErrorHandler):
        self.data_manager = data_manager
        self.error_handler = error_handler
    
    async def process_new_data_request(self, request: NewDataRequest) -> Dict[str, Any]:
        """处理新数据请求"""
        try:
            # 业务逻辑处理
            data = await self._fetch_data(request)
            processed_data = await self._process_data(data, request.parameters)
            
            return {
                'success': True,
                'data': processed_data,
                'metadata': {
                    'source': 'new_service',
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, custom_message=f"New data request failed: {request.symbol}")
            raise
    
    async def _fetch_data(self, request: NewDataRequest) -> List[Dict[str, Any]]:
        """获取数据"""
        # 实现数据获取逻辑
        pass
    
    async def _process_data(self, data: List[Dict[str, Any]], parameters: dict) -> List[Dict[str, Any]]:
        """处理数据"""
        # 实现数据处理逻辑
        pass
```

#### 3. 创建API端点

```python
# stock_analysis_system/api/new_endpoints.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..services.new_data_service import NewDataService
from .models import NewDataRequest, NewDataResponse
from .dependencies import get_current_user, get_new_data_service

router = APIRouter(prefix="/new-data", tags=["new-data"])

@router.post("/process", response_model=NewDataResponse)
async def process_new_data(
    request: NewDataRequest,
    service: NewDataService = Depends(get_new_data_service),
    current_user = Depends(get_current_user)
):
    """处理新数据请求"""
    try:
        result = await service.process_new_data_request(request)
        return NewDataResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_service_status():
    """获取服务状态"""
    return {"status": "active", "version": "1.0.0"}
```

#### 4. 注册路由

```python
# stock_analysis_system/api/main.py

from fastapi import FastAPI
from .new_endpoints import router as new_data_router

app = FastAPI(title="Stock Analysis System API")

# 注册路由
app.include_router(new_data_router, prefix="/api/v1")
```

### 添加新的分析算法

#### 1. 创建分析器基类

```python
# stock_analysis_system/analysis/base_analyzer.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class BaseAnalyzer(ABC):
    """分析器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def analyze(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        pass
    
    def get_required_columns(self) -> List[str]:
        """获取必需的数据列"""
        return ['date', 'open', 'high', 'low', 'close', 'volume']
```

#### 2. 实现具体分析器

```python
# stock_analysis_system/analysis/custom_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_analyzer import BaseAnalyzer

class CustomTechnicalAnalyzer(BaseAnalyzer):
    """自定义技术分析器"""
    
    def __init__(self):
        super().__init__("custom_technical")
    
    async def analyze(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行技术分析"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
        
        # 计算技术指标
        indicators = self._calculate_indicators(data, parameters)
        
        # 生成信号
        signals = self._generate_signals(data, indicators, parameters)
        
        # 计算评分
        score = self._calculate_score(indicators, signals)
        
        return {
            'indicators': indicators,
            'signals': signals,
            'score': score,
            'analysis_time': pd.Timestamp.now().isoformat()
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据"""
        required_columns = self.get_required_columns()
        return all(col in data.columns for col in required_columns) and len(data) > 0
    
    def _calculate_indicators(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """计算技术指标"""
        # 移动平均线
        ma_period = parameters.get('ma_period', 20)
        data[f'ma_{ma_period}'] = data['close'].rolling(window=ma_period).mean()
        
        # RSI
        rsi_period = parameters.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9).mean()
        
        return {
            'ma': data[f'ma_{ma_period}'].iloc[-1],
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal.iloc[-1]
        }
    
    def _generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any], parameters: Dict[str, Any]) -> List[str]:
        """生成交易信号"""
        signals = []
        
        # RSI信号
        if indicators['rsi'] > 70:
            signals.append('RSI超买')
        elif indicators['rsi'] < 30:
            signals.append('RSI超卖')
        
        # MACD信号
        if indicators['macd'] > indicators['macd_signal']:
            signals.append('MACD金叉')
        elif indicators['macd'] < indicators['macd_signal']:
            signals.append('MACD死叉')
        
        # 价格与均线关系
        current_price = data['close'].iloc[-1]
        if current_price > indicators['ma']:
            signals.append('价格突破均线')
        elif current_price < indicators['ma']:
            signals.append('价格跌破均线')
        
        return signals
    
    def _calculate_score(self, indicators: Dict[str, Any], signals: List[str]) -> int:
        """计算综合评分"""
        score = 50  # 基础分
        
        # 根据RSI调整分数
        rsi = indicators['rsi']
        if 30 <= rsi <= 70:
            score += 10  # RSI在正常范围
        elif rsi < 30:
            score += 5   # 超卖，可能反弹
        else:
            score -= 5   # 超买，可能回调
        
        # 根据MACD调整分数
        if indicators['macd'] > indicators['macd_signal']:
            score += 10
        else:
            score -= 10
        
        # 根据信号数量调整
        score += len(signals) * 2
        
        return max(0, min(100, score))
```## 
测试指南

### 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_data_adapters.py
│   ├── test_cache_manager.py
│   └── test_analyzers.py
├── integration/             # 集成测试
│   ├── test_api_endpoints.py
│   ├── test_data_flow.py
│   └── test_database.py
├── performance/             # 性能测试
│   ├── test_load.py
│   └── test_stress.py
├── fixtures/                # 测试数据
│   ├── sample_data.json
│   └── mock_responses.py
└── conftest.py             # 测试配置
```

### 测试配置

```python
# tests/conftest.py

import pytest
import asyncio
from unittest.mock import AsyncMock
from stock_analysis_system.core.database import get_database
from stock_analysis_system.data.data_source_manager import DataSourceManager

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_database():
    """测试数据库连接"""
    # 使用测试数据库
    db = get_database("sqlite:///test.db")
    yield db
    await db.close()

@pytest.fixture
def mock_data_manager():
    """模拟数据管理器"""
    manager = AsyncMock(spec=DataSourceManager)
    manager.get_realtime_data.return_value = {
        'symbol': '000001.SZ',
        'price': 10.50,
        'change': 0.15,
        'change_pct': 1.45
    }
    return manager

@pytest.fixture
def sample_stock_data():
    """示例股票数据"""
    return [
        {
            'date': '2024-01-01',
            'open': 10.00,
            'high': 10.50,
            'low': 9.80,
            'close': 10.30,
            'volume': 1000000
        },
        {
            'date': '2024-01-02',
            'open': 10.30,
            'high': 10.80,
            'low': 10.10,
            'close': 10.65,
            'volume': 1200000
        }
    ]
```

### 单元测试示例

```python
# tests/unit/test_data_adapters.py

import pytest
from unittest.mock import AsyncMock, patch
from stock_analysis_system.data.adapters.tushare_adapter import TushareAdapter

class TestTushareAdapter:
    """Tushare适配器测试"""
    
    @pytest.fixture
    def adapter(self):
        return TushareAdapter(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_success(self, adapter):
        """测试成功获取实时数据"""
        mock_response = {
            'ts_code': '000001.SZ',
            'name': '平安银行',
            'trade': 10.50,
            'pct_chg': 1.45,
            'vol': 1000000
        }
        
        with patch.object(adapter, '_make_request', return_value=mock_response):
            result = await adapter.get_realtime_data('000001.SZ')
            
            assert result['symbol'] == '000001.SZ'
            assert result['price'] == 10.50
            assert result['change_pct'] == 1.45
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_failure(self, adapter):
        """测试获取实时数据失败"""
        with patch.object(adapter, '_make_request', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                await adapter.get_realtime_data('000001.SZ')
    
    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """测试健康检查"""
        with patch.object(adapter, '_make_request', return_value={'status': 'ok'}):
            result = await adapter.health_check()
            assert result is True
```

### 集成测试示例

```python
# tests/integration/test_api_endpoints.py

import pytest
from fastapi.testclient import TestClient
from stock_analysis_system.api.main import app

class TestAPIEndpoints:
    """API端点集成测试"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """获取认证头"""
        response = client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_realtime_data(self, client, auth_headers):
        """测试获取实时数据"""
        response = client.get(
            "/api/v1/data/realtime/000001.SZ",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["symbol"] == "000001.SZ"
    
    def test_unauthorized_access(self, client):
        """测试未授权访问"""
        response = client.get("/api/v1/data/realtime/000001.SZ")
        assert response.status_code == 401
```

### 性能测试

```python
# tests/performance/test_load.py

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import requests

class TestPerformance:
    """性能测试"""
    
    @pytest.fixture
    def base_url(self):
        return "http://localhost:8000"
    
    @pytest.fixture
    def auth_token(self, base_url):
        """获取认证令牌"""
        response = requests.post(f"{base_url}/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        return response.json()["access_token"]
    
    def test_concurrent_requests(self, base_url, auth_token):
        """测试并发请求性能"""
        headers = {"Authorization": f"Bearer {auth_token}"}
        url = f"{base_url}/api/v1/data/realtime/000001.SZ"
        
        def make_request():
            start_time = time.time()
            response = requests.get(url, headers=headers)
            end_time = time.time()
            return {
                'status_code': response.status_code,
                'response_time': end_time - start_time
            }
        
        # 并发50个请求
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # 分析结果
        successful_requests = [r for r in results if r['status_code'] == 200]
        response_times = [r['response_time'] for r in successful_requests]
        
        assert len(successful_requests) >= 45  # 至少90%成功率
        assert max(response_times) < 5.0       # 最大响应时间不超过5秒
        assert sum(response_times) / len(response_times) < 1.0  # 平均响应时间不超过1秒
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """测试内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量操作
        tasks = []
        for i in range(1000):
            # 模拟数据处理任务
            task = asyncio.create_task(self._simulate_data_processing())
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长不应超过100MB
        assert memory_increase < 100 * 1024 * 1024
    
    async def _simulate_data_processing(self):
        """模拟数据处理"""
        # 模拟一些计算密集型操作
        data = list(range(1000))
        result = sum(x * x for x in data)
        await asyncio.sleep(0.001)  # 模拟异步操作
        return result
```##
 部署指南

### Docker部署

#### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "start_server.py"]
```

#### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/stockdb
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=stockdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes部署

#### 1. 部署配置

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-analysis-api
  labels:
    app: stock-analysis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stock-analysis-api
  template:
    metadata:
      labels:
        app: stock-analysis-api
    spec:
      containers:
      - name: api
        image: stock-analysis:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### 2. 服务配置

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: stock-analysis-service
spec:
  selector:
    app: stock-analysis-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 3. Ingress配置

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: stock-analysis-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.stockanalysis.com
    secretName: stock-analysis-tls
  rules:
  - host: api.stockanalysis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: stock-analysis-service
            port:
              number: 80
```

### CI/CD流水线

#### GitHub Actions

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 stock_analysis_system tests
        black --check stock_analysis_system tests
        isort --check-only stock_analysis_system tests
    
    - name: Run type checking
      run: mypy stock_analysis_system
    
    - name: Run tests
      run: |
        pytest tests/ --cov=stock_analysis_system --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t stock-analysis:${{ github.sha }} .
        docker tag stock-analysis:${{ github.sha }} stock-analysis:latest
    
    - name: Deploy to staging
      run: |
        # 部署到测试环境的脚本
        echo "Deploying to staging..."
    
    - name: Run integration tests
      run: |
        # 运行集成测试
        echo "Running integration tests..."
    
    - name: Deploy to production
      if: success()
      run: |
        # 部署到生产环境的脚本
        echo "Deploying to production..."
```

## 贡献指南

### 贡献流程

1. **Fork项目**
   ```bash
   # 在GitHub上fork项目
   git clone https://github.com/your-username/stock-analysis-system.git
   cd stock-analysis-system
   ```

2. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **开发和测试**
   ```bash
   # 进行开发
   # 运行测试
   pytest tests/
   
   # 检查代码质量
   flake8 stock_analysis_system
   black stock_analysis_system
   mypy stock_analysis_system
   ```

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **推送分支**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建Pull Request**
   - 在GitHub上创建PR
   - 填写详细的描述
   - 等待代码审查

### 提交信息规范

使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat(data): add new data source adapter for Wind API

- Implement WindAdapter class with standard interface
- Add configuration for Wind API credentials
- Include comprehensive unit tests
- Update documentation

Closes #123
```

### 代码审查清单

在提交PR前，请确保：

- [ ] 代码遵循项目编码规范
- [ ] 所有测试通过
- [ ] 代码覆盖率不低于80%
- [ ] 添加了必要的文档
- [ ] 更新了CHANGELOG.md
- [ ] 没有引入安全漏洞
- [ ] 性能没有明显下降
- [ ] 向后兼容性得到保证

### 发布流程

1. **更新版本号**
   ```bash
   # 更新 setup.py 中的版本号
   # 更新 CHANGELOG.md
   ```

2. **创建发布分支**
   ```bash
   git checkout -b release/v1.2.0
   ```

3. **最终测试**
   ```bash
   pytest tests/
   # 运行完整的集成测试
   ```

4. **合并到主分支**
   ```bash
   git checkout main
   git merge release/v1.2.0
   git tag v1.2.0
   git push origin main --tags
   ```

5. **发布到PyPI**
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

### 社区支持

- **问题反馈**: [GitHub Issues](https://github.com/your-org/stock-analysis-system/issues)
- **功能请求**: [GitHub Discussions](https://github.com/your-org/stock-analysis-system/discussions)
- **技术交流**: [Discord社区](https://discord.gg/your-invite)
- **邮件列表**: dev@stockanalysis.com

感谢您对项目的贡献！