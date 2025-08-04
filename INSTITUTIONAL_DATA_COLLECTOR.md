# Institutional Data Collector

## Overview

The Institutional Data Collector implements comprehensive institutional data collection including dragon-tiger list data, shareholder information, block trades, and institutional classification for the stock analysis system. This implementation addresses task 6.1 of the stock analysis system specification.

## Key Features

### Data Collection Sources

1. **Dragon-Tiger List Collection**
   - Daily trading seat data for unusual market activity
   - Buy and sell transaction amounts
   - Trading seat identification and classification
   - Market-specific data (Shanghai/Shenzhen)

2. **Shareholder Data Collection**
   - Quarterly top-10 shareholder reports
   - Shareholding percentages and share counts
   - Shareholder change tracking
   - Institutional vs. individual classification

3. **Block Trade Collection**
   - Large block transaction data
   - Buyer and seller seat information
   - Price discount/premium analysis
   - Volume and amount tracking

### Institutional Classification

1. **Comprehensive Institution Types**
   - Mutual Funds
   - Social Security Funds
   - QFII/RQFII (Foreign Institutional Investors)
   - Insurance Companies
   - Securities Firms
   - Banks
   - Hot Money/Speculative Capital
   - Other institutional types

2. **Pattern-Based Classification**
   - Regular expression matching
   - Confidence scoring
   - Alias and pattern recognition
   - Caching for performance

3. **Activity Timeline Tracking**
   - Consolidated activity records
   - Multi-source data integration
   - Chronological activity ordering
   - Institution behavior analysis

## Architecture

### Core Components

```
InstitutionalDataCollector
├── InstitutionClassifier
├── DragonTigerCollector
├── ShareholderCollector
├── BlockTradeCollector
└── Activity Consolidation Engine
```

### Data Flow

1. **Collection Phase**
   - Parallel data collection from multiple sources
   - Institution identification and classification
   - Data validation and standardization

2. **Consolidation Phase**
   - Activity timeline generation
   - Cross-source data correlation
   - Institution behavior tracking

3. **Analysis Phase**
   - Summary statistics generation
   - Institution type analysis
   - Activity pattern identification

## Usage

### Basic Data Collection

```python
import asyncio
from datetime import date
from stock_analysis_system.analysis.institutional_data_collector import InstitutionalDataCollector

# Initialize collector
collector = InstitutionalDataCollector()

# Collect data for multiple stocks
async def collect_data():
    stock_codes = ["000001", "000002", "600000"]
    start_date = date(2023, 1, 1)
    end_date = date(2023, 3, 31)
    
    all_data = await collector.collect_all_data(stock_codes, start_date, end_date)
    
    for stock_code, data in all_data.items():
        print(f"{stock_code}:")
        print(f"  Dragon-Tiger: {len(data['dragon_tiger'])} records")
        print(f"  Shareholders: {len(data['shareholders'])} records")
        print(f"  Block Trades: {len(data['block_trades'])} records")

asyncio.run(collect_data())
```

### Institution Classification

```python
from stock_analysis_system.analysis.institutional_data_collector import InstitutionClassifier

# Initialize classifier
classifier = InstitutionClassifier()

# Classify institutions
institutions = [
    "易方达基金管理有限公司",
    "全国社会保障基金理事会",
    "中信证券股份有限公司",
    "摩根士丹利QFII"
]

for name in institutions:
    institution_type, confidence = classifier.classify_institution(name)
    print(f"{name}: {institution_type.value} ({confidence:.1%} confidence)")
    
    # Create institution object
    institution = classifier.create_institution(name)
    print(f"  ID: {institution.institution_id}")
    print(f"  Type: {institution.institution_type.value}")
```

### Activity Timeline Analysis

```python
# Get activity timeline for a stock
stock_code = "000001"
activities = collector.get_institution_activity_timeline(stock_code)

print(f"Total activities for {stock_code}: {len(activities)}")

# Filter by institution type
from stock_analysis_system.analysis.institutional_data_collector import InstitutionType

fund_activities = collector.get_institution_activity_timeline(
    stock_code, InstitutionType.MUTUAL_FUND
)
print(f"Mutual fund activities: {len(fund_activities)}")

# Get summary statistics
summary = collector.get_institution_summary(stock_code)
print(f"Summary: {summary}")
```

### Data Export

```python
# Export to pandas DataFrames
dataframes = collector.export_data_to_dataframes("000001")

activities_df = dataframes['activities']
summary_df = dataframes['summary']

print(f"Activities DataFrame shape: {activities_df.shape}")
print(f"Columns: {list(activities_df.columns)}")

# Analyze by institution type
type_analysis = activities_df.groupby('institution_type').agg({
    'amount': ['count', 'sum', 'mean'],
    'confidence_score': 'mean'
})
print(type_analysis)
```

## Data Structures

### InstitutionalInvestor

```python
@dataclass
class InstitutionalInvestor:
    institution_id: str                    # Unique identifier
    name: str                             # Institution name
    institution_type: InstitutionType     # Classification type
    
    # Optional details
    parent_company: Optional[str] = None
    fund_manager: Optional[str] = None
    registration_country: Optional[str] = None
    aum: Optional[float] = None           # Assets Under Management
    
    # Pattern matching
    name_patterns: List[str]              # Name matching patterns
    aliases: List[str]                    # Known aliases
    
    # Metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence_score: float = 1.0         # Classification confidence
```

### DragonTigerRecord

```python
@dataclass
class DragonTigerRecord:
    trade_date: date                      # Trading date
    stock_code: str                       # Stock symbol
    stock_name: str                       # Stock name
    
    # Trading information
    seat_name: str                        # Trading seat name
    seat_type: str                        # "buy" or "sell"
    amount: float                         # Trading amount (yuan)
    net_amount: Optional[float] = None    # Net amount if available
    
    # Institution identification
    institution: Optional[InstitutionalInvestor] = None
    institution_confidence: float = 0.0
    
    # Metadata
    rank: Optional[int] = None            # Rank in Dragon-Tiger list
    market: Optional[str] = None          # "SH" or "SZ"
    reason: Optional[str] = None          # Inclusion reason
```

### ShareholderRecord

```python
@dataclass
class ShareholderRecord:
    report_date: date                     # Report date
    stock_code: str                       # Stock symbol
    stock_name: str                       # Stock name
    
    # Shareholder information
    shareholder_name: str                 # Shareholder name
    shareholding_ratio: float             # Percentage holding
    shares_held: Optional[int] = None     # Number of shares
    shares_change: Optional[int] = None   # Change in shares
    
    # Institution identification
    institution: Optional[InstitutionalInvestor] = None
    institution_confidence: float = 0.0
    
    # Metadata
    rank: Optional[int] = None            # Shareholder rank
    shareholder_type: Optional[str] = None
    is_restricted: bool = False           # Restricted shares flag
```

### BlockTradeRecord

```python
@dataclass
class BlockTradeRecord:
    trade_date: date                      # Trading date
    stock_code: str                       # Stock symbol
    stock_name: str                       # Stock name
    
    # Trade details
    volume: int                           # Number of shares
    price: float                          # Trade price
    total_amount: float                   # Total amount
    
    # Counterparty information
    buyer_seat: Optional[str] = None
    seller_seat: Optional[str] = None
    buyer_institution: Optional[InstitutionalInvestor] = None
    seller_institution: Optional[InstitutionalInvestor] = None
    
    # Price analysis
    discount_rate: Optional[float] = None  # Discount to market
    premium_rate: Optional[float] = None   # Premium to market
    market_price: Optional[float] = None   # Market price reference
```

### InstitutionalActivity

```python
@dataclass
class InstitutionalActivity:
    activity_id: str                      # Unique activity ID
    activity_date: date                   # Activity date
    stock_code: str                       # Stock symbol
    institution: InstitutionalInvestor   # Institution involved
    activity_type: ActivityType           # Type of activity
    
    # Activity details
    amount: Optional[float] = None        # Monetary amount
    volume: Optional[int] = None          # Share volume
    price: Optional[float] = None         # Price per share
    
    # Context
    market_cap: Optional[float] = None
    daily_volume: Optional[int] = None
    price_impact: Optional[float] = None
    
    # Source tracking
    source_type: str = "unknown"          # Data source
    source_record_id: Optional[str] = None
    confidence_score: float = 1.0
```

## Institution Types

### Supported Institution Types

```python
class InstitutionType(str, Enum):
    MUTUAL_FUND = "mutual_fund"           # 基金公司
    SOCIAL_SECURITY = "social_security"   # 社保基金
    QFII = "qfii"                        # 合格境外机构投资者
    RQFII = "rqfii"                      # 人民币合格境外机构投资者
    HOT_MONEY = "hot_money"              # 游资/热钱
    INSURANCE = "insurance"               # 保险公司
    PENSION_FUND = "pension_fund"         # 养老基金
    PRIVATE_EQUITY = "private_equity"     # 私募股权
    HEDGE_FUND = "hedge_fund"            # 对冲基金
    BANK = "bank"                        # 银行
    SECURITIES_FIRM = "securities_firm"   # 证券公司
    TRUST = "trust"                      # 信托公司
    OTHER = "other"                      # 其他
```

### Activity Types

```python
class ActivityType(str, Enum):
    DRAGON_TIGER_BUY = "dragon_tiger_buy"         # 龙虎榜买入
    DRAGON_TIGER_SELL = "dragon_tiger_sell"       # 龙虎榜卖出
    BLOCK_TRADE_BUY = "block_trade_buy"           # 大宗交易买入
    BLOCK_TRADE_SELL = "block_trade_sell"         # 大宗交易卖出
    SHAREHOLDING_INCREASE = "shareholding_increase" # 持股增加
    SHAREHOLDING_DECREASE = "shareholding_decrease" # 持股减少
    NEW_POSITION = "new_position"                 # 新建仓位
    POSITION_EXIT = "position_exit"               # 清仓退出
```

## Classification Rules

### Pattern-Based Classification

The classifier uses regular expressions to identify institution types:

```python
classification_rules = {
    InstitutionType.MUTUAL_FUND: [
        {"pattern": r".*基金.*", "confidence": 0.9},
        {"pattern": r".*fund.*", "confidence": 0.8, "case_sensitive": False},
        {"pattern": r".*资产管理.*", "confidence": 0.7}
    ],
    InstitutionType.SOCIAL_SECURITY: [
        {"pattern": r".*社保.*", "confidence": 0.95},
        {"pattern": r".*社会保障.*", "confidence": 0.95},
        {"pattern": r".*全国社会保障基金.*", "confidence": 1.0}
    ],
    # ... more rules
}
```

### Confidence Scoring

- **1.0**: Perfect match (e.g., "全国社会保障基金理事会")
- **0.9**: High confidence (e.g., ".*基金.*")
- **0.8**: Good confidence (e.g., ".*securities.*")
- **0.7**: Moderate confidence (e.g., ".*资产管理.*")
- **0.6**: Lower confidence (e.g., company name patterns)
- **0.0**: No match (classified as OTHER)

## Data Collection Process

### Collection Workflow

1. **Initialization**
   - Create collector instances
   - Initialize classification engine
   - Set up async session management

2. **Parallel Collection**
   - Dragon-Tiger list data collection
   - Shareholder data collection
   - Block trade data collection
   - Institution classification

3. **Data Consolidation**
   - Activity timeline generation
   - Cross-source correlation
   - Duplicate detection and removal

4. **Quality Assessment**
   - Classification confidence scoring
   - Data completeness analysis
   - Consistency validation

### Mock Data Generation

For demonstration purposes, the collectors generate realistic mock data:

- **Dragon-Tiger**: 10% daily probability, 1-5 buy/sell records
- **Shareholders**: Quarterly reports with top-10 shareholders
- **Block Trades**: 5% daily probability, 1-3 trades per day

## Performance Considerations

### Async Operations

All data collection operations are asynchronous:

```python
async with collector.dragon_tiger_collector, \
          collector.shareholder_collector, \
          collector.block_trade_collector:
    
    # Parallel data collection
    dragon_tiger_data = await collector.dragon_tiger_collector.collect_data(...)
    shareholder_data = await collector.shareholder_collector.collect_data(...)
    block_trade_data = await collector.block_trade_collector.collect_data(...)
```

### Caching Strategy

- **Institution Classification**: Results cached by name
- **Pattern Matching**: Compiled regex patterns
- **Activity Timeline**: Cached per stock code

### Memory Management

- **Streaming Processing**: Large datasets processed in chunks
- **Lazy Loading**: Data loaded on demand
- **Garbage Collection**: Explicit cleanup of large objects

## Integration Points

### Data Source Integration

In production, replace mock collectors with real data sources:

```python
class TushareDataCollector(DataCollector):
    """Real Tushare API integration"""
    
    async def collect_data(self, stock_code: str, start_date: date, end_date: date):
        # Actual API calls to Tushare
        import tushare as ts
        
        pro = ts.pro_api('your_token')
        df = pro.dragon_tiger(ts_code=stock_code, 
                             start_date=start_date.strftime('%Y%m%d'),
                             end_date=end_date.strftime('%Y%m%d'))
        
        # Convert to DragonTigerRecord objects
        return self._convert_to_records(df)
```

### Database Integration

Store collected data in database:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Store activities in database
def store_activities(activities: List[InstitutionalActivity]):
    engine = create_engine('postgresql://...')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    for activity in activities:
        # Convert to database model and save
        db_activity = ActivityModel(
            activity_id=activity.activity_id,
            activity_date=activity.activity_date,
            stock_code=activity.stock_code,
            # ... other fields
        )
        session.add(db_activity)
    
    session.commit()
    session.close()
```

### Analysis Integration

Connect with other analysis engines:

```python
# Integration with Spring Festival Engine
from stock_analysis_system.analysis.spring_festival_engine import SpringFestivalEngine

sf_engine = SpringFestivalEngine()
institutional_collector = InstitutionalDataCollector()

# Analyze institutional activity around Spring Festival
activities = institutional_collector.get_institution_activity_timeline("000001")
sf_analysis = await sf_engine.analyze_seasonal_patterns(activities)
```

## Testing

### Run Tests

```bash
python -m pytest tests/test_institutional_data_collector.py -v
```

### Run Demo

```bash
python test_institutional_data_demo.py
```

### Test Coverage

The test suite covers:

- Institution classification accuracy
- Data collection functionality
- Activity consolidation logic
- Timeline generation
- Export functionality
- Error handling

## Production Deployment

### Real Data Source Integration

1. **Tushare Integration**
   ```python
   # Replace mock collectors with Tushare API calls
   import tushare as ts
   pro = ts.pro_api('your_token')
   ```

2. **Wind Integration**
   ```python
   # Wind API integration
   from WindPy import w
   w.start()
   ```

3. **AkShare Integration**
   ```python
   # AkShare integration
   import akshare as ak
   ```

### Database Schema

```sql
-- Institutional investors table
CREATE TABLE institutional_investors (
    institution_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    institution_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
);

-- Activities table
CREATE TABLE institutional_activities (
    activity_id VARCHAR(100) PRIMARY KEY,
    activity_date DATE NOT NULL,
    stock_code VARCHAR(10) NOT NULL,
    institution_id VARCHAR(50) REFERENCES institutional_investors(institution_id),
    activity_type VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2),
    volume BIGINT,
    price DECIMAL(10,2),
    source_type VARCHAR(50),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_activities_stock_date ON institutional_activities(stock_code, activity_date);
CREATE INDEX idx_activities_institution ON institutional_activities(institution_id);
CREATE INDEX idx_activities_type ON institutional_activities(activity_type);
```

### Monitoring and Alerting

```python
# Data quality monitoring
def monitor_data_quality(collector: InstitutionalDataCollector):
    for stock_code in stock_codes:
        summary = collector.get_institution_summary(stock_code)
        
        # Alert on low data quality
        if summary.get('total_activities', 0) < expected_minimum:
            send_alert(f"Low activity count for {stock_code}")
        
        # Alert on classification issues
        avg_confidence = summary.get('avg_confidence', 0)
        if avg_confidence < 0.7:
            send_alert(f"Low classification confidence for {stock_code}")
```

## Requirements Addressed

This implementation addresses the following requirements:

- **3.1**: Dragon-tiger list data collection and parsing
- **3.2**: Shareholder data collection and standardization
- **3.3**: Block trade data collection and analysis
- **3.4**: Institutional classification and activity timeline tracking

## Future Enhancements

1. **Advanced Classification**
   - Machine learning-based classification
   - Natural language processing for name matching
   - Cross-reference with regulatory databases

2. **Real-time Processing**
   - Streaming data ingestion
   - Real-time activity detection
   - Live institutional monitoring

3. **Enhanced Analytics**
   - Institution behavior modeling
   - Coordination detection algorithms
   - Market impact analysis

4. **Data Quality**
   - Automated data validation
   - Anomaly detection
   - Data lineage tracking

## Dependencies

- asyncio (built-in)
- aiohttp >= 3.8.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- dataclasses (built-in)
- datetime (built-in)
- re (built-in)
- logging (built-in)