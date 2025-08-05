"""
Institutional Data Collector

This module implements comprehensive institutional data collection including dragon-tiger list,
shareholder data, block trades, and institutional classification for the stock analysis system.

Requirements addressed: 3.1, 3.2, 3.3, 3.4
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InstitutionType(str, Enum):
    """Types of institutional investors"""

    MUTUAL_FUND = "mutual_fund"
    SOCIAL_SECURITY = "social_security"
    QFII = "qfii"  # Qualified Foreign Institutional Investor
    RQFII = "rqfii"  # RMB Qualified Foreign Institutional Investor
    HOT_MONEY = "hot_money"
    INSURANCE = "insurance"
    PENSION_FUND = "pension_fund"
    PRIVATE_EQUITY = "private_equity"
    HEDGE_FUND = "hedge_fund"
    BANK = "bank"
    SECURITIES_FIRM = "securities_firm"
    TRUST = "trust"
    OTHER = "other"


class ActivityType(str, Enum):
    """Types of institutional activities"""

    DRAGON_TIGER_BUY = "dragon_tiger_buy"
    DRAGON_TIGER_SELL = "dragon_tiger_sell"
    BLOCK_TRADE_BUY = "block_trade_buy"
    BLOCK_TRADE_SELL = "block_trade_sell"
    SHAREHOLDING_INCREASE = "shareholding_increase"
    SHAREHOLDING_DECREASE = "shareholding_decrease"
    NEW_POSITION = "new_position"
    POSITION_EXIT = "position_exit"


@dataclass
class InstitutionalInvestor:
    """Institutional investor information"""

    institution_id: str
    name: str
    institution_type: InstitutionType

    # Classification details
    parent_company: Optional[str] = None
    fund_manager: Optional[str] = None
    registration_country: Optional[str] = None
    aum: Optional[float] = None  # Assets Under Management

    # Identification patterns
    name_patterns: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    # Metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence_score: float = 1.0  # Classification confidence

    def __post_init__(self):
        if not self.name_patterns:
            self.name_patterns = [self.name]


@dataclass
class DragonTigerRecord:
    """Dragon-Tiger list record"""

    trade_date: date
    stock_code: str
    stock_name: str

    # Trading seat information
    seat_name: str
    seat_type: str  # "buy" or "sell"
    amount: float  # Trading amount in yuan
    net_amount: Optional[float] = None  # Net amount if available

    # Institution identification
    institution: Optional[InstitutionalInvestor] = None
    institution_confidence: float = 0.0

    # Additional metadata
    rank: Optional[int] = None  # Rank in the list
    market: Optional[str] = None  # Market (SH/SZ)
    reason: Optional[str] = None  # Reason for inclusion in list


@dataclass
class ShareholderRecord:
    """Shareholder record"""

    report_date: date
    stock_code: str
    stock_name: str

    # Shareholder information
    shareholder_name: str
    shareholding_ratio: float  # Percentage
    shares_held: Optional[int] = None
    shares_change: Optional[int] = None

    # Institution identification
    institution: Optional[InstitutionalInvestor] = None
    institution_confidence: float = 0.0

    # Ranking and metadata
    rank: Optional[int] = None  # Rank among shareholders
    shareholder_type: Optional[str] = None  # Type classification
    is_restricted: bool = False  # Restricted shares


@dataclass
class BlockTradeRecord:
    """Block trade record"""

    trade_date: date
    stock_code: str
    stock_name: str

    # Trade details
    volume: int  # Number of shares
    price: float  # Trade price
    total_amount: float  # Total transaction amount

    # Counterparty information
    buyer_seat: Optional[str] = None
    seller_seat: Optional[str] = None
    buyer_institution: Optional[InstitutionalInvestor] = None
    seller_institution: Optional[InstitutionalInvestor] = None

    # Trade characteristics
    discount_rate: Optional[float] = None  # Discount to market price
    premium_rate: Optional[float] = None  # Premium to market price
    market_price: Optional[float] = None  # Market price at trade time


@dataclass
class InstitutionalActivity:
    """Consolidated institutional activity record"""

    activity_id: str
    activity_date: date
    stock_code: str
    institution: InstitutionalInvestor
    activity_type: ActivityType

    # Activity details
    amount: Optional[float] = None  # Monetary amount
    volume: Optional[int] = None  # Share volume
    price: Optional[float] = None  # Price per share

    # Context information
    market_cap: Optional[float] = None
    daily_volume: Optional[int] = None
    price_impact: Optional[float] = None

    # Source tracking
    source_type: str = "unknown"  # dragon_tiger, shareholder, block_trade
    source_record_id: Optional[str] = None
    confidence_score: float = 1.0


class InstitutionClassifier:
    """Classifier for identifying institutional investor types"""

    def __init__(self):
        self.classification_rules = self._load_classification_rules()
        self.known_institutions = {}  # Cache for known institutions

    def _load_classification_rules(self) -> Dict[InstitutionType, List[Dict]]:
        """Load classification rules for different institution types"""

        return {
            InstitutionType.MUTUAL_FUND: [
                {"pattern": r".*基金.*", "confidence": 0.9},
                {"pattern": r".*fund.*", "confidence": 0.8, "case_sensitive": False},
                {"pattern": r".*资产管理.*", "confidence": 0.7},
                {
                    "pattern": r".*asset.*management.*",
                    "confidence": 0.7,
                    "case_sensitive": False,
                },
            ],
            InstitutionType.SOCIAL_SECURITY: [
                {"pattern": r".*社保.*", "confidence": 0.95},
                {"pattern": r".*社会保障.*", "confidence": 0.95},
                {"pattern": r".*全国社会保障基金.*", "confidence": 1.0},
            ],
            InstitutionType.QFII: [
                {"pattern": r".*QFII.*", "confidence": 0.9},
                {"pattern": r".*合格境外机构投资者.*", "confidence": 0.9},
                {"pattern": r".*摩根.*", "confidence": 0.6},
                {"pattern": r".*高盛.*", "confidence": 0.6},
                {"pattern": r".*瑞银.*", "confidence": 0.6},
            ],
            InstitutionType.RQFII: [
                {"pattern": r".*RQFII.*", "confidence": 0.9},
                {"pattern": r".*人民币合格境外机构投资者.*", "confidence": 0.9},
            ],
            InstitutionType.INSURANCE: [
                {"pattern": r".*保险.*", "confidence": 0.8},
                {
                    "pattern": r".*insurance.*",
                    "confidence": 0.8,
                    "case_sensitive": False,
                },
                {"pattern": r".*人寿.*", "confidence": 0.7},
                {"pattern": r".*平安.*", "confidence": 0.6},
                {"pattern": r".*太平洋.*", "confidence": 0.6},
            ],
            InstitutionType.SECURITIES_FIRM: [
                {"pattern": r".*证券.*", "confidence": 0.8},
                {
                    "pattern": r".*securities.*",
                    "confidence": 0.8,
                    "case_sensitive": False,
                },
                {"pattern": r".*中信.*", "confidence": 0.6},
                {"pattern": r".*华泰.*", "confidence": 0.6},
                {"pattern": r".*国泰君安.*", "confidence": 0.8},
            ],
            InstitutionType.BANK: [
                {"pattern": r".*银行.*", "confidence": 0.8},
                {"pattern": r".*bank.*", "confidence": 0.8, "case_sensitive": False},
                {"pattern": r".*工商银行.*", "confidence": 0.9},
                {"pattern": r".*建设银行.*", "confidence": 0.9},
                {"pattern": r".*农业银行.*", "confidence": 0.9},
            ],
            InstitutionType.HOT_MONEY: [
                {"pattern": r".*游资.*", "confidence": 0.9},
                {"pattern": r".*热钱.*", "confidence": 0.9},
                {"pattern": r".*营业部.*", "confidence": 0.6},
                {"pattern": r".*个人.*", "confidence": 0.5},
            ],
        }

    def classify_institution(
        self, name: str, additional_info: Optional[Dict] = None
    ) -> Tuple[InstitutionType, float]:
        """
        Classify an institution based on its name and additional information.

        Args:
            name: Institution name
            additional_info: Additional information for classification

        Returns:
            Tuple of (institution_type, confidence_score)
        """

        # Check cache first
        if name in self.known_institutions:
            cached = self.known_institutions[name]
            return cached["type"], cached["confidence"]

        best_type = InstitutionType.OTHER
        best_confidence = 0.0

        # Apply classification rules
        for institution_type, rules in self.classification_rules.items():
            for rule in rules:
                pattern = rule["pattern"]
                confidence = rule["confidence"]
                case_sensitive = rule.get("case_sensitive", True)

                flags = 0 if case_sensitive else re.IGNORECASE

                if re.search(pattern, name, flags):
                    if confidence > best_confidence:
                        best_type = institution_type
                        best_confidence = confidence

        # Use additional information if available
        if additional_info:
            # Enhance classification based on additional info
            if "fund_type" in additional_info:
                if additional_info["fund_type"] in ["公募基金", "私募基金"]:
                    if best_confidence < 0.8:
                        best_type = InstitutionType.MUTUAL_FUND
                        best_confidence = 0.8

        # Cache the result
        self.known_institutions[name] = {
            "type": best_type,
            "confidence": best_confidence,
        }

        return best_type, best_confidence

    def create_institution(
        self, name: str, additional_info: Optional[Dict] = None
    ) -> InstitutionalInvestor:
        """Create an InstitutionalInvestor object with classification."""

        institution_type, confidence = self.classify_institution(name, additional_info)

        # Generate unique ID
        institution_id = f"{institution_type.value}_{hash(name) % 1000000:06d}"

        return InstitutionalInvestor(
            institution_id=institution_id,
            name=name,
            institution_type=institution_type,
            confidence_score=confidence,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )


class DataCollector(ABC):
    """Abstract base class for data collectors"""

    def __init__(self, classifier: InstitutionClassifier):
        self.classifier = classifier
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def collect_data(
        self, stock_code: str, start_date: date, end_date: date
    ) -> List[Any]:
        """Collect data for a specific stock and date range"""
        pass


class DragonTigerCollector(DataCollector):
    """Collector for Dragon-Tiger list data"""

    async def collect_data(
        self, stock_code: str, start_date: date, end_date: date
    ) -> List[DragonTigerRecord]:
        """
        Collect Dragon-Tiger list data.

        Note: This is a mock implementation. In production, this would integrate
        with actual data sources like Tushare, Wind, or other financial data APIs.
        """

        records = []

        try:
            # Mock data generation for demonstration
            # In production, replace with actual API calls
            current_date = start_date

            while current_date <= end_date:
                # Simulate some Dragon-Tiger records
                if np.random.random() < 0.1:  # 10% chance of Dragon-Tiger activity

                    # Generate mock buy records
                    for i in range(np.random.randint(1, 6)):  # 1-5 buy records
                        seat_name = self._generate_mock_seat_name("buy")
                        amount = np.random.uniform(
                            10000000, 500000000
                        )  # 10M to 500M yuan

                        # Classify institution
                        institution = self.classifier.create_institution(seat_name)

                        record = DragonTigerRecord(
                            trade_date=current_date,
                            stock_code=stock_code,
                            stock_name=f"Stock_{stock_code}",
                            seat_name=seat_name,
                            seat_type="buy",
                            amount=amount,
                            institution=institution,
                            institution_confidence=institution.confidence_score,
                            rank=i + 1,
                            market="SZ" if stock_code.startswith("0") else "SH",
                        )
                        records.append(record)

                    # Generate mock sell records
                    for i in range(np.random.randint(1, 6)):  # 1-5 sell records
                        seat_name = self._generate_mock_seat_name("sell")
                        amount = np.random.uniform(10000000, 500000000)

                        institution = self.classifier.create_institution(seat_name)

                        record = DragonTigerRecord(
                            trade_date=current_date,
                            stock_code=stock_code,
                            stock_name=f"Stock_{stock_code}",
                            seat_name=seat_name,
                            seat_type="sell",
                            amount=amount,
                            institution=institution,
                            institution_confidence=institution.confidence_score,
                            rank=i + 1,
                            market="SZ" if stock_code.startswith("0") else "SH",
                        )
                        records.append(record)

                current_date += timedelta(days=1)

            logger.info(
                f"Collected {len(records)} Dragon-Tiger records for {stock_code}"
            )
            return records

        except Exception as e:
            logger.error(f"Error collecting Dragon-Tiger data for {stock_code}: {e}")
            return []

    def _generate_mock_seat_name(self, trade_type: str) -> str:
        """Generate mock trading seat names for demonstration"""

        institution_templates = [
            "中信证券股份有限公司{}营业部",
            "华泰证券股份有限公司{}营业部",
            "国泰君安证券股份有限公司{}营业部",
            "招商证券股份有限公司{}营业部",
            "广发证券股份有限公司{}营业部",
            "申万宏源证券有限公司{}营业部",
            "海通证券股份有限公司{}营业部",
            "东方财富证券股份有限公司{}营业部",
        ]

        fund_templates = [
            "易方达基金管理有限公司",
            "华夏基金管理有限公司",
            "嘉实基金管理有限公司",
            "南方基金管理股份有限公司",
            "博时基金管理有限公司",
            "广发基金管理有限公司",
            "工银瑞信基金管理有限公司",
            "建信基金管理有限责任公司",
        ]

        cities = ["北京", "上海", "深圳", "广州", "杭州", "南京", "成都", "武汉"]

        if np.random.random() < 0.7:  # 70% chance of securities firm
            template = np.random.choice(institution_templates)
            city = np.random.choice(cities)
            return template.format(city)
        else:  # 30% chance of fund
            return np.random.choice(fund_templates)


class ShareholderCollector(DataCollector):
    """Collector for shareholder data"""

    async def collect_data(
        self, stock_code: str, start_date: date, end_date: date
    ) -> List[ShareholderRecord]:
        """
        Collect shareholder data.

        Note: This is a mock implementation for demonstration.
        """

        records = []

        try:
            # Generate quarterly shareholder data
            current_date = start_date

            while current_date <= end_date:
                # Check if it's a quarter end (March, June, September, December)
                if current_date.month in [3, 6, 9, 12] and current_date.day >= 28:

                    # Generate top 10 shareholders
                    total_shares = 1000000000  # 1 billion shares
                    remaining_shares = total_shares

                    for rank in range(1, 11):  # Top 10 shareholders
                        if rank == 1:
                            # Largest shareholder (usually controlling shareholder)
                            shares = int(total_shares * np.random.uniform(0.15, 0.35))
                            shareholder_name = f"{stock_code}控股集团有限公司"
                        elif rank <= 3:
                            # Major institutional shareholders
                            shares = int(
                                remaining_shares * np.random.uniform(0.05, 0.15)
                            )
                            shareholder_name = self._generate_mock_institutional_name()
                        else:
                            # Smaller institutional shareholders
                            shares = int(
                                remaining_shares * np.random.uniform(0.01, 0.08)
                            )
                            shareholder_name = self._generate_mock_institutional_name()

                        shareholding_ratio = (shares / total_shares) * 100
                        remaining_shares -= shares

                        # Classify institution
                        institution = self.classifier.create_institution(
                            shareholder_name
                        )

                        record = ShareholderRecord(
                            report_date=current_date,
                            stock_code=stock_code,
                            stock_name=f"Stock_{stock_code}",
                            shareholder_name=shareholder_name,
                            shareholding_ratio=shareholding_ratio,
                            shares_held=shares,
                            institution=institution,
                            institution_confidence=institution.confidence_score,
                            rank=rank,
                            shareholder_type=(
                                "institutional" if rank > 1 else "controlling"
                            ),
                        )
                        records.append(record)

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(
                        year=current_date.year + 1, month=1
                    )
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

            logger.info(
                f"Collected {len(records)} shareholder records for {stock_code}"
            )
            return records

        except Exception as e:
            logger.error(f"Error collecting shareholder data for {stock_code}: {e}")
            return []

    def _generate_mock_institutional_name(self) -> str:
        """Generate mock institutional shareholder names"""

        templates = [
            "全国社会保障基金理事会",
            "中央汇金资产管理有限责任公司",
            "香港中央结算有限公司",
            "中国人寿保险股份有限公司",
            "中国平安人寿保险股份有限公司",
            "易方达基金管理有限公司",
            "华夏基金管理有限公司",
            "嘉实基金管理有限公司",
            "南方基金管理股份有限公司",
            "博时基金管理有限公司",
            "挪威中央银行",
            "新加坡政府投资公司",
            "淡马锡控股私人有限公司",
            "摩根士丹利投资管理公司",
            "贝莱德资产管理公司",
        ]

        return np.random.choice(templates)


class BlockTradeCollector(DataCollector):
    """Collector for block trade data"""

    async def collect_data(
        self, stock_code: str, start_date: date, end_date: date
    ) -> List[BlockTradeRecord]:
        """
        Collect block trade data.

        Note: This is a mock implementation for demonstration.
        """

        records = []

        try:
            current_date = start_date

            while current_date <= end_date:
                # Simulate block trades (less frequent than Dragon-Tiger)
                if np.random.random() < 0.05:  # 5% chance of block trade

                    # Generate 1-3 block trades per day
                    for _ in range(np.random.randint(1, 4)):
                        volume = np.random.randint(
                            100000, 10000000
                        )  # 100K to 10M shares
                        price = np.random.uniform(10, 100)  # Price between 10-100 yuan
                        total_amount = volume * price

                        # Generate buyer and seller information
                        buyer_seat = self._generate_mock_trading_seat()
                        seller_seat = self._generate_mock_trading_seat()

                        buyer_institution = self.classifier.create_institution(
                            buyer_seat
                        )
                        seller_institution = self.classifier.create_institution(
                            seller_seat
                        )

                        # Calculate discount/premium
                        market_price = price * np.random.uniform(
                            0.98, 1.02
                        )  # Market price
                        if price < market_price:
                            discount_rate = (market_price - price) / market_price
                            premium_rate = None
                        else:
                            discount_rate = None
                            premium_rate = (price - market_price) / market_price

                        record = BlockTradeRecord(
                            trade_date=current_date,
                            stock_code=stock_code,
                            stock_name=f"Stock_{stock_code}",
                            volume=volume,
                            price=price,
                            total_amount=total_amount,
                            buyer_seat=buyer_seat,
                            seller_seat=seller_seat,
                            buyer_institution=buyer_institution,
                            seller_institution=seller_institution,
                            discount_rate=discount_rate,
                            premium_rate=premium_rate,
                            market_price=market_price,
                        )
                        records.append(record)

                current_date += timedelta(days=1)

            logger.info(
                f"Collected {len(records)} block trade records for {stock_code}"
            )
            return records

        except Exception as e:
            logger.error(f"Error collecting block trade data for {stock_code}: {e}")
            return []

    def _generate_mock_trading_seat(self) -> str:
        """Generate mock trading seat names"""

        seats = [
            "机构专用",
            "中信证券股份有限公司总部",
            "华泰证券股份有限公司总部",
            "国泰君安证券股份有限公司总部",
            "招商证券股份有限公司总部",
            "广发证券股份有限公司总部",
            "申万宏源证券有限公司总部",
            "海通证券股份有限公司总部",
            "东方财富证券股份有限公司总部",
            "易方达基金管理有限公司",
            "华夏基金管理有限公司",
            "嘉实基金管理有限公司",
            "南方基金管理股份有限公司",
        ]

        return np.random.choice(seats)


class InstitutionalDataCollector:
    """
    Main institutional data collector that coordinates all data collection activities
    """

    def __init__(self):
        self.classifier = InstitutionClassifier()
        self.dragon_tiger_collector = DragonTigerCollector(self.classifier)
        self.shareholder_collector = ShareholderCollector(self.classifier)
        self.block_trade_collector = BlockTradeCollector(self.classifier)

        # Activity timeline storage
        self.activity_timeline = {}  # stock_code -> List[InstitutionalActivity]

    async def collect_all_data(
        self, stock_codes: List[str], start_date: date, end_date: date
    ) -> Dict[str, Dict[str, List]]:
        """
        Collect all institutional data for given stocks and date range.

        Args:
            stock_codes: List of stock codes to collect data for
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            Dictionary with collected data organized by stock code and data type
        """

        all_data = {}

        try:
            # Use context managers for proper session handling
            async with (
                self.dragon_tiger_collector,
                self.shareholder_collector,
                self.block_trade_collector,
            ):

                for stock_code in stock_codes:
                    logger.info(f"Collecting institutional data for {stock_code}")

                    # Collect data from all sources
                    dragon_tiger_data = await self.dragon_tiger_collector.collect_data(
                        stock_code, start_date, end_date
                    )

                    shareholder_data = await self.shareholder_collector.collect_data(
                        stock_code, start_date, end_date
                    )

                    block_trade_data = await self.block_trade_collector.collect_data(
                        stock_code, start_date, end_date
                    )

                    # Store collected data
                    all_data[stock_code] = {
                        "dragon_tiger": dragon_tiger_data,
                        "shareholders": shareholder_data,
                        "block_trades": block_trade_data,
                    }

                    # Generate consolidated activity timeline
                    activities = self._consolidate_activities(
                        stock_code,
                        dragon_tiger_data,
                        shareholder_data,
                        block_trade_data,
                    )
                    self.activity_timeline[stock_code] = activities

                    logger.info(
                        f"Collected {len(dragon_tiger_data)} Dragon-Tiger, "
                        f"{len(shareholder_data)} shareholder, "
                        f"{len(block_trade_data)} block trade records for {stock_code}"
                    )

            return all_data

        except Exception as e:
            logger.error(f"Error in institutional data collection: {e}")
            raise

    def _consolidate_activities(
        self,
        stock_code: str,
        dragon_tiger_data: List[DragonTigerRecord],
        shareholder_data: List[ShareholderRecord],
        block_trade_data: List[BlockTradeRecord],
    ) -> List[InstitutionalActivity]:
        """Consolidate all data sources into unified activity timeline"""

        activities = []

        # Process Dragon-Tiger data
        for record in dragon_tiger_data:
            if record.institution:
                activity_type = (
                    ActivityType.DRAGON_TIGER_BUY
                    if record.seat_type == "buy"
                    else ActivityType.DRAGON_TIGER_SELL
                )

                activity = InstitutionalActivity(
                    activity_id=f"dt_{stock_code}_{record.trade_date}_{record.rank}_{record.seat_type}",
                    activity_date=record.trade_date,
                    stock_code=stock_code,
                    institution=record.institution,
                    activity_type=activity_type,
                    amount=record.amount,
                    source_type="dragon_tiger",
                    confidence_score=record.institution_confidence,
                )
                activities.append(activity)

        # Process shareholder data
        for record in shareholder_data:
            if record.institution and record.shares_change:
                activity_type = (
                    ActivityType.SHAREHOLDING_INCREASE
                    if record.shares_change > 0
                    else ActivityType.SHAREHOLDING_DECREASE
                )

                activity = InstitutionalActivity(
                    activity_id=f"sh_{stock_code}_{record.report_date}_{record.rank}",
                    activity_date=record.report_date,
                    stock_code=stock_code,
                    institution=record.institution,
                    activity_type=activity_type,
                    volume=abs(record.shares_change),
                    source_type="shareholder",
                    confidence_score=record.institution_confidence,
                )
                activities.append(activity)

        # Process block trade data
        for record in block_trade_data:
            # Buyer activity
            if record.buyer_institution:
                activity = InstitutionalActivity(
                    activity_id=f"bt_buy_{stock_code}_{record.trade_date}_{hash(record.buyer_seat) % 10000}",
                    activity_date=record.trade_date,
                    stock_code=stock_code,
                    institution=record.buyer_institution,
                    activity_type=ActivityType.BLOCK_TRADE_BUY,
                    amount=record.total_amount,
                    volume=record.volume,
                    price=record.price,
                    source_type="block_trade",
                )
                activities.append(activity)

            # Seller activity
            if record.seller_institution:
                activity = InstitutionalActivity(
                    activity_id=f"bt_sell_{stock_code}_{record.trade_date}_{hash(record.seller_seat) % 10000}",
                    activity_date=record.trade_date,
                    stock_code=stock_code,
                    institution=record.seller_institution,
                    activity_type=ActivityType.BLOCK_TRADE_SELL,
                    amount=record.total_amount,
                    volume=record.volume,
                    price=record.price,
                    source_type="block_trade",
                )
                activities.append(activity)

        # Sort activities by date
        activities.sort(key=lambda x: x.activity_date)

        return activities

    def get_institution_activity_timeline(
        self, stock_code: str, institution_type: Optional[InstitutionType] = None
    ) -> List[InstitutionalActivity]:
        """Get activity timeline for a specific stock, optionally filtered by institution type"""

        if stock_code not in self.activity_timeline:
            return []

        activities = self.activity_timeline[stock_code]

        if institution_type:
            activities = [
                a
                for a in activities
                if a.institution.institution_type == institution_type
            ]

        return activities

    def get_institution_summary(self, stock_code: str) -> Dict[str, Any]:
        """Get summary statistics of institutional activity for a stock"""

        if stock_code not in self.activity_timeline:
            return {}

        activities = self.activity_timeline[stock_code]

        # Count activities by institution type
        type_counts = {}
        type_amounts = {}

        for activity in activities:
            inst_type = activity.institution.institution_type.value

            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1

            if activity.amount:
                type_amounts[inst_type] = (
                    type_amounts.get(inst_type, 0) + activity.amount
                )

        # Calculate summary metrics
        total_activities = len(activities)
        total_amount = sum(a.amount for a in activities if a.amount)
        unique_institutions = len(set(a.institution.institution_id for a in activities))

        # Recent activity (last 30 days)
        recent_date = max(a.activity_date for a in activities) if activities else None
        if recent_date:
            cutoff_date = recent_date - timedelta(days=30)
            recent_activities = [
                a for a in activities if a.activity_date >= cutoff_date
            ]
        else:
            recent_activities = []

        return {
            "total_activities": total_activities,
            "total_amount": total_amount,
            "unique_institutions": unique_institutions,
            "institution_type_counts": type_counts,
            "institution_type_amounts": type_amounts,
            "recent_activities_30d": len(recent_activities),
            "date_range": {
                "start": (
                    min(a.activity_date for a in activities) if activities else None
                ),
                "end": max(a.activity_date for a in activities) if activities else None,
            },
        }

    def export_data_to_dataframes(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        """Export collected data to pandas DataFrames for analysis"""

        if stock_code not in self.activity_timeline:
            return {}

        activities = self.activity_timeline[stock_code]

        # Convert to DataFrame
        activity_data = []
        for activity in activities:
            activity_data.append(
                {
                    "activity_id": activity.activity_id,
                    "activity_date": activity.activity_date,
                    "stock_code": activity.stock_code,
                    "institution_id": activity.institution.institution_id,
                    "institution_name": activity.institution.name,
                    "institution_type": activity.institution.institution_type.value,
                    "activity_type": activity.activity_type.value,
                    "amount": activity.amount,
                    "volume": activity.volume,
                    "price": activity.price,
                    "source_type": activity.source_type,
                    "confidence_score": activity.confidence_score,
                }
            )

        df = pd.DataFrame(activity_data)

        return {
            "activities": df,
            "summary": pd.DataFrame([self.get_institution_summary(stock_code)]),
        }
