"""
Unit tests for Institutional Data Collector

Tests dragon-tiger list collection, shareholder data collection, block trades,
and institutional classification as specified in task 6.1.
"""

import pytest
import asyncio
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from stock_analysis_system.analysis.institutional_data_collector import (
    InstitutionalDataCollector,
    InstitutionClassifier,
    DragonTigerCollector,
    ShareholderCollector,
    BlockTradeCollector,
    InstitutionType,
    ActivityType,
    InstitutionalInvestor,
    DragonTigerRecord,
    ShareholderRecord,
    BlockTradeRecord,
    InstitutionalActivity
)


class TestInstitutionClassifier:
    """Test cases for Institution Classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance for testing."""
        return InstitutionClassifier()
    
    def test_classify_mutual_fund(self, classifier):
        """Test classification of mutual funds."""
        
        test_cases = [
            ("易方达基金管理有限公司", InstitutionType.MUTUAL_FUND),
            ("华夏基金管理有限公司", InstitutionType.MUTUAL_FUND),
            ("嘉实基金管理有限公司", InstitutionType.MUTUAL_FUND),
            ("某某资产管理有限公司", InstitutionType.MUTUAL_FUND),
            ("ABC Fund Management", InstitutionType.MUTUAL_FUND)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5  # Should have reasonable confidence
    
    def test_classify_social_security(self, classifier):
        """Test classification of social security funds."""
        
        test_cases = [
            ("全国社会保障基金理事会", InstitutionType.SOCIAL_SECURITY),
            ("社保基金", InstitutionType.SOCIAL_SECURITY),
            ("社会保障基金", InstitutionType.SOCIAL_SECURITY)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence >= 0.9  # Should have high confidence
    
    def test_classify_qfii(self, classifier):
        """Test classification of QFII institutions."""
        
        test_cases = [
            ("某某QFII", InstitutionType.QFII),
            ("合格境外机构投资者", InstitutionType.QFII),
            ("摩根士丹利", InstitutionType.QFII),
            ("高盛集团", InstitutionType.QFII)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5
    
    def test_classify_insurance(self, classifier):
        """Test classification of insurance companies."""
        
        test_cases = [
            ("中国人寿保险股份有限公司", InstitutionType.INSURANCE),
            ("中国平安人寿保险股份有限公司", InstitutionType.INSURANCE),
            ("某某保险公司", InstitutionType.INSURANCE),
            ("ABC Insurance Company", InstitutionType.INSURANCE)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5
    
    def test_classify_securities_firm(self, classifier):
        """Test classification of securities firms."""
        
        test_cases = [
            ("中信证券股份有限公司", InstitutionType.SECURITIES_FIRM),
            ("华泰证券股份有限公司", InstitutionType.SECURITIES_FIRM),
            ("国泰君安证券股份有限公司", InstitutionType.SECURITIES_FIRM),
            ("某某证券公司", InstitutionType.SECURITIES_FIRM)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5
    
    def test_classify_bank(self, classifier):
        """Test classification of banks."""
        
        test_cases = [
            ("中国工商银行股份有限公司", InstitutionType.BANK),
            ("中国建设银行股份有限公司", InstitutionType.BANK),
            ("某某银行", InstitutionType.BANK),
            ("ABC Bank", InstitutionType.BANK)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5
    
    def test_classify_hot_money(self, classifier):
        """Test classification of hot money."""
        
        test_cases = [
            ("某某游资", InstitutionType.HOT_MONEY),
            ("热钱机构", InstitutionType.HOT_MONEY),
            ("某某营业部", InstitutionType.HOT_MONEY)
        ]
        
        for name, expected_type in test_cases:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == expected_type
            assert confidence > 0.5
    
    def test_classify_unknown(self, classifier):
        """Test classification of unknown institutions."""
        
        unknown_names = [
            "Unknown Institution",
            "随机公司名称",
            "XYZ Corporation"
        ]
        
        for name in unknown_names:
            institution_type, confidence = classifier.classify_institution(name)
            assert institution_type == InstitutionType.OTHER
            assert confidence == 0.0
    
    def test_create_institution(self, classifier):
        """Test creation of InstitutionalInvestor objects."""
        
        name = "易方达基金管理有限公司"
        institution = classifier.create_institution(name)
        
        assert isinstance(institution, InstitutionalInvestor)
        assert institution.name == name
        assert institution.institution_type == InstitutionType.MUTUAL_FUND
        assert institution.confidence_score > 0.5
        assert institution.institution_id.startswith("mutual_fund_")
        assert isinstance(institution.first_seen, datetime)
        assert isinstance(institution.last_seen, datetime)
    
    def test_caching(self, classifier):
        """Test that classification results are cached."""
        
        name = "华夏基金管理有限公司"
        
        # First classification
        type1, conf1 = classifier.classify_institution(name)
        
        # Second classification (should use cache)
        type2, conf2 = classifier.classify_institution(name)
        
        assert type1 == type2
        assert conf1 == conf2
        assert name in classifier.known_institutions


class TestDragonTigerCollector:
    """Test cases for Dragon-Tiger Collector"""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance for testing."""
        classifier = InstitutionClassifier()
        return DragonTigerCollector(classifier)
    
    @pytest.mark.asyncio
    async def test_collect_data(self, collector):
        """Test Dragon-Tiger data collection."""
        
        stock_code = "000001"
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        async with collector:
            records = await collector.collect_data(stock_code, start_date, end_date)
        
        # Verify records structure
        assert isinstance(records, list)
        
        for record in records:
            assert isinstance(record, DragonTigerRecord)
            assert record.stock_code == stock_code
            assert start_date <= record.trade_date <= end_date
            assert record.seat_type in ["buy", "sell"]
            assert record.amount > 0
            assert isinstance(record.institution, InstitutionalInvestor)
            assert 0 <= record.institution_confidence <= 1.0
            assert record.market in ["SH", "SZ"]
    
    def test_generate_mock_seat_name(self, collector):
        """Test mock seat name generation."""
        
        # Test multiple generations
        for _ in range(10):
            seat_name = collector._generate_mock_seat_name("buy")
            assert isinstance(seat_name, str)
            assert len(seat_name) > 0
            
            # Should contain either "营业部" or "基金"
            assert "营业部" in seat_name or "基金" in seat_name


class TestShareholderCollector:
    """Test cases for Shareholder Collector"""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance for testing."""
        classifier = InstitutionClassifier()
        return ShareholderCollector(classifier)
    
    @pytest.mark.asyncio
    async def test_collect_data(self, collector):
        """Test shareholder data collection."""
        
        stock_code = "000001"
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        async with collector:
            records = await collector.collect_data(stock_code, start_date, end_date)
        
        # Verify records structure
        assert isinstance(records, list)
        
        # Should have quarterly data (4 quarters * 10 shareholders = 40 records)
        assert len(records) >= 40
        
        for record in records:
            assert isinstance(record, ShareholderRecord)
            assert record.stock_code == stock_code
            assert start_date <= record.report_date <= end_date
            assert 0 < record.shareholding_ratio <= 100
            assert record.shares_held > 0
            assert isinstance(record.institution, InstitutionalInvestor)
            assert 1 <= record.rank <= 10
    
    def test_generate_mock_institutional_name(self, collector):
        """Test mock institutional name generation."""
        
        for _ in range(10):
            name = collector._generate_mock_institutional_name()
            assert isinstance(name, str)
            assert len(name) > 0


class TestBlockTradeCollector:
    """Test cases for Block Trade Collector"""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance for testing."""
        classifier = InstitutionClassifier()
        return BlockTradeCollector(classifier)
    
    @pytest.mark.asyncio
    async def test_collect_data(self, collector):
        """Test block trade data collection."""
        
        stock_code = "000001"
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        async with collector:
            records = await collector.collect_data(stock_code, start_date, end_date)
        
        # Verify records structure
        assert isinstance(records, list)
        
        for record in records:
            assert isinstance(record, BlockTradeRecord)
            assert record.stock_code == stock_code
            assert start_date <= record.trade_date <= end_date
            assert record.volume > 0
            assert record.price > 0
            assert record.total_amount > 0
            assert record.buyer_seat is not None
            assert record.seller_seat is not None
            assert isinstance(record.buyer_institution, InstitutionalInvestor)
            assert isinstance(record.seller_institution, InstitutionalInvestor)
            
            # Either discount or premium should be set, not both
            if record.discount_rate is not None:
                assert record.premium_rate is None
                assert record.discount_rate >= 0
            elif record.premium_rate is not None:
                assert record.discount_rate is None
                assert record.premium_rate >= 0
    
    def test_generate_mock_trading_seat(self, collector):
        """Test mock trading seat generation."""
        
        for _ in range(10):
            seat = collector._generate_mock_trading_seat()
            assert isinstance(seat, str)
            assert len(seat) > 0


class TestInstitutionalDataCollector:
    """Test cases for main Institutional Data Collector"""
    
    @pytest.fixture
    def collector(self):
        """Create main collector instance for testing."""
        return InstitutionalDataCollector()
    
    @pytest.mark.asyncio
    async def test_collect_all_data(self, collector):
        """Test comprehensive data collection."""
        
        stock_codes = ["000001", "000002"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        all_data = await collector.collect_all_data(stock_codes, start_date, end_date)
        
        # Verify structure
        assert isinstance(all_data, dict)
        assert len(all_data) == len(stock_codes)
        
        for stock_code in stock_codes:
            assert stock_code in all_data
            stock_data = all_data[stock_code]
            
            # Check data types
            assert 'dragon_tiger' in stock_data
            assert 'shareholders' in stock_data
            assert 'block_trades' in stock_data
            
            # Verify data types
            assert isinstance(stock_data['dragon_tiger'], list)
            assert isinstance(stock_data['shareholders'], list)
            assert isinstance(stock_data['block_trades'], list)
            
            # Check activity timeline was created
            assert stock_code in collector.activity_timeline
            activities = collector.activity_timeline[stock_code]
            assert isinstance(activities, list)
            
            # Verify activities are sorted by date
            if len(activities) > 1:
                for i in range(1, len(activities)):
                    assert activities[i-1].activity_date <= activities[i].activity_date
    
    def test_consolidate_activities(self, collector):
        """Test activity consolidation."""
        
        # Create mock data
        classifier = InstitutionClassifier()
        
        # Mock Dragon-Tiger record
        dt_institution = classifier.create_institution("华夏基金管理有限公司")
        dt_record = DragonTigerRecord(
            trade_date=date(2023, 1, 15),
            stock_code="000001",
            stock_name="Test Stock",
            seat_name="华夏基金管理有限公司",
            seat_type="buy",
            amount=50000000,
            institution=dt_institution,
            institution_confidence=0.9,
            rank=1
        )
        
        # Mock shareholder record
        sh_institution = classifier.create_institution("易方达基金管理有限公司")
        sh_record = ShareholderRecord(
            report_date=date(2023, 3, 31),
            stock_code="000001",
            stock_name="Test Stock",
            shareholder_name="易方达基金管理有限公司",
            shareholding_ratio=5.5,
            shares_held=55000000,
            shares_change=5000000,
            institution=sh_institution,
            institution_confidence=0.9,
            rank=3
        )
        
        # Mock block trade record
        bt_buyer = classifier.create_institution("机构专用")
        bt_seller = classifier.create_institution("中信证券股份有限公司总部")
        bt_record = BlockTradeRecord(
            trade_date=date(2023, 2, 10),
            stock_code="000001",
            stock_name="Test Stock",
            volume=1000000,
            price=25.50,
            total_amount=25500000,
            buyer_seat="机构专用",
            seller_seat="中信证券股份有限公司总部",
            buyer_institution=bt_buyer,
            seller_institution=bt_seller
        )
        
        # Consolidate activities
        activities = collector._consolidate_activities(
            "000001", [dt_record], [sh_record], [bt_record]
        )
        
        # Should have 4 activities: 1 DT + 1 SH + 2 BT (buy + sell)
        assert len(activities) == 4
        
        # Verify activity types
        activity_types = [a.activity_type for a in activities]
        assert ActivityType.DRAGON_TIGER_BUY in activity_types
        assert ActivityType.SHAREHOLDING_INCREASE in activity_types
        assert ActivityType.BLOCK_TRADE_BUY in activity_types
        assert ActivityType.BLOCK_TRADE_SELL in activity_types
        
        # Verify activities are sorted by date
        dates = [a.activity_date for a in activities]
        assert dates == sorted(dates)
    
    def test_get_institution_activity_timeline(self, collector):
        """Test getting institution activity timeline."""
        
        # Setup mock activity timeline
        classifier = InstitutionClassifier()
        
        activities = [
            InstitutionalActivity(
                activity_id="test1",
                activity_date=date(2023, 1, 15),
                stock_code="000001",
                institution=classifier.create_institution("华夏基金管理有限公司"),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=50000000
            ),
            InstitutionalActivity(
                activity_id="test2",
                activity_date=date(2023, 1, 20),
                stock_code="000001",
                institution=classifier.create_institution("中信证券股份有限公司"),
                activity_type=ActivityType.BLOCK_TRADE_SELL,
                amount=30000000
            )
        ]
        
        collector.activity_timeline["000001"] = activities
        
        # Test getting all activities
        all_activities = collector.get_institution_activity_timeline("000001")
        assert len(all_activities) == 2
        
        # Test filtering by institution type
        fund_activities = collector.get_institution_activity_timeline(
            "000001", InstitutionType.MUTUAL_FUND
        )
        assert len(fund_activities) == 1
        assert fund_activities[0].institution.institution_type == InstitutionType.MUTUAL_FUND
        
        # Test non-existent stock
        empty_activities = collector.get_institution_activity_timeline("999999")
        assert len(empty_activities) == 0
    
    def test_get_institution_summary(self, collector):
        """Test getting institution summary."""
        
        # Setup mock activity timeline
        classifier = InstitutionClassifier()
        
        activities = [
            InstitutionalActivity(
                activity_id="test1",
                activity_date=date(2023, 1, 15),
                stock_code="000001",
                institution=classifier.create_institution("华夏基金管理有限公司"),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=50000000
            ),
            InstitutionalActivity(
                activity_id="test2",
                activity_date=date(2023, 1, 20),
                stock_code="000001",
                institution=classifier.create_institution("易方达基金管理有限公司"),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=30000000
            ),
            InstitutionalActivity(
                activity_id="test3",
                activity_date=date(2023, 1, 25),
                stock_code="000001",
                institution=classifier.create_institution("中信证券股份有限公司"),
                activity_type=ActivityType.BLOCK_TRADE_SELL,
                amount=20000000
            )
        ]
        
        collector.activity_timeline["000001"] = activities
        
        summary = collector.get_institution_summary("000001")
        
        # Verify summary structure
        assert isinstance(summary, dict)
        assert summary['total_activities'] == 3
        assert summary['total_amount'] == 100000000  # 50M + 30M + 20M
        assert summary['unique_institutions'] == 3
        
        # Check institution type counts
        assert 'institution_type_counts' in summary
        type_counts = summary['institution_type_counts']
        assert type_counts.get('mutual_fund', 0) == 2  # Two fund activities
        assert type_counts.get('securities_firm', 0) == 1  # One securities firm
        
        # Check date range
        assert 'date_range' in summary
        date_range = summary['date_range']
        assert date_range['start'] == date(2023, 1, 15)
        assert date_range['end'] == date(2023, 1, 25)
    
    def test_export_data_to_dataframes(self, collector):
        """Test exporting data to pandas DataFrames."""
        
        # Setup mock activity timeline
        classifier = InstitutionClassifier()
        
        activities = [
            InstitutionalActivity(
                activity_id="test1",
                activity_date=date(2023, 1, 15),
                stock_code="000001",
                institution=classifier.create_institution("华夏基金管理有限公司"),
                activity_type=ActivityType.DRAGON_TIGER_BUY,
                amount=50000000,
                volume=2000000,
                price=25.0,
                source_type="dragon_tiger",
                confidence_score=0.9
            )
        ]
        
        collector.activity_timeline["000001"] = activities
        
        dataframes = collector.export_data_to_dataframes("000001")
        
        # Verify structure
        assert isinstance(dataframes, dict)
        assert 'activities' in dataframes
        assert 'summary' in dataframes
        
        # Check activities DataFrame
        activities_df = dataframes['activities']
        assert isinstance(activities_df, pd.DataFrame)
        assert len(activities_df) == 1
        
        # Check required columns
        required_columns = [
            'activity_id', 'activity_date', 'stock_code', 'institution_id',
            'institution_name', 'institution_type', 'activity_type',
            'amount', 'volume', 'price', 'source_type', 'confidence_score'
        ]
        
        for col in required_columns:
            assert col in activities_df.columns
        
        # Check summary DataFrame
        summary_df = dataframes['summary']
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 1


class TestDataStructures:
    """Test data structure classes"""
    
    def test_institutional_investor_creation(self):
        """Test InstitutionalInvestor creation."""
        
        investor = InstitutionalInvestor(
            institution_id="test_001",
            name="Test Fund",
            institution_type=InstitutionType.MUTUAL_FUND,
            confidence_score=0.9
        )
        
        assert investor.institution_id == "test_001"
        assert investor.name == "Test Fund"
        assert investor.institution_type == InstitutionType.MUTUAL_FUND
        assert investor.confidence_score == 0.9
        assert investor.name_patterns == ["Test Fund"]  # Auto-populated
    
    def test_dragon_tiger_record_creation(self):
        """Test DragonTigerRecord creation."""
        
        record = DragonTigerRecord(
            trade_date=date(2023, 1, 15),
            stock_code="000001",
            stock_name="Test Stock",
            seat_name="Test Seat",
            seat_type="buy",
            amount=50000000
        )
        
        assert record.trade_date == date(2023, 1, 15)
        assert record.stock_code == "000001"
        assert record.seat_type == "buy"
        assert record.amount == 50000000
    
    def test_shareholder_record_creation(self):
        """Test ShareholderRecord creation."""
        
        record = ShareholderRecord(
            report_date=date(2023, 3, 31),
            stock_code="000001",
            stock_name="Test Stock",
            shareholder_name="Test Shareholder",
            shareholding_ratio=5.5,
            shares_held=55000000
        )
        
        assert record.report_date == date(2023, 3, 31)
        assert record.stock_code == "000001"
        assert record.shareholding_ratio == 5.5
        assert record.shares_held == 55000000
    
    def test_block_trade_record_creation(self):
        """Test BlockTradeRecord creation."""
        
        record = BlockTradeRecord(
            trade_date=date(2023, 2, 10),
            stock_code="000001",
            stock_name="Test Stock",
            volume=1000000,
            price=25.50,
            total_amount=25500000
        )
        
        assert record.trade_date == date(2023, 2, 10)
        assert record.stock_code == "000001"
        assert record.volume == 1000000
        assert record.price == 25.50
        assert record.total_amount == 25500000
    
    def test_institutional_activity_creation(self):
        """Test InstitutionalActivity creation."""
        
        classifier = InstitutionClassifier()
        institution = classifier.create_institution("Test Institution")
        
        activity = InstitutionalActivity(
            activity_id="test_activity",
            activity_date=date(2023, 1, 15),
            stock_code="000001",
            institution=institution,
            activity_type=ActivityType.DRAGON_TIGER_BUY,
            amount=50000000
        )
        
        assert activity.activity_id == "test_activity"
        assert activity.activity_date == date(2023, 1, 15)
        assert activity.stock_code == "000001"
        assert activity.institution == institution
        assert activity.activity_type == ActivityType.DRAGON_TIGER_BUY
        assert activity.amount == 50000000


if __name__ == "__main__":
    pytest.main([__file__])