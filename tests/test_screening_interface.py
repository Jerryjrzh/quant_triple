"""
Unit tests for the Advanced Screening Interface

This module contains comprehensive tests for the stock screening system
including criteria evaluation, template management, and result analysis.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from stock_analysis_system.screening import (
    ScreeningInterface, ScreeningEngine, ScreeningCriteriaBuilder,
    TechnicalCriteria, SeasonalCriteria, InstitutionalCriteria, RiskCriteria,
    ScreeningTemplate, ScreeningResult, StockScore, PredefinedTemplates
)


class TestScreeningCriteria:
    """Test screening criteria classes."""
    
    def test_technical_criteria_creation(self):
        """Test technical criteria creation and serialization."""
        criteria = TechnicalCriteria(
            name="Test Technical",
            description="Test technical criteria",
            price_min=10.0,
            price_max=100.0,
            rsi_min=30.0,
            rsi_max=70.0,
            ma20_position="above"
        )
        
        assert criteria.name == "Test Technical"
        assert criteria.price_min == 10.0
        assert criteria.rsi_min == 30.0
        assert criteria.ma20_position == "above"
        
        # Test serialization
        criteria_dict = criteria.to_dict()
        assert criteria_dict['price_min'] == 10.0
        assert criteria_dict['type'] == 'TechnicalCriteria'
    
    def test_seasonal_criteria_creation(self):
        """Test seasonal criteria creation."""
        criteria = SeasonalCriteria(
            name="Test Seasonal",
            description="Test seasonal criteria",
            spring_festival_days_range=(-30, 30),
            spring_festival_pattern_strength=0.7,
            pattern_confidence_min=0.6
        )
        
        assert criteria.spring_festival_days_range == (-30, 30)
        assert criteria.spring_festival_pattern_strength == 0.7
        assert criteria.pattern_confidence_min == 0.6
    
    def test_institutional_criteria_creation(self):
        """Test institutional criteria creation."""
        criteria = InstitutionalCriteria(
            name="Test Institutional",
            description="Test institutional criteria",
            attention_score_min=60.0,
            new_institutional_entry=True,
            dragon_tiger_appearances=2,
            mutual_fund_activity=True
        )
        
        assert criteria.attention_score_min == 60.0
        assert criteria.new_institutional_entry is True
        assert criteria.dragon_tiger_appearances == 2
        assert criteria.mutual_fund_activity is True
    
    def test_risk_criteria_creation(self):
        """Test risk criteria creation."""
        criteria = RiskCriteria(
            name="Test Risk",
            description="Test risk criteria",
            volatility_max=0.3,
            sharpe_ratio_min=0.5,
            beta_max=1.2,
            max_drawdown_max=0.15
        )
        
        assert criteria.volatility_max == 0.3
        assert criteria.sharpe_ratio_min == 0.5
        assert criteria.beta_max == 1.2
        assert criteria.max_drawdown_max == 0.15


class TestScreeningCriteriaBuilder:
    """Test screening criteria builder."""
    
    def test_builder_pattern(self):
        """Test builder pattern functionality."""
        builder = ScreeningCriteriaBuilder()
        
        template = builder.with_technical_criteria(
            price_min=10.0,
            rsi_min=30.0
        ).with_seasonal_criteria(
            spring_festival_pattern_strength=0.7
        ).with_institutional_criteria(
            attention_score_min=60.0
        ).with_risk_criteria(
            volatility_max=0.3
        ).build_template(
            name="Test Template",
            description="Test template description",
            tags=['test', 'demo']
        )
        
        assert template.name == "Test Template"
        assert template.technical_criteria is not None
        assert template.seasonal_criteria is not None
        assert template.institutional_criteria is not None
        assert template.risk_criteria is not None
        assert len(template.tags) == 2
    
    def test_builder_reset(self):
        """Test builder reset functionality."""
        builder = ScreeningCriteriaBuilder()
        
        # Build first template
        template1 = builder.with_technical_criteria(
            price_min=10.0
        ).build_template("Template 1", "Description 1")
        
        # Reset and build second template
        template2 = builder.reset().with_seasonal_criteria(
            spring_festival_pattern_strength=0.8
        ).build_template("Template 2", "Description 2")
        
        assert template1.technical_criteria is not None
        assert template1.seasonal_criteria is None
        assert template2.technical_criteria is None
        assert template2.seasonal_criteria is not None


class TestPredefinedTemplates:
    """Test predefined template generation."""
    
    def test_growth_momentum_template(self):
        """Test growth momentum template."""
        template = PredefinedTemplates.growth_momentum_template()
        
        assert template.name == "Growth Momentum"
        assert template.technical_criteria is not None
        assert template.risk_criteria is not None
        assert 'growth' in template.tags
        assert 'momentum' in template.tags
    
    def test_spring_festival_template(self):
        """Test Spring Festival opportunity template."""
        template = PredefinedTemplates.spring_festival_opportunity_template()
        
        assert template.name == "Spring Festival Opportunity"
        assert template.seasonal_criteria is not None
        assert template.institutional_criteria is not None
        assert 'seasonal' in template.tags
        assert 'spring_festival' in template.tags
    
    def test_low_risk_value_template(self):
        """Test low risk value template."""
        template = PredefinedTemplates.low_risk_value_template()
        
        assert template.name == "Low Risk Value"
        assert template.technical_criteria is not None
        assert template.risk_criteria is not None
        assert 'low_risk' in template.tags
        assert 'conservative' in template.tags


class TestScreeningResult:
    """Test screening result classes."""
    
    def test_stock_score_creation(self):
        """Test stock score creation and serialization."""
        score = StockScore(
            stock_code="000001",
            stock_name="Test Stock",
            composite_score=75.5,
            technical_score=80.0,
            seasonal_score=70.0,
            institutional_score=75.0,
            risk_score=77.0,
            current_price=25.50,
            price_change_pct=2.5,
            sector="Technology",
            industry="Software"
        )
        
        assert score.stock_code == "000001"
        assert score.composite_score == 75.5
        assert score.sector == "Technology"
        
        # Test serialization
        score_dict = score.to_dict()
        assert score_dict['stock_code'] == "000001"
        assert score_dict['composite_score'] == 75.5
    
    def test_screening_result_creation(self):
        """Test screening result creation."""
        scores = [
            StockScore("000001", "Stock 1", 80.0),
            StockScore("000002", "Stock 2", 75.0),
            StockScore("000003", "Stock 3", 70.0)
        ]
        
        result = ScreeningResult(
            screening_id="test-123",
            template_name="Test Template",
            execution_time=datetime.now(),
            total_stocks_screened=100,
            stocks_passed=3,
            execution_duration_ms=1500,
            stock_scores=scores
        )
        
        assert result.screening_id == "test-123"
        assert result.stocks_passed == 3
        assert len(result.stock_scores) == 3
        
        # Test top stocks
        top_stocks = result.get_top_stocks(2)
        assert len(top_stocks) == 2
        assert top_stocks[0].composite_score == 80.0
        assert top_stocks[1].composite_score == 75.0
    
    def test_screening_result_filtering(self):
        """Test screening result filtering methods."""
        scores = [
            StockScore("000001", "Stock 1", 80.0, sector="Technology"),
            StockScore("000002", "Stock 2", 75.0, sector="Finance"),
            StockScore("000003", "Stock 3", 70.0, sector="Technology"),
            StockScore("000004", "Stock 4", 65.0, sector="Healthcare")
        ]
        
        result = ScreeningResult(
            screening_id="test-123",
            template_name="Test Template",
            execution_time=datetime.now(),
            total_stocks_screened=100,
            stocks_passed=4,
            execution_duration_ms=1500,
            stock_scores=scores
        )
        
        # Test sector filtering
        tech_stocks = result.filter_by_sector(["Technology"])
        assert len(tech_stocks) == 2
        assert all(stock.sector == "Technology" for stock in tech_stocks)
        
        # Test score range filtering
        high_score_stocks = result.filter_by_score_range(75.0, 85.0)
        assert len(high_score_stocks) == 2
        assert all(75.0 <= stock.composite_score <= 85.0 for stock in high_score_stocks)


@pytest.fixture
def mock_engines():
    """Create mock engines for testing."""
    data_source = Mock()
    sf_engine = Mock()
    inst_engine = Mock()
    risk_engine = Mock()
    
    return data_source, sf_engine, inst_engine, risk_engine


@pytest.fixture
def screening_engine(mock_engines):
    """Create screening engine with mocked dependencies."""
    data_source, sf_engine, inst_engine, risk_engine = mock_engines
    return ScreeningEngine(data_source, sf_engine, inst_engine, risk_engine)


@pytest.fixture
def screening_interface(screening_engine):
    """Create screening interface with mocked engine."""
    return ScreeningInterface(screening_engine)


class TestScreeningInterface:
    """Test screening interface functionality."""
    
    def test_interface_initialization(self, screening_interface):
        """Test interface initialization."""
        interface = screening_interface
        
        # Should have predefined templates loaded
        assert len(interface.templates) > 0
        assert "Growth Momentum" in interface.templates
        assert "Spring Festival Opportunity" in interface.templates
        assert "Low Risk Value" in interface.templates
        assert "Institutional Following" in interface.templates
    
    def test_get_template_list(self, screening_interface):
        """Test template list retrieval."""
        interface = screening_interface
        template_list = interface.get_template_list()
        
        assert len(template_list) > 0
        assert all('name' in template for template in template_list)
        assert all('description' in template for template in template_list)
        assert all('tags' in template for template in template_list)
    
    def test_get_template_details(self, screening_interface):
        """Test template details retrieval."""
        interface = screening_interface
        
        details = interface.get_template_details("Growth Momentum")
        assert details is not None
        assert details['name'] == "Growth Momentum"
        assert 'technical_criteria' in details
        assert 'risk_criteria' in details
        
        # Test non-existent template
        details = interface.get_template_details("Non-existent")
        assert details is None
    
    @pytest.mark.asyncio
    async def test_create_custom_template(self, screening_interface):
        """Test custom template creation."""
        interface = screening_interface
        
        template_name = await interface.create_custom_template(
            name="Test Custom Template",
            description="Test description",
            technical_params={
                'price_min': 10.0,
                'rsi_min': 30.0
            },
            seasonal_params={
                'spring_festival_pattern_strength': 0.7
            },
            tags=['test', 'custom']
        )
        
        assert template_name == "Test Custom Template"
        assert template_name in interface.templates
        
        template = interface.templates[template_name]
        assert template.technical_criteria is not None
        assert template.seasonal_criteria is not None
        assert template.institutional_criteria is None
        assert template.risk_criteria is None
        assert 'test' in template.tags
    
    @pytest.mark.asyncio
    async def test_update_template(self, screening_interface):
        """Test template updating."""
        interface = screening_interface
        
        # Create a test template first
        await interface.create_custom_template(
            name="Update Test Template",
            description="Original description",
            tags=['original']
        )
        
        # Update the template
        updated = await interface.update_template(
            "Update Test Template",
            description="Updated description",
            tags=['updated']
        )
        
        assert updated is True
        
        template = interface.templates["Update Test Template"]
        assert template.description == "Updated description"
        assert template.tags == ['updated']
        
        # Test updating non-existent template
        updated = await interface.update_template(
            "Non-existent Template",
            description="New description"
        )
        assert updated is False
    
    @pytest.mark.asyncio
    async def test_delete_template(self, screening_interface):
        """Test template deletion."""
        interface = screening_interface
        
        # Create a test template first
        await interface.create_custom_template(
            name="Delete Test Template",
            description="To be deleted"
        )
        
        assert "Delete Test Template" in interface.templates
        
        # Delete the template
        deleted = await interface.delete_template("Delete Test Template")
        assert deleted is True
        assert "Delete Test Template" not in interface.templates
        
        # Test deleting non-existent template
        deleted = await interface.delete_template("Non-existent Template")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_export_import_template(self, screening_interface):
        """Test template export and import."""
        interface = screening_interface
        
        # Create a test template
        await interface.create_custom_template(
            name="Export Test Template",
            description="Template for export testing",
            technical_params={'price_min': 10.0},
            tags=['export', 'test']
        )
        
        # Export template
        exported_json = await interface.export_template("Export Test Template")
        assert isinstance(exported_json, str)
        
        # Verify JSON structure
        template_data = json.loads(exported_json)
        assert template_data['name'] == "Export Test Template"
        assert template_data['description'] == "Template for export testing"
        assert 'export' in template_data['tags']
        
        # Delete original template
        await interface.delete_template("Export Test Template")
        assert "Export Test Template" not in interface.templates
        
        # Import template back
        imported_name = await interface.import_template(exported_json)
        assert imported_name == "Export Test Template"
        assert "Export Test Template" in interface.templates
        
        # Verify imported template
        imported_template = interface.templates["Export Test Template"]
        assert imported_template.description == "Template for export testing"
        assert 'export' in imported_template.tags
    
    @pytest.mark.asyncio
    async def test_export_nonexistent_template(self, screening_interface):
        """Test exporting non-existent template."""
        interface = screening_interface
        
        with pytest.raises(ValueError, match="Template 'Non-existent' not found"):
            await interface.export_template("Non-existent")
    
    @pytest.mark.asyncio
    async def test_import_invalid_template(self, screening_interface):
        """Test importing invalid template JSON."""
        interface = screening_interface
        
        with pytest.raises(ValueError, match="Failed to import template"):
            await interface.import_template("invalid json")
    
    @pytest.mark.asyncio
    async def test_cache_management(self, screening_interface):
        """Test cache management functionality."""
        interface = screening_interface
        
        # Test initial cache stats
        stats = await interface.get_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0
        assert stats['cache_ttl_minutes'] == 30
        
        # Clear cache (should work even when empty)
        await interface.clear_cache()
        
        # Test cleanup (should work even when empty)
        expired_count = await interface.cleanup_expired_cache()
        assert expired_count == 0


class TestScreeningEngine:
    """Test screening engine functionality."""
    
    @pytest.mark.asyncio
    async def test_get_default_stock_universe(self, screening_engine):
        """Test default stock universe generation."""
        engine = screening_engine
        
        universe = await engine._get_default_stock_universe()
        assert isinstance(universe, list)
        assert len(universe) > 0
        assert all(isinstance(code, str) for code in universe)
    
    @pytest.mark.asyncio
    async def test_get_stock_data(self, screening_engine):
        """Test stock data retrieval."""
        engine = screening_engine
        
        stock_data = await engine._get_stock_data("000001")
        assert stock_data is not None
        assert 'stock_code' in stock_data
        assert 'stock_name' in stock_data
        assert 'current_price' in stock_data
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, screening_engine):
        """Test technical indicators calculation."""
        engine = screening_engine
        
        stock_data = {'current_price': 10.0, 'volume': 1000000}
        indicators = await engine._calculate_technical_indicators("000001", stock_data)
        
        assert isinstance(indicators, dict)
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'ma20' in indicators
    
    def test_calculate_composite_score(self, screening_engine):
        """Test composite score calculation."""
        engine = screening_engine
        
        # Create a mock template
        template = Mock()
        template.technical_criteria = Mock()
        template.technical_criteria.enabled = True
        template.seasonal_criteria = Mock()
        template.seasonal_criteria.enabled = True
        template.institutional_criteria = None
        template.risk_criteria = Mock()
        template.risk_criteria.enabled = True
        
        composite = engine._calculate_composite_score(80.0, 70.0, 0.0, 75.0, template)
        
        assert 0.0 <= composite <= 100.0
        assert composite > 0  # Should be positive with valid scores
    
    def test_evaluate_ma_criteria(self, screening_engine):
        """Test moving average criteria evaluation."""
        engine = screening_engine
        
        indicators = {
            'ma5': 9.8,
            'ma20': 9.5,
            'ma50': 9.0
        }
        
        # Mock criteria
        criteria = Mock()
        criteria.ma5_position = 'above'
        criteria.ma20_position = 'above'
        criteria.ma50_position = None
        
        score = engine._evaluate_ma_criteria(indicators, criteria)
        assert score >= 0.0
    
    def test_evaluate_macd_signal(self, screening_engine):
        """Test MACD signal evaluation."""
        engine = screening_engine
        
        # Test bullish signal
        indicators = {'macd': 0.5, 'macd_signal': 0.3}
        score = engine._evaluate_macd_signal(indicators, 'bullish')
        assert score == 25.0
        
        # Test bearish signal
        indicators = {'macd': 0.3, 'macd_signal': 0.5}
        score = engine._evaluate_macd_signal(indicators, 'bearish')
        assert score == 25.0
        
        # Test neutral signal
        score = engine._evaluate_macd_signal(indicators, 'neutral')
        assert score == 15.0
        
        # Test failed signal
        indicators = {'macd': 0.5, 'macd_signal': 0.3}
        score = engine._evaluate_macd_signal(indicators, 'bearish')
        assert score == 0.0
    
    def test_create_criteria_summary(self, screening_engine):
        """Test criteria summary creation."""
        engine = screening_engine
        
        # Create mock template
        template = Mock()
        template.name = "Test Template"
        template.technical_criteria = Mock()
        template.technical_criteria.enabled = True
        template.seasonal_criteria = None
        template.institutional_criteria = Mock()
        template.institutional_criteria.enabled = False
        template.risk_criteria = Mock()
        template.risk_criteria.enabled = True
        template.logical_operator = Mock()
        template.logical_operator.value = "and"
        
        summary = engine._create_criteria_summary(template)
        
        assert summary['template_name'] == "Test Template"
        assert summary['has_technical'] is True
        assert summary['has_seasonal'] is False
        assert summary['has_institutional'] is False
        assert summary['has_risk'] is True
        assert summary['logical_operator'] == "and"
    
    def test_calculate_score_distribution(self, screening_engine):
        """Test score distribution calculation."""
        engine = screening_engine
        
        scores = [
            StockScore("000001", "Stock 1", 85.0),  # excellent
            StockScore("000002", "Stock 2", 75.0),  # good
            StockScore("000003", "Stock 3", 65.0),  # good
            StockScore("000004", "Stock 4", 45.0),  # fair
            StockScore("000005", "Stock 5", 25.0),  # poor
        ]
        
        distribution = engine._calculate_score_distribution(scores)
        
        assert distribution['excellent'] == 1
        assert distribution['good'] == 2
        assert distribution['fair'] == 1
        assert distribution['poor'] == 1
        
        # Test empty scores
        empty_distribution = engine._calculate_score_distribution([])
        assert empty_distribution == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])