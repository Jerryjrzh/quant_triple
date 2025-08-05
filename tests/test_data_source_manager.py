"""Tests for Data Source Manager."""

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from stock_analysis_system.data.data_source_manager import (
    AkshareDataSource,
    CircuitBreaker,
    DataSourceManager,
    DataSourceStatus,
    DataSourceType,
    RateLimitConfig,
    RateLimiter,
    TushareDataSource,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.can_execute() is True
        assert cb.state == "closed"

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Record failures
        for _ in range(3):
            cb.on_failure()

        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(
            failure_threshold=2, recovery_timeout=0
        )  # Immediate recovery

        # Trigger circuit open
        cb.on_failure()
        cb.on_failure()
        assert cb.state == "open"

        # Should allow execution after timeout
        assert cb.can_execute() is True
        assert cb.state == "half_open"

        # Success should close circuit
        cb.on_success()
        assert cb.state == "closed"


class TestRateLimiter:
    """Test rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        config = RateLimitConfig(
            requests_per_minute=2, burst_limit=1, cooldown_period=1
        )
        limiter = RateLimiter(config)

        # First request should be immediate
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed < 0.1  # Should be immediate

    @pytest.mark.asyncio
    async def test_rate_limiter_burst_limit(self):
        """Test burst limit enforcement."""
        config = RateLimitConfig(
            requests_per_minute=10, burst_limit=2, cooldown_period=1
        )
        limiter = RateLimiter(config)

        # First two requests should be immediate
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed >= 0.9  # Should wait for cooldown


class TestTushareDataSource:
    """Test Tushare data source."""

    @pytest.fixture
    def mock_tushare_api(self):
        """Mock Tushare API."""
        with patch("tushare.pro_api") as mock_api:
            mock_instance = Mock()
            mock_api.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_get_stock_data_success(self, mock_tushare_api):
        """Test successful stock data retrieval."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240101"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "vol": [1000000],
                "amount": [10500000],
            }
        )
        mock_tushare_api.daily.return_value = mock_data

        with patch(
            "stock_analysis_system.data.data_source_manager.settings"
        ) as mock_settings:
            mock_settings.data_sources.tushare_token = "test_token"
            mock_settings.data_sources.requests_per_minute = 100

            source = TushareDataSource()
            result = await source.get_stock_data(
                "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
            )

            assert not result.empty
            assert "stock_code" in result.columns
            assert "trade_date" in result.columns
            assert "open_price" in result.columns

    @pytest.mark.asyncio
    async def test_get_stock_data_empty_result(self, mock_tushare_api):
        """Test handling of empty data result."""
        mock_tushare_api.daily.return_value = pd.DataFrame()

        with patch(
            "stock_analysis_system.data.data_source_manager.settings"
        ) as mock_settings:
            mock_settings.data_sources.tushare_token = "test_token"
            mock_settings.data_sources.requests_per_minute = 100

            source = TushareDataSource()
            result = await source.get_stock_data(
                "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
            )

            assert result.empty

    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_tushare_api):
        """Test successful connection test."""
        mock_tushare_api.daily.return_value = pd.DataFrame({"ts_code": ["000001.SZ"]})

        with patch(
            "stock_analysis_system.data.data_source_manager.settings"
        ) as mock_settings:
            mock_settings.data_sources.tushare_token = "test_token"
            mock_settings.data_sources.requests_per_minute = 100

            source = TushareDataSource()
            result = await source.test_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, mock_tushare_api):
        """Test connection test failure."""
        mock_tushare_api.daily.side_effect = Exception("Connection failed")

        with patch(
            "stock_analysis_system.data.data_source_manager.settings"
        ) as mock_settings:
            mock_settings.data_sources.tushare_token = "test_token"
            mock_settings.data_sources.requests_per_minute = 100

            source = TushareDataSource()
            result = await source.test_connection()

            assert result is False


class TestAkshareDataSource:
    """Test AkShare data source."""

    @pytest.mark.asyncio
    async def test_get_stock_data_success(self):
        """Test successful stock data retrieval from AkShare."""
        mock_data = pd.DataFrame(
            {
                "日期": ["2024-01-01"],
                "开盘": [10.0],
                "最高": [11.0],
                "最低": [9.0],
                "收盘": [10.5],
                "成交量": [1000000],
                "成交额": [10500000],
            }
        )

        with patch("akshare.stock_zh_a_hist", return_value=mock_data):
            source = AkshareDataSource()
            result = await source.get_stock_data(
                "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
            )

            assert not result.empty
            assert "stock_code" in result.columns
            assert "trade_date" in result.columns
            assert result["stock_code"].iloc[0] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_get_stock_list_success(self):
        """Test successful stock list retrieval from AkShare."""
        mock_data = pd.DataFrame(
            {"code": ["000001", "000002"], "name": ["平安银行", "万科A"]}
        )

        with patch("akshare.stock_info_a_code_name", return_value=mock_data):
            source = AkshareDataSource()
            result = await source.get_stock_list()

            assert not result.empty
            assert "stock_code" in result.columns
            assert "name" in result.columns
            assert result["stock_code"].iloc[0] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        mock_data = pd.DataFrame({"日期": ["2024-01-01"]})

        with patch("akshare.stock_zh_a_hist", return_value=mock_data):
            source = AkshareDataSource()
            result = await source.test_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection test failure."""
        with patch(
            "akshare.stock_zh_a_hist", side_effect=Exception("Connection failed")
        ):
            source = AkshareDataSource()
            result = await source.test_connection()

            assert result is False


class TestDataSourceManager:
    """Test Data Source Manager."""

    @pytest.fixture
    def manager(self):
        """Create a data source manager for testing."""
        with patch(
            "stock_analysis_system.data.data_source_manager.settings"
        ) as mock_settings:
            mock_settings.data_sources.tushare_token = "test_token"
            mock_settings.data_sources.requests_per_minute = 100

            manager = DataSourceManager()
            return manager

    @pytest.mark.asyncio
    async def test_get_stock_data_primary_success(self, manager):
        """Test successful data retrieval from primary source."""
        mock_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2024-01-01")],
                "open_price": [10.0],
                "close_price": [10.5],
            }
        )

        # Mock the primary source (Tushare)
        with patch.object(
            manager.data_sources[DataSourceType.TUSHARE],
            "get_stock_data",
            return_value=mock_data,
        ):
            result = await manager.get_stock_data(
                "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
            )

            assert not result.empty
            assert result["stock_code"].iloc[0] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_get_stock_data_failover(self, manager):
        """Test failover to secondary source when primary fails."""
        mock_data = pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2024-01-01")],
                "open_price": [10.0],
                "close_price": [10.5],
            }
        )

        # Mock primary source failure and secondary source success
        with patch.object(
            manager.data_sources[DataSourceType.TUSHARE],
            "get_stock_data",
            side_effect=Exception("Primary failed"),
        ):
            with patch.object(
                manager.data_sources[DataSourceType.AKSHARE],
                "get_stock_data",
                return_value=mock_data,
            ):
                result = await manager.get_stock_data(
                    "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
                )

                assert not result.empty
                assert result["stock_code"].iloc[0] == "000001.SZ"

    @pytest.mark.asyncio
    async def test_get_stock_data_all_sources_fail(self, manager):
        """Test behavior when all sources fail."""
        # Mock all sources to fail
        with patch.object(
            manager.data_sources[DataSourceType.TUSHARE],
            "get_stock_data",
            side_effect=Exception("Tushare failed"),
        ):
            with patch.object(
                manager.data_sources[DataSourceType.AKSHARE],
                "get_stock_data",
                side_effect=Exception("AkShare failed"),
            ):

                with pytest.raises(Exception):
                    await manager.get_stock_data(
                        "000001.SZ", date(2024, 1, 1), date(2024, 1, 1)
                    )

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """Test health check functionality."""
        # Mock connection tests
        with patch.object(
            manager.data_sources[DataSourceType.TUSHARE],
            "test_connection",
            return_value=True,
        ):
            with patch.object(
                manager.data_sources[DataSourceType.AKSHARE],
                "test_connection",
                return_value=False,
            ):

                health_status = await manager.health_check()

                assert DataSourceType.TUSHARE in health_status
                assert DataSourceType.AKSHARE in health_status
                assert (
                    health_status[DataSourceType.TUSHARE].status
                    == DataSourceStatus.HEALTHY
                )
                assert (
                    health_status[DataSourceType.AKSHARE].status
                    == DataSourceStatus.DEGRADED
                )

    def test_get_health_summary(self, manager):
        """Test health summary generation."""
        summary = manager.get_health_summary()

        assert "total_sources" in summary
        assert "healthy_sources" in summary
        assert "degraded_sources" in summary
        assert "failed_sources" in summary
        assert "primary_source" in summary
        assert "sources" in summary

        assert summary["total_sources"] == len(manager.data_sources)
        assert summary["primary_source"] == DataSourceType.TUSHARE.value


if __name__ == "__main__":
    pytest.main([__file__])
