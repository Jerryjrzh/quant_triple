"""Main FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from config.settings import get_settings
from stock_analysis_system.core.database import Base, engine
from stock_analysis_system.data.data_source_manager import get_data_source_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Stock Analysis System...")

    # Create database tables if they don't exist
    # (This is handled by Alembic migrations in production)

    yield

    # Shutdown
    print("🛑 Shutting down Stock Analysis System...")


# Get application settings
settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize password context for JWT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize HTTP Bearer for JWT
security = HTTPBearer()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive stock analysis system with Spring Festival temporal analysis",
    lifespan=lifespan,
    debug=settings.debug,
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.api.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.api.secret_key, algorithm=settings.api.algorithm
    )
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.api.secret_key,
            algorithms=[settings.api.algorithm],
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Welcome to Stock Analysis System",
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        # Test database connection
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Test data sources
    try:
        data_manager = await get_data_source_manager()
        data_sources_health = await data_manager.health_check()
        data_sources_summary = data_manager.get_health_summary()
    except Exception as e:
        data_sources_health = {}
        data_sources_summary = {"error": str(e)}

    return {
        "status": "ok",
        "database": db_status,
        "data_sources": data_sources_summary,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/api/v1/info")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def api_info(request: Request) -> Dict[str, Any]:
    """API information endpoint."""
    return {
        "api_version": "v1",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "features": [
            "Spring Festival Analysis",
            "Institutional Fund Tracking",
            "Risk Management",
            "Stock Screening",
            "Real-time Alerts",
        ],
    }


@app.get("/api/v1/stocks")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def list_stocks(
    request: Request, limit: int = 50, offset: int = 0, search: Optional[str] = None
) -> Dict[str, Any]:
    """List available stocks with optional search."""
    try:
        data_manager = await get_data_source_manager()
        stock_list_df = await data_manager.get_stock_list()

        if stock_list_df.empty:
            # Fallback to mock data if no real data available
            stocks = [
                {
                    "symbol": "000001.SZ",
                    "name": "平安银行",
                    "market": "深圳",
                    "sector": "金融",
                },
                {
                    "symbol": "600000.SH",
                    "name": "浦发银行",
                    "market": "上海",
                    "sector": "金融",
                },
            ]
        else:
            # Convert DataFrame to list of dictionaries
            stocks = []
            for _, row in stock_list_df.iterrows():
                stock = {
                    "symbol": row.get("stock_code", row.get("symbol", "")),
                    "name": row.get("name", ""),
                    "market": (
                        "深圳" if row.get("stock_code", "").endswith(".SZ") else "上海"
                    ),
                    "sector": row.get("industry", "未知"),
                }
                stocks.append(stock)

        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            stocks = [
                stock
                for stock in stocks
                if search_lower in stock["name"].lower()
                or search_lower in stock["symbol"].lower()
            ]

        # Apply pagination
        total = len(stocks)
        paginated_stocks = stocks[offset : offset + limit]

        return {
            "stocks": paginated_stocks,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Error fetching stock list: {e}")
        # Return mock data as fallback
        stocks = [
            {
                "symbol": "000001.SZ",
                "name": "平安银行",
                "market": "深圳",
                "sector": "金融",
            }
        ]

        return {
            "stocks": stocks[offset : offset + limit],
            "total": len(stocks),
            "limit": limit,
            "offset": offset,
            "warning": "Using fallback data due to data source issues",
        }


@app.get("/api/v1/stocks/{symbol}")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def get_stock_info(request: Request, symbol: str) -> Dict[str, Any]:
    """Get detailed information for a specific stock."""
    # This is a placeholder implementation
    # In a real implementation, this would query the database

    stock_info = {
        "symbol": symbol,
        "name": "平安银行" if symbol == "000001.SZ" else "未知股票",
        "market": "深圳" if symbol.endswith(".SZ") else "上海",
        "sector": "金融",
        "industry": "银行",
        "current_price": 12.50,
        "change": 0.26,
        "change_percent": 2.1,
        "volume": 125000000,
        "market_cap": 241500000000,
        "pe_ratio": 5.8,
        "pb_ratio": 0.65,
        "dividend_yield": 3.2,
        "last_updated": datetime.now().isoformat(),
    }

    return stock_info


@app.get("/api/v1/stocks/{symbol}/data")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def get_stock_data(
    request: Request,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 30,
) -> Dict[str, Any]:
    """Get historical stock data."""
    try:
        # Parse dates
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end_dt = datetime.now().date()

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            start_dt = end_dt - timedelta(days=days)

        # Get data from data source manager
        data_manager = await get_data_source_manager()
        stock_data = await data_manager.get_stock_data(symbol, start_dt, end_dt)

        if stock_data.empty:
            return {
                "symbol": symbol,
                "data": [],
                "message": "No data available for the specified period",
            }

        # Convert to API format
        data_records = []
        for _, row in stock_data.iterrows():
            record = {
                "date": (
                    row["trade_date"].strftime("%Y-%m-%d")
                    if pd.notna(row["trade_date"])
                    else None
                ),
                "open": (
                    float(row["open_price"]) if pd.notna(row["open_price"]) else None
                ),
                "high": (
                    float(row["high_price"]) if pd.notna(row["high_price"]) else None
                ),
                "low": float(row["low_price"]) if pd.notna(row["low_price"]) else None,
                "close": (
                    float(row["close_price"]) if pd.notna(row["close_price"]) else None
                ),
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
            }
            data_records.append(record)

        return {
            "symbol": symbol,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "data": data_records,
            "count": len(data_records),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stock data")


@app.get("/api/v1/stocks/{symbol}/spring-festival")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def get_spring_festival_analysis(
    request: Request, symbol: str, years: int = 5
) -> Dict[str, Any]:
    """Get Spring Festival analysis for a specific stock using real data."""
    try:
        from datetime import date, timedelta

        from stock_analysis_system.analysis.spring_festival_engine import (
            SpringFestivalAlignmentEngine,
        )

        # Get stock data
        data_manager = await get_data_source_manager()
        end_date = date.today()
        start_date = date(end_date.year - years, 1, 1)

        stock_data = await data_manager.get_stock_data(symbol, start_date, end_date)

        if stock_data.empty:
            return {
                "symbol": symbol,
                "error": "No data available for this symbol",
                "analysis_period": f"{start_date} to {end_date}",
            }

        # Perform Spring Festival analysis
        engine = SpringFestivalAlignmentEngine(window_days=60)

        try:
            # Align data to Spring Festival
            aligned_data = engine.align_to_spring_festival(stock_data)

            # Identify patterns
            pattern = engine.identify_seasonal_patterns(aligned_data)

            # Get current position
            position = engine.get_current_position(symbol)

            # Generate trading signals
            signals = engine.generate_trading_signals(pattern, position)

            # Build yearly data
            yearly_data = []
            for year in aligned_data.years:
                sf_date = engine.chinese_calendar.get_spring_festival(year)
                if sf_date:
                    year_data = aligned_data.get_year_data(year)

                    # Calculate returns for this year
                    before_data = [
                        dp for dp in year_data if dp.is_before_spring_festival
                    ]
                    after_data = [dp for dp in year_data if dp.is_after_spring_festival]

                    return_before = (
                        np.mean([dp.normalized_price for dp in before_data])
                        if before_data
                        else 0
                    )
                    return_after = (
                        np.mean([dp.normalized_price for dp in after_data])
                        if after_data
                        else 0
                    )

                    yearly_data.append(
                        {
                            "year": year,
                            "spring_festival_date": sf_date.isoformat(),
                            "return_before": round(return_before, 2),
                            "return_after": round(return_after, 2),
                            "data_points": len(year_data),
                        }
                    )

            # Generate recommendations
            recommendations = []
            if pattern.is_bullish_before and pattern.pattern_strength > 0.5:
                recommendations.append(
                    f"历史数据显示该股票在春节前平均上涨{pattern.average_return_before:.1f}%"
                )
            elif pattern.average_return_before < -1.0:
                recommendations.append(
                    f"历史数据显示该股票在春节前平均下跌{abs(pattern.average_return_before):.1f}%"
                )

            if pattern.is_bullish_after and pattern.pattern_strength > 0.5:
                recommendations.append(
                    f"春节后该股票平均上涨{pattern.average_return_after:.1f}%"
                )
            elif pattern.average_return_after < -1.0:
                recommendations.append(
                    f"春节后该股票平均下跌{abs(pattern.average_return_after):.1f}%"
                )

            if pattern.volatility_ratio > 1.5:
                recommendations.append("春节期间波动性显著增加，注意风险控制")

            if pattern.confidence_level < 0.6:
                recommendations.append("模式置信度较低，建议结合其他分析方法")

            analysis = {
                "symbol": symbol,
                "analysis_period": f"{aligned_data.years[0]}-{aligned_data.years[-1]}",
                "data_points": len(aligned_data.data_points),
                "spring_festival_pattern": {
                    "average_return_before": round(pattern.average_return_before, 2),
                    "average_return_after": round(pattern.average_return_after, 2),
                    "volatility_before": round(pattern.volatility_before, 2),
                    "volatility_after": round(pattern.volatility_after, 2),
                    "volatility_ratio": round(pattern.volatility_ratio, 2),
                    "pattern_strength": round(pattern.pattern_strength, 3),
                    "confidence_score": round(pattern.confidence_level, 3),
                    "consistency_score": round(pattern.consistency_score, 3),
                    "peak_day": pattern.peak_day,
                    "trough_day": pattern.trough_day,
                },
                "current_position": {
                    "position": position["position"],
                    "days_to_spring_festival": position["days_to_spring_festival"],
                    "in_analysis_window": position["in_analysis_window"],
                    "spring_festival_date": (
                        position["spring_festival_date"].isoformat()
                        if position["spring_festival_date"]
                        else None
                    ),
                },
                "trading_signals": {
                    "signal": signals["signal"],
                    "strength": round(signals["strength"], 2),
                    "recommended_action": signals["recommended_action"],
                    "reason": signals["reason"],
                },
                "yearly_data": yearly_data,
                "recommendations": recommendations,
                "analysis_date": datetime.now().isoformat(),
            }

            return analysis

        except ValueError as e:
            # Handle insufficient data or other analysis errors
            return {
                "symbol": symbol,
                "error": str(e),
                "analysis_period": f"{start_date} to {end_date}",
                "data_points": len(stock_data),
                "message": "Unable to perform Spring Festival analysis with available data",
            }

    except Exception as e:
        logger.error(f"Spring Festival analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Spring Festival analysis failed")


@app.get("/api/v1/stocks/{symbol}/intraday")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def get_intraday_data(
    request: Request, symbol: str, timeframe: str = "5min", days: int = 7
) -> Dict[str, Any]:
    """Get intraday stock data."""
    try:
        # Validate timeframe
        valid_timeframes = ["5min", "15min", "30min", "60min"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timeframe. Must be one of: {valid_timeframes}",
            )

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Get data from data source manager
        data_manager = await get_data_source_manager()
        intraday_data = await data_manager.get_intraday_data(
            symbol, start_date, end_date, timeframe
        )

        if intraday_data.empty:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": [],
                "message": f"No {timeframe} data available for the specified period",
            }

        # Convert to API format
        data_records = []
        for _, row in intraday_data.iterrows():
            record = {
                "datetime": (
                    row["datetime"].isoformat() if pd.notna(row["datetime"]) else None
                ),
                "date": (
                    row["trade_date"].isoformat()
                    if "trade_date" in row and pd.notna(row["trade_date"])
                    else None
                ),
                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
            }
            data_records.append(record)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data": data_records,
            "count": len(data_records),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching intraday data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch intraday data")


@app.get("/api/v1/screening")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def screen_stocks(
    request: Request,
    min_market_cap: Optional[float] = None,
    max_pe_ratio: Optional[float] = None,
    min_dividend_yield: Optional[float] = None,
    sector: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Screen stocks based on various criteria."""
    # This is a placeholder implementation
    # In a real implementation, this would use the screening engine

    screened_stocks = [
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "market_cap": 241500000000,
            "pe_ratio": 5.8,
            "dividend_yield": 3.2,
            "sector": "金融",
            "score": 85.2,
        },
        {
            "symbol": "600036.SH",
            "name": "招商银行",
            "market_cap": 1250000000000,
            "pe_ratio": 6.2,
            "dividend_yield": 2.8,
            "sector": "金融",
            "score": 88.7,
        },
    ]

    return {
        "screened_stocks": screened_stocks[:limit],
        "total_matches": len(screened_stocks),
        "criteria": {
            "min_market_cap": min_market_cap,
            "max_pe_ratio": max_pe_ratio,
            "min_dividend_yield": min_dividend_yield,
            "sector": sector,
        },
    }


@app.get("/api/v1/alerts")
@limiter.limit(f"{settings.api.rate_limit_requests}/minute")
async def get_alerts(
    request: Request,
    username: str = Depends(verify_token),
    active_only: bool = True,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get user alerts (requires authentication)."""
    # This is a placeholder implementation
    # In a real implementation, this would query user-specific alerts

    alerts = [
        {
            "id": 1,
            "symbol": "000001.SZ",
            "type": "price_target",
            "condition": "price >= 13.00",
            "status": "active",
            "created_at": "2024-01-15T10:30:00",
            "triggered_at": None,
        },
        {
            "id": 2,
            "symbol": "600036.SH",
            "type": "spring_festival",
            "condition": "春节前15天提醒",
            "status": "triggered",
            "created_at": "2024-01-10T09:15:00",
            "triggered_at": "2024-01-25T14:20:00",
        },
    ]

    if active_only:
        alerts = [alert for alert in alerts if alert["status"] == "active"]

    return {"alerts": alerts[:limit], "total": len(alerts), "user": username}


# Include routers
from stock_analysis_system.api.visualization import router as visualization_router

app.include_router(visualization_router)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        debug=settings.debug,
    )
