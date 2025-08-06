"""
Pool Export/Import Functionality

This module provides comprehensive export/import capabilities for stock pools
including multiple formats (CSV, JSON, Excel), sharing features, backup/restore,
and integration with external portfolio management tools.
"""

import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import zipfile
import io
import logging
from dataclasses import asdict

from .stock_pool_manager import StockPoolManager, StockPool, StockInfo, PoolMetrics

logger = logging.getLogger(__name__)

class PoolExportImport:
    """
    Comprehensive Pool Export/Import System
    
    Provides multiple format support, sharing capabilities, backup/restore,
    and external tool integration for stock pool management.
    """
    
    def __init__(self, pool_manager: StockPoolManager):
        self.pool_manager = pool_manager
        self.supported_formats = ['csv', 'json', 'excel', 'xml', 'yaml']
        self.backup_directory = Path("pool_backups")
        self.backup_directory.mkdir(exist_ok=True)
    
    async def export_pool(
        self,
        pool_id: str,
        format_type: str = 'json',
        include_history: bool = True,
        include_analytics: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export a single pool to specified format"""
        
        if format_type not in self.supported_formats:
            return {"error": f"Unsupported format: {format_type}"}
        
        if pool_id not in self.pool_manager.pools:
            return {"error": "Pool not found"}
        
        pool = self.pool_manager.pools[pool_id]
        
        # Prepare export data
        export_data = await self._prepare_pool_export_data(
            pool, include_history, include_analytics
        )
        
        # Generate filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pool_export_{pool.name}_{timestamp}.{format_type}"
        
        # Export based on format
        try:
            if format_type == 'json':
                result = await self._export_to_json(export_data, output_path)
            elif format_type == 'csv':
                result = await self._export_to_csv(export_data, output_path)
            elif format_type == 'excel':
                result = await self._export_to_excel(export_data, output_path)
            elif format_type == 'xml':
                result = await self._export_to_xml(export_data, output_path)
            elif format_type == 'yaml':
                result = await self._export_to_yaml(export_data, output_path)
            
            logger.info(f"Successfully exported pool {pool_id} to {output_path}")
            return {
                "success": True,
                "file_path": output_path,
                "format": format_type,
                "export_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Export failed for pool {pool_id}: {e}")
            return {"error": f"Export failed: {str(e)}"}
    
    async def export_multiple_pools(
        self,
        pool_ids: List[str],
        format_type: str = 'json',
        create_archive: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export multiple pools"""
        
        if not pool_ids:
            return {"error": "No pools specified"}
        
        export_results = []
        exported_files = []
        
        # Export each pool
        for pool_id in pool_ids:
            if pool_id in self.pool_manager.pools:
                pool = self.pool_manager.pools[pool_id]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"pool_{pool.name}_{timestamp}.{format_type}"
                
                result = await self.export_pool(
                    pool_id=pool_id,
                    format_type=format_type,
                    output_path=file_path
                )
                
                export_results.append(result)
                if result.get("success"):
                    exported_files.append(file_path)
        
        # Create archive if requested
        if create_archive and exported_files:
            archive_path = output_path or f"pools_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            try:
                with zipfile.ZipFile(archive_path, 'w') as zipf:
                    for file_path in exported_files:
                        if Path(file_path).exists():
                            zipf.write(file_path, Path(file_path).name)
                            # Clean up individual files
                            Path(file_path).unlink()
                
                return {
                    "success": True,
                    "archive_path": archive_path,
                    "exported_pools": len(exported_files),
                    "export_results": export_results
                }
                
            except Exception as e:
                logger.error(f"Failed to create archive: {e}")
                return {"error": f"Archive creation failed: {str(e)}"}
        
        return {
            "success": True,
            "exported_files": exported_files,
            "export_results": export_results
        }
    
    async def import_pool(
        self,
        file_path: str,
        format_type: Optional[str] = None,
        merge_strategy: str = 'replace',
        validate_data: bool = True
    ) -> Dict[str, Any]:
        """Import a pool from file"""
        
        if not Path(file_path).exists():
            return {"error": "File not found"}
        
        # Auto-detect format if not specified
        if not format_type:
            format_type = Path(file_path).suffix.lstrip('.')
        
        if format_type not in self.supported_formats:
            return {"error": f"Unsupported format: {format_type}"}
        
        try:
            # Import based on format
            if format_type == 'json':
                import_data = await self._import_from_json(file_path)
            elif format_type == 'csv':
                import_data = await self._import_from_csv(file_path)
            elif format_type == 'excel':
                import_data = await self._import_from_excel(file_path)
            elif format_type == 'xml':
                import_data = await self._import_from_xml(file_path)
            elif format_type == 'yaml':
                import_data = await self._import_from_yaml(file_path)
            
            # Validate data if requested
            if validate_data:
                validation_result = await self._validate_import_data(import_data)
                if not validation_result["valid"]:
                    return {"error": f"Data validation failed: {validation_result['errors']}"}
            
            # Create or update pool
            result = await self._create_pool_from_import_data(import_data, merge_strategy)
            
            logger.info(f"Successfully imported pool from {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Import failed for {file_path}: {e}")
            return {"error": f"Import failed: {str(e)}"}
    
    async def create_pool_backup(
        self,
        pool_id: str,
        include_full_history: bool = True,
        backup_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a comprehensive backup of a pool"""
        
        if pool_id not in self.pool_manager.pools:
            return {"error": "Pool not found"}
        
        pool = self.pool_manager.pools[pool_id]
        
        # Generate backup name
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{pool.name}_{timestamp}"
        
        backup_path = self.backup_directory / f"{backup_name}.json"
        
        # Prepare comprehensive backup data
        backup_data = {
            "backup_info": {
                "pool_id": pool_id,
                "pool_name": pool.name,
                "backup_timestamp": datetime.now().isoformat(),
                "backup_version": "1.0",
                "include_full_history": include_full_history
            },
            "pool_data": await self._prepare_pool_export_data(
                pool, include_full_history, True
            ),
            "system_info": {
                "pool_manager_version": "1.0",
                "export_format_version": "1.0"
            }
        }
        
        # Add full history if requested
        if include_full_history:
            backup_data["pool_history"] = await self.pool_manager.get_pool_history(pool_id)
        
        try:
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Created backup for pool {pool_id} at {backup_path}")
            return {
                "success": True,
                "backup_path": str(backup_path),
                "backup_name": backup_name,
                "backup_size": backup_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Backup creation failed for pool {pool_id}: {e}")
            return {"error": f"Backup creation failed: {str(e)}"}
    
    async def restore_pool_from_backup(
        self,
        backup_path: str,
        restore_strategy: str = 'replace'
    ) -> Dict[str, Any]:
        """Restore a pool from backup"""
        
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return {"error": "Backup file not found"}
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Validate backup format
            if "backup_info" not in backup_data or "pool_data" not in backup_data:
                return {"error": "Invalid backup format"}
            
            backup_info = backup_data["backup_info"]
            pool_data = backup_data["pool_data"]
            
            # Restore pool
            result = await self._create_pool_from_import_data(pool_data, restore_strategy)
            
            # Restore history if available
            if "pool_history" in backup_data and result.get("success"):
                pool_id = result["pool_id"]
                self.pool_manager.pool_history[pool_id] = backup_data["pool_history"]
            
            logger.info(f"Successfully restored pool from backup {backup_path}")
            return {
                "success": True,
                "restored_pool_id": result.get("pool_id"),
                "backup_info": backup_info,
                "restore_strategy": restore_strategy
            }
            
        except Exception as e:
            logger.error(f"Restore failed for backup {backup_path}: {e}")
            return {"error": f"Restore failed: {str(e)}"}
    
    async def create_shareable_pool_link(
        self,
        pool_id: str,
        access_level: str = 'read_only',
        expiry_days: int = 30,
        include_analytics: bool = True
    ) -> Dict[str, Any]:
        """Create a shareable link for pool collaboration"""
        
        if pool_id not in self.pool_manager.pools:
            return {"error": "Pool not found"}
        
        pool = self.pool_manager.pools[pool_id]
        
        # Generate share token
        import hashlib
        import secrets
        
        share_token = secrets.token_urlsafe(32)
        share_id = hashlib.md5(f"{pool_id}_{share_token}".encode()).hexdigest()
        
        # Create shareable data
        share_data = {
            "share_info": {
                "share_id": share_id,
                "pool_id": pool_id,
                "pool_name": pool.name,
                "access_level": access_level,
                "created_timestamp": datetime.now().isoformat(),
                "expiry_timestamp": (datetime.now() + pd.Timedelta(days=expiry_days)).isoformat(),
                "include_analytics": include_analytics
            },
            "pool_data": await self._prepare_pool_export_data(
                pool, include_history=False, include_analytics=include_analytics
            )
        }
        
        # Save share data
        share_path = self.backup_directory / f"share_{share_id}.json"
        
        try:
            with open(share_path, 'w') as f:
                json.dump(share_data, f, indent=2, default=str)
            
            # Generate shareable URL (mock)
            share_url = f"https://stockanalysis.app/shared/{share_id}"
            
            logger.info(f"Created shareable link for pool {pool_id}")
            return {
                "success": True,
                "share_id": share_id,
                "share_url": share_url,
                "access_level": access_level,
                "expiry_date": share_data["share_info"]["expiry_timestamp"],
                "share_token": share_token
            }
            
        except Exception as e:
            logger.error(f"Share creation failed for pool {pool_id}: {e}")
            return {"error": f"Share creation failed: {str(e)}"}
    
    async def export_for_external_tools(
        self,
        pool_id: str,
        tool_type: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export pool data for external portfolio management tools"""
        
        supported_tools = {
            'portfolio_visualizer': self._export_for_portfolio_visualizer,
            'morningstar': self._export_for_morningstar,
            'yahoo_finance': self._export_for_yahoo_finance,
            'google_sheets': self._export_for_google_sheets,
            'excel_template': self._export_for_excel_template
        }
        
        if tool_type not in supported_tools:
            return {"error": f"Unsupported tool type: {tool_type}"}
        
        if pool_id not in self.pool_manager.pools:
            return {"error": "Pool not found"}
        
        try:
            result = await supported_tools[tool_type](pool_id, output_path)
            logger.info(f"Successfully exported pool {pool_id} for {tool_type}")
            return result
            
        except Exception as e:
            logger.error(f"External tool export failed for {tool_type}: {e}")
            return {"error": f"Export failed: {str(e)}"}
    
    async def _prepare_pool_export_data(
        self,
        pool: StockPool,
        include_history: bool,
        include_analytics: bool
    ) -> Dict[str, Any]:
        """Prepare comprehensive pool data for export"""
        
        export_data = {
            "pool_info": {
                "pool_id": pool.pool_id,
                "name": pool.name,
                "pool_type": pool.pool_type.value,
                "description": pool.description,
                "created_date": pool.created_date.isoformat(),
                "last_modified": pool.last_modified.isoformat(),
                "status": pool.status.value,
                "max_stocks": pool.max_stocks,
                "rebalance_frequency": pool.rebalance_frequency
            },
            "stocks": [
                {
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "added_date": stock.added_date.isoformat(),
                    "added_price": stock.added_price,
                    "current_price": stock.current_price,
                    "weight": stock.weight,
                    "notes": stock.notes,
                    "tags": stock.tags,
                    "return_pct": ((stock.current_price - stock.added_price) / stock.added_price * 100) if stock.added_price > 0 else 0
                }
                for stock in pool.stocks
            ],
            "metrics": asdict(pool.metrics),
            "auto_update_rules": pool.auto_update_rules
        }
        
        # Add analytics if requested
        if include_analytics:
            analytics = await self.pool_manager.get_pool_analytics(pool.pool_id)
            if "error" not in analytics:
                export_data["analytics"] = analytics
        
        # Add history if requested
        if include_history:
            history = await self.pool_manager.get_pool_history(pool.pool_id)
            export_data["history"] = history
        
        return export_data
    
    async def _export_to_json(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data to JSON format"""
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    
    async def _export_to_csv(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data to CSV format"""
        
        # Create CSV with stock data
        stocks_data = data.get("stocks", [])
        
        if stocks_data:
            df = pd.DataFrame(stocks_data)
            df.to_csv(output_path, index=False)
        else:
            # Create empty CSV with headers
            headers = ["symbol", "name", "added_date", "added_price", "current_price", "weight", "notes", "tags", "return_pct"]
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        return True
    
    async def _export_to_excel(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data to Excel format with multiple sheets"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Pool info sheet
            pool_info_df = pd.DataFrame([data["pool_info"]])
            pool_info_df.to_excel(writer, sheet_name='Pool Info', index=False)
            
            # Stocks sheet
            if data.get("stocks"):
                stocks_df = pd.DataFrame(data["stocks"])
                stocks_df.to_excel(writer, sheet_name='Stocks', index=False)
            
            # Metrics sheet
            if data.get("metrics"):
                metrics_df = pd.DataFrame([data["metrics"]])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Analytics sheet (if available)
            if data.get("analytics"):
                analytics_summary = {
                    "total_stocks": data["analytics"]["basic_info"]["total_stocks"],
                    "total_return": data["analytics"]["performance_metrics"]["total_return"],
                    "volatility": data["analytics"]["performance_metrics"]["volatility"],
                    "sharpe_ratio": data["analytics"]["performance_metrics"]["sharpe_ratio"]
                }
                analytics_df = pd.DataFrame([analytics_summary])
                analytics_df.to_excel(writer, sheet_name='Analytics', index=False)
        
        return True
    
    async def _export_to_xml(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data to XML format"""
        
        import xml.etree.ElementTree as ET
        
        root = ET.Element("StockPool")
        
        # Pool info
        pool_info = ET.SubElement(root, "PoolInfo")
        for key, value in data["pool_info"].items():
            elem = ET.SubElement(pool_info, key)
            elem.text = str(value)
        
        # Stocks
        stocks = ET.SubElement(root, "Stocks")
        for stock in data.get("stocks", []):
            stock_elem = ET.SubElement(stocks, "Stock")
            for key, value in stock.items():
                elem = ET.SubElement(stock_elem, key)
                elem.text = str(value)
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return True
    
    async def _export_to_yaml(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data to YAML format"""
        
        import yaml
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, default=str)
        
        return True
    
    async def _import_from_json(self, file_path: str) -> Dict[str, Any]:
        """Import data from JSON format"""
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    async def _import_from_csv(self, file_path: str) -> Dict[str, Any]:
        """Import data from CSV format"""
        
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to pool format
        stocks = []
        for _, row in df.iterrows():
            stock = {
                "symbol": row.get("symbol", ""),
                "name": row.get("name", ""),
                "added_date": row.get("added_date", datetime.now().isoformat()),
                "added_price": float(row.get("added_price", 0)),
                "current_price": float(row.get("current_price", 0)),
                "weight": float(row.get("weight", 0)),
                "notes": row.get("notes", ""),
                "tags": str(row.get("tags", "")).split(",") if row.get("tags") else []
            }
            stocks.append(stock)
        
        return {
            "pool_info": {
                "name": f"Imported Pool {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "pool_type": "custom",
                "description": f"Imported from CSV on {datetime.now().isoformat()}"
            },
            "stocks": stocks
        }
    
    async def _import_from_excel(self, file_path: str) -> Dict[str, Any]:
        """Import data from Excel format"""
        
        excel_file = pd.ExcelFile(file_path)
        
        import_data = {}
        
        # Read pool info if available
        if 'Pool Info' in excel_file.sheet_names:
            pool_info_df = pd.read_excel(file_path, sheet_name='Pool Info')
            if not pool_info_df.empty:
                import_data["pool_info"] = pool_info_df.iloc[0].to_dict()
        
        # Read stocks data
        if 'Stocks' in excel_file.sheet_names:
            stocks_df = pd.read_excel(file_path, sheet_name='Stocks')
            import_data["stocks"] = stocks_df.to_dict('records')
        
        return import_data
    
    async def _import_from_xml(self, file_path: str) -> Dict[str, Any]:
        """Import data from XML format"""
        
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        import_data = {}
        
        # Parse pool info
        pool_info_elem = root.find('PoolInfo')
        if pool_info_elem is not None:
            pool_info = {}
            for child in pool_info_elem:
                pool_info[child.tag] = child.text
            import_data["pool_info"] = pool_info
        
        # Parse stocks
        stocks_elem = root.find('Stocks')
        if stocks_elem is not None:
            stocks = []
            for stock_elem in stocks_elem.findall('Stock'):
                stock = {}
                for child in stock_elem:
                    stock[child.tag] = child.text
                stocks.append(stock)
            import_data["stocks"] = stocks
        
        return import_data
    
    async def _import_from_yaml(self, file_path: str) -> Dict[str, Any]:
        """Import data from YAML format"""
        
        import yaml
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def _validate_import_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imported data"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if "pool_info" not in data:
            validation_result["errors"].append("Missing pool_info section")
        
        if "stocks" not in data:
            validation_result["errors"].append("Missing stocks section")
        
        # Validate stocks data
        if "stocks" in data:
            for i, stock in enumerate(data["stocks"]):
                if not stock.get("symbol"):
                    validation_result["errors"].append(f"Stock {i}: Missing symbol")
                
                try:
                    float(stock.get("added_price", 0))
                    float(stock.get("current_price", 0))
                except (ValueError, TypeError):
                    validation_result["errors"].append(f"Stock {i}: Invalid price data")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    async def _create_pool_from_import_data(
        self,
        data: Dict[str, Any],
        merge_strategy: str
    ) -> Dict[str, Any]:
        """Create pool from imported data"""
        
        pool_info = data.get("pool_info", {})
        stocks_data = data.get("stocks", [])
        
        # Create new pool
        pool_id = await self.pool_manager.create_pool(
            name=pool_info.get("name", f"Imported Pool {datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            pool_type=pool_info.get("pool_type", "custom"),
            description=pool_info.get("description", "Imported pool"),
            max_stocks=pool_info.get("max_stocks", 100)
        )
        
        # Add stocks
        added_stocks = 0
        for stock_data in stocks_data:
            success = await self.pool_manager.add_stock_to_pool(
                pool_id=pool_id,
                symbol=stock_data.get("symbol", ""),
                name=stock_data.get("name", ""),
                weight=float(stock_data.get("weight", 0)),
                notes=stock_data.get("notes", ""),
                tags=stock_data.get("tags", [])
            )
            
            if success:
                added_stocks += 1
        
        return {
            "success": True,
            "pool_id": pool_id,
            "added_stocks": added_stocks,
            "total_stocks": len(stocks_data)
        }
    
    async def _export_for_portfolio_visualizer(self, pool_id: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Export for Portfolio Visualizer format"""
        
        pool = self.pool_manager.pools[pool_id]
        
        # Portfolio Visualizer expects specific CSV format
        data = []
        for stock in pool.stocks:
            data.append({
                'Symbol': stock.symbol,
                'Weight': stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
            })
        
        df = pd.DataFrame(data)
        
        if not output_path:
            output_path = f"portfolio_visualizer_{pool.name}.csv"
        
        df.to_csv(output_path, index=False)
        
        return {
            "success": True,
            "file_path": output_path,
            "format": "portfolio_visualizer_csv"
        }
    
    async def _export_for_morningstar(self, pool_id: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Export for Morningstar format"""
        
        pool = self.pool_manager.pools[pool_id]
        
        # Morningstar portfolio format
        data = []
        for stock in pool.stocks:
            data.append({
                'Ticker': stock.symbol,
                'Company Name': stock.name,
                'Shares': 100,  # Mock shares
                'Price': stock.current_price,
                'Market Value': stock.current_price * 100,
                'Weight': stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
            })
        
        df = pd.DataFrame(data)
        
        if not output_path:
            output_path = f"morningstar_{pool.name}.csv"
        
        df.to_csv(output_path, index=False)
        
        return {
            "success": True,
            "file_path": output_path,
            "format": "morningstar_csv"
        }
    
    async def _export_for_yahoo_finance(self, pool_id: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Export for Yahoo Finance format"""
        
        pool = self.pool_manager.pools[pool_id]
        
        # Simple symbol list for Yahoo Finance
        symbols = [stock.symbol for stock in pool.stocks]
        
        if not output_path:
            output_path = f"yahoo_finance_{pool.name}.txt"
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(symbols))
        
        return {
            "success": True,
            "file_path": output_path,
            "format": "yahoo_finance_symbols"
        }
    
    async def _export_for_google_sheets(self, pool_id: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Export for Google Sheets format"""
        
        pool = self.pool_manager.pools[pool_id]
        
        # Google Sheets compatible format
        data = []
        data.append(['Symbol', 'Name', 'Weight', 'Current Price', 'Return %', 'Notes'])
        
        for stock in pool.stocks:
            return_pct = ((stock.current_price - stock.added_price) / stock.added_price * 100) if stock.added_price > 0 else 0
            data.append([
                stock.symbol,
                stock.name,
                stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks),
                stock.current_price,
                return_pct,
                stock.notes
            ])
        
        if not output_path:
            output_path = f"google_sheets_{pool.name}.csv"
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        return {
            "success": True,
            "file_path": output_path,
            "format": "google_sheets_csv"
        }
    
    async def _export_for_excel_template(self, pool_id: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Export as Excel template with formulas"""
        
        pool = self.pool_manager.pools[pool_id]
        
        if not output_path:
            output_path = f"excel_template_{pool.name}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Portfolio sheet
            portfolio_data = []
            for i, stock in enumerate(pool.stocks, 1):
                return_pct = ((stock.current_price - stock.added_price) / stock.added_price * 100) if stock.added_price > 0 else 0
                portfolio_data.append({
                    'Symbol': stock.symbol,
                    'Name': stock.name,
                    'Shares': 100,  # Mock
                    'Purchase Price': stock.added_price,
                    'Current Price': stock.current_price,
                    'Market Value': f'=C{i+1}*E{i+1}',  # Formula
                    'Return %': return_pct,
                    'Weight': stock.weight if stock.weight > 0 else 1.0 / len(pool.stocks)
                })
            
            df = pd.DataFrame(portfolio_data)
            df.to_excel(writer, sheet_name='Portfolio', index=False)
            
            # Summary sheet with formulas
            summary_data = {
                'Metric': ['Total Value', 'Total Return %', 'Number of Stocks'],
                'Value': [f'=SUM(Portfolio!F:F)', f'=AVERAGE(Portfolio!G:G)', len(pool.stocks)]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return {
            "success": True,
            "file_path": output_path,
            "format": "excel_template"
        }